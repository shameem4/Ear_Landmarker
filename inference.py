"""End-to-end ear landmark inference pipeline.

BlazeEar detector (128x128) -> ear ROI crop -> EarLandmarker (192x192) -> 55 landmarks

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --webcam
    python inference.py --webcam --camera 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.ear_landmarker import EarLandmarker, EarLandmarkerHeatmap
from model.smoothing import EarTracker

# BlazeEar lives in a sibling directory
BLAZEEAR_DIR = Path(os.environ.get("BLAZEEAR_DIR", Path(__file__).resolve().parents[1] / "BlazeEar"))
PROJECT = Path(__file__).resolve().parent

LANDMARKER_INPUT_SIZE = 192
DETECTOR_INPUT_SIZE = 128
ROI_EXPAND = 1.3  # expand detected bbox by 30% for context
NMS_IOU_THRESH = 0.3  # suppress duplicate detections on same ear
NMS_IOMIN_THRESH = 0.35  # also suppress when overlap covers this much of the SMALLER box

# Adaptive ROI refinement.
#
# ROI_EXPAND alone cannot frame the ear correctly, because the detector box is
# not a fixed fraction of the ear: measured on real captures, true ear extent
# over detector box extent ranges 1.02-1.53. At 1.3 expansion a tight box yields
# a crop SMALLER than the ear, so the landmarks jam against the crop border and
# can never reach the rim.
#
# So the ROI is re-derived from the landmarks, which do know where the ear is:
# crop, predict, measure the extent, and re-crop so the ear sits at the
# occupancy the model was trained on. This is the ROI-from-landmarks refinement
# MediaPipe uses for face and hand tracking.
TRAIN_OCCUPANCY = 0.777   # ear extent / crop side, measured over all 5,870 training samples
ROI_OCC_TOL = 0.06        # skip refinement when occupancy is already this close
ROI_SATURATED = 0.88      # above this the ear is clipped, so measured extent under-reads
ROI_SAT_BOOST = 1.25      # ...so grow more aggressively than the measurement implies
ROI_MAX_REFINE = 1        # refinement passes; 1 lands within 1% of the fixed point


# ---------------------------------------------------------------------------
# BlazeEar detector wrapper (lightweight, avoids importing full BlazeEar pkg)
# ---------------------------------------------------------------------------

class EarDetector:
    """Wraps BlazeEar for detection-only use."""

    def __init__(self, weights_path: str | Path, device: str = "cpu",
                 confidence: float | None = None) -> None:
        sys.path.insert(0, str(BLAZEEAR_DIR))
        from blazeear import BlazeEar  # type: ignore
        from utils.anchor_utils import anchor_options  # type: ignore

        self.device = torch.device(device)
        self.model = BlazeEar()
        if confidence is not None:
            self.model.min_score_thresh = confidence

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        # Generate anchors after .to(device) so they land on the right device
        self.model.generate_anchors(anchor_options)

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Detect ears in an RGB frame.

        Returns:
            (N, 5) array of [ymin, xmin, ymax, xmax, confidence] in pixel coords.

        BlazeEar.process() actually returns (N, 17): box in columns 0-3, six
        BlazeFace-style keypoints in 4-15, and confidence in the LAST column.
        Reading column 4 as the confidence -- as this file used to -- picks up a
        keypoint x-coordinate instead, which is in the hundreds of pixels rather
        than [0, 1]. That made NMS sort by a keypoint and report nonsense scores.
        Normalise here so the rest of the pipeline sees the documented shape.
        """
        detections = self.model.process(frame_rgb)
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        detections = np.asarray(detections, dtype=np.float64)
        if detections.size == 0:
            return np.zeros((0, 5))
        detections = np.atleast_2d(detections)
        if detections.shape[1] > 5:
            detections = np.column_stack([detections[:, :4], detections[:, -1]])
        return detections


# ---------------------------------------------------------------------------
# EarLandmarker inference wrapper
# ---------------------------------------------------------------------------

class LandmarkPredictor:
    """Wraps EarLandmarker for inference."""

    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        # Architecture comes from the checkpoint's saved hyperparameters, so a
        # heatmap-head checkpoint loads without passing a matching flag.
        hparams = ckpt.get("hyper_parameters", {}) or {}
        if hparams.get("arch", "gap") == "heatmap":
            self.model = EarLandmarkerHeatmap(
                num_landmarks=55, tau=hparams.get("tau", 1.0),
            )
        else:
            self.model = EarLandmarker(num_landmarks=55)

        state = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix from Lightning checkpoint keys
        state = {k.removeprefix("model."): v for k, v in state.items()
                 if k.startswith("model.")} or state
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, crop_rgb: np.ndarray, with_confidence: bool = False):
        """Predict 55 landmarks on a cropped ear image.

        Args:
            crop_rgb: (H, W, 3) uint8 RGB ear crop.
            with_confidence: Also return per-point confidence. Only the heatmap
                architecture can produce it; the GAP head returns None.

        Returns:
            (55, 2) float32 landmarks in pixel coords of the crop, or a
            (landmarks, confidence) tuple when `with_confidence` is set.
        """
        h, w = crop_rgb.shape[:2]
        tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0), size=(LANDMARKER_INPUT_SIZE, LANDMARKER_INPUT_SIZE),
                               mode="bilinear", align_corners=False)
        tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = tensor.to(self.device)

        conf = None
        if with_confidence and hasattr(self.model, "predict_with_confidence"):
            out, c = self.model.predict_with_confidence(tensor)
            conf = c.cpu().numpy().reshape(-1)
        else:
            out = self.model(tensor)  # (1, 110)

        lm = out.cpu().numpy().reshape(55, 2)
        lm[:, 0] *= w
        lm[:, 1] *= h
        return (lm, conf) if with_confidence else lm


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def square_roi_crop(frame_rgb: np.ndarray, cx: float, cy: float, side: float,
                    pad_value: int = 128):
    """Crop a SQUARE ROI centred on (cx, cy), padding outside the frame.

    Clamping the window to the frame instead -- which is what this used to do --
    yields a non-square crop, and resizing that to the model's 192x192 input
    stretches the ear along one axis. The model never saw that distortion in
    training, so accuracy drops: measured on held-out samples with a 20% edge
    overlap, padding beats clamping by 17% NME, on 89% of samples. 15% of real
    ears sit close enough to a frame edge for this to matter.

    Grey 128 is the same fill data/dataset.py uses when rotation or translation
    augmentation exposes area outside the source image, so padded regions look
    to the model like padding it was trained through.

    Returns (crop, x1, y1) where (x1, y1) is the ROI origin in frame
    coordinates -- possibly negative -- or None if the ROI is unusable.
    """
    h, w = frame_rgb.shape[:2]
    n = int(round(side))
    if n < 16:
        return None

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))

    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x1 + n), min(h, y1 + n)
    if sx2 - sx1 < 8 or sy2 - sy1 < 8:
        return None

    crop = np.full((n, n, 3), pad_value, dtype=frame_rgb.dtype)
    crop[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame_rgb[sy1:sy2, sx1:sx2]
    return crop, x1, y1


def refine_roi_side(side: float, extent: float) -> float:
    """Next ROI side so the ear lands at the training occupancy.

    `extent` is the landmark bounding extent measured in the same units as
    `side`. Normally the answer is just extent / TRAIN_OCCUPANCY. But when the
    ear is clipped by the crop, the measured extent is bounded by the crop
    itself and under-reads the true ear, so that update converges slowly. Above
    ROI_SATURATED the crop is treated as clipped and grown by a boosted factor
    instead, which reaches the fixed point in one pass rather than two.
    """
    occ = extent / side if side > 0 else 0.0
    if occ > ROI_SATURATED:
        return side * (occ / TRAIN_OCCUPANCY) * ROI_SAT_BOOST
    return extent / TRAIN_OCCUPANCY


class EarLandmarkerPipeline:
    """Full pipeline: detect ears -> crop -> predict landmarks."""

    def __init__(
        self,
        detector_weights: str | Path,
        landmarker_weights: str | Path,
        device: str = "auto",
        detector_confidence: float | None = None,
        smooth: bool = True,
        smoothing_kwargs: dict | None = None,
        refine_roi: bool = True,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = EarDetector(detector_weights, device, detector_confidence)
        self.landmarker = LandmarkPredictor(landmarker_weights, device)
        # Defaults tuned on scripts/eval_smoothing.py: these cut jitter ~66-80%
        # while also REDUCING error against a perfect-detector reference, in both
        # the static and realistic-motion scenarios.
        self.tracker = EarTracker(**(smoothing_kwargs or {
            "landmark_kwargs": {"min_cutoff": 3.0, "beta": 0.4, "conf_strength": 0.5},
            "box_kwargs": {"min_cutoff": 3.0, "beta": 0.4,
                           "size_min_cutoff": 1.5, "size_beta": 0.2},
        })) if smooth else None
        # Re-derive the ROI from the landmarks rather than trusting the detector
        # box; see the ROI_* constants. Pass refine_roi=False for the old
        # single-pass behaviour.
        self.refine_roi = refine_roi
        self._roi_expand: dict[int, float] = {}

    def reset(self) -> None:
        """Clear smoothing and ROI state, e.g. when switching video sources.

        Required, not optional: EarTracker.reset() restarts track ids at 0, so a
        cached ROI expansion keyed by id would be inherited by an unrelated ear
        in the next source.
        """
        if self.tracker is not None:
            self.tracker.reset()
        self._roi_expand.clear()

    def __call__(self, frame_rgb: np.ndarray, timestamp: float | None = None) -> List[dict]:
        """Run full pipeline on an RGB frame.

        Args:
            frame_rgb: (H, W, 3) uint8 RGB frame.
            timestamp: Monotonic time in seconds. Required for smoothing; when
                the tracker is enabled and this is None, wall-clock is used.

        Returns:
            List of dicts, each with:
                "bbox": (4,) [ymin, xmin, ymax, xmax] in pixels
                "confidence": float
                "landmarks": (55, 2) in original frame pixel coords
        """
        h, w = frame_rgb.shape[:2]
        detections = self.detector.detect(frame_rgb)

        # Suppress duplicate boxes (secondary NMS on denormalized detections)
        detections = self._nms(detections)

        # Smooth the detector boxes before cropping. The jitter benchmark showed
        # box wobble is ~90% of frame-to-frame jitter, so this is where the win
        # is -- a steady crop means the landmarker sees a consistent input.
        track_ids: List[int] = []
        if self.tracker is not None:
            t = time.perf_counter() if timestamp is None else timestamp
            boxes = [np.asarray(d[:4], dtype=np.float64) for d in detections]
            track_ids = self.tracker.assign(boxes)
            detections = [
                np.concatenate([self.tracker.smooth_box(tid, b, t), d[4:]])
                for tid, b, d in zip(track_ids, boxes, detections)
            ]

        results = []
        live: set[int] = set()
        for k, det in enumerate(detections):
            ymin, xmin, ymax, xmax, conf = det[:5]  # normalised in EarDetector.detect

            # Expand bbox for context
            bw, bh = xmax - xmin, ymax - ymin
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

            # Seed the ROI from what this track needed last frame, so video
            # settles to one pass instead of paying for refinement every frame.
            tid = track_ids[k] if track_ids else None
            expand = self._roi_expand.get(tid, ROI_EXPAND) if tid is not None else ROI_EXPAND
            side = max(bw, bh) * expand

            lm_frame = point_conf = None
            for attempt in range(ROI_MAX_REFINE + 1):
                got = square_roi_crop(frame_rgb, cx, cy, side)
                if got is None:
                    break
                crop, x1, y1 = got
                lm_crop, point_conf = self.landmarker.predict(crop, with_confidence=True)

                # Map landmarks back to full frame coords
                lm_frame = lm_crop.copy()
                lm_frame[:, 0] += x1
                lm_frame[:, 1] += y1

                if not self.refine_roi or attempt == ROI_MAX_REFINE:
                    break
                extent = float(max(np.ptp(lm_frame[:, 0]), np.ptp(lm_frame[:, 1])))
                if abs(extent / side - TRAIN_OCCUPANCY) <= ROI_OCC_TOL:
                    break  # already framed the way the model was trained
                side = refine_roi_side(side, extent)
                cx = float((lm_frame[:, 0].min() + lm_frame[:, 0].max()) / 2)
                cy = float((lm_frame[:, 1].min() + lm_frame[:, 1].max()) / 2)

            if lm_frame is None:
                continue
            if tid is not None and max(bw, bh) > 0:
                self._roi_expand[tid] = side / max(bw, bh)
                live.add(tid)

            # Smooth in frame coords, so the filter sees real motion rather than
            # motion induced by the crop moving underneath it.
            if self.tracker is not None:
                t = time.perf_counter() if timestamp is None else timestamp
                lm_frame = self.tracker.smooth_landmarks(
                    track_ids[k], lm_frame, t, point_conf,
                )

            results.append({
                "bbox": np.array([ymin, xmin, ymax, xmax]),
                "confidence": float(conf),
                "landmarks": lm_frame,
            })

        # Track ids increment forever, so without this the cache grows for the
        # life of the process on any stream where ears come and go.
        if track_ids:
            for tid in [k for k in self._roi_expand if k not in live]:
                del self._roi_expand[tid]

        return results

    @staticmethod
    def _nms(detections: np.ndarray, iou_thresh: float = NMS_IOU_THRESH,
             io_min_thresh: float = NMS_IOMIN_THRESH) -> np.ndarray:
        """Greedy NMS removing duplicate detections on the same ear.

        Suppresses on EITHER IoU or intersection-over-minimum; see the comment
        at the io_min computation for why IoU alone is not enough.
        """
        if len(detections) <= 1:
            return detections

        # detections: (N, 5+) with [ymin, xmin, ymax, xmax, conf, ...]
        scores = detections[:, 4]
        order = scores.argsort()[::-1]

        y1 = detections[:, 0]
        x1 = detections[:, 1]
        y2 = detections[:, 2]
        x2 = detections[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            rest = order[1:]
            inter_y1 = np.maximum(y1[i], y1[rest])
            inter_x1 = np.maximum(x1[i], x1[rest])
            inter_y2 = np.minimum(y2[i], y2[rest])
            inter_x2 = np.minimum(x2[i], x2[rest])
            inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
            union = areas[i] + areas[rest] - inter
            iou = inter / np.maximum(union, 1e-6)

            # Plain IoU misses the duplicates this detector actually produces on
            # a single ear: a small box inside a larger one scores
            # IoU = areaSmall/areaLarge, which drops below any sane threshold
            # once the larger box is ~3x the smaller, so both survive and two
            # landmark sets get drawn on one ear. Intersection-over-minimum
            # catches containment, and stays ~0 for two genuinely different
            # ears, which are far apart in frame.
            min_area = np.minimum(areas[i], areas[rest])
            io_min = inter / np.maximum(min_area, 1e-6)

            order = rest[(iou <= iou_thresh) & (io_min <= io_min_thresh)]

        return detections[keep]


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

LINESTRIP_RANGES = [(0, 20), (20, 35), (35, 50), (50, 55)]
LINESTRIP_COLORS = [
    (0, 255, 0),    # outer helix + lobe (0-19)    - green
    (255, 128, 0),  # inner helix (20-34)          - orange
    (0, 128, 255),  # concha border (35-49)        - blue
    (255, 0, 128),  # superior crus (50-54)        - pink
]


def draw_results(frame_bgr: np.ndarray, results: List[dict]) -> np.ndarray:
    """Draw bounding boxes and landmarks on a BGR frame."""
    out = frame_bgr.copy()
    for r in results:
        ymin, xmin, ymax, xmax = r["bbox"].astype(int)
        conf = r["confidence"]
        lm = r["landmarks"]

        # Bbox
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Landmarks as connected linestrips
        for (start, end), color in zip(LINESTRIP_RANGES, LINESTRIP_COLORS):
            pts = lm[start:end].astype(int)
            for j in range(len(pts) - 1):
                cv2.line(out, tuple(pts[j]), tuple(pts[j + 1]), color, 1)
            for pt in pts:
                cv2.circle(out, tuple(pt), 2, color, -1)

    return out


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def find_best_checkpoint() -> Path:
    """Find the best landmarker checkpoint by NME in filename."""
    ckpt_dir = PROJECT / "runs" / "checkpoints"
    # Search both flat and nested (old val/nme created subdirs on Windows)
    candidates = list(ckpt_dir.glob("EarLandmarker_*.ckpt"))
    candidates += list(ckpt_dir.rglob("nme=*.ckpt"))
    # Also check for last.ckpt
    last = ckpt_dir / "last.ckpt"
    if not candidates and last.exists():
        return last
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    best = candidates[0]
    best_nme = 999.0
    for c in candidates:
        try:
            nme = float(c.stem.split("nme=")[1])
            if nme < best_nme:
                best_nme = nme
                best = c
        except (IndexError, ValueError):
            continue
    return best


def run_image(args: argparse.Namespace) -> None:
    """Run on a single image and display / save result."""
    detector_weights = args.detector_weights or BLAZEEAR_DIR / "runs/checkpoints/BlazeEar_best.pth"
    landmarker_weights = args.landmarker_weights or find_best_checkpoint()

    pipeline = EarLandmarkerPipeline(
        detector_weights, landmarker_weights,
        device=args.device, detector_confidence=args.confidence,
    )

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pipeline(rgb)
    print(f"Detected {len(results)} ear(s)")

    out = draw_results(img, results)

    if args.output:
        cv2.imwrite(args.output, out)
        print(f"Saved to {args.output}")
    else:
        cv2.imshow("Ear Landmarks", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(args: argparse.Namespace) -> None:
    """Run on webcam with real-time display."""
    detector_weights = args.detector_weights or BLAZEEAR_DIR / "runs/checkpoints/BlazeEar_best.pth"
    landmarker_weights = args.landmarker_weights or find_best_checkpoint()

    pipeline = EarLandmarkerPipeline(
        detector_weights, landmarker_weights,
        device=args.device, detector_confidence=args.confidence,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("Press 'q' to quit")
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pipeline(rgb)
        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps

        out = draw_results(frame, results)
        cv2.putText(out, f"FPS: {fps_smooth:.0f} | Ears: {len(results)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Ear Landmarker", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ear Landmarker inference")
    sub = parser.add_subparsers(dest="mode")

    # Image mode
    img_parser = sub.add_parser("image", help="Run on a single image")
    img_parser.add_argument("image", type=str, help="Path to input image")
    img_parser.add_argument("--output", type=str, default=None, help="Save output to file")

    # Webcam mode
    cam_parser = sub.add_parser("webcam", help="Run on webcam")
    cam_parser.add_argument("--camera", type=int, default=0, help="Camera index")

    # Shared args
    for p in [img_parser, cam_parser]:
        p.add_argument("--device", type=str, default="auto")
        p.add_argument("--confidence", type=float, default=None,
                       help="Detector confidence threshold (default: use BlazeEar model default)")
        p.add_argument("--detector-weights", type=str, default=None,
                       help="BlazeEar weights (default: BlazeEar/runs/checkpoints/BlazeEar_best.pth)")
        p.add_argument("--landmarker-weights", type=str, default=None,
                       help="EarLandmarker weights (default: best checkpoint)")

    args = parser.parse_args()
    if args.mode == "image":
        run_image(args)
    elif args.mode == "webcam":
        run_webcam(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
