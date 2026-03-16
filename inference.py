"""End-to-end ear landmark inference pipeline.

BlazeEar detector (128x128) -> ear ROI crop -> EarLandmarker (192x192) -> 55 landmarks

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --webcam
    python inference.py --webcam --camera 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.ear_landmarker import EarLandmarker

# BlazeEar lives in a sibling directory
BLAZEEAR_DIR = Path("C:/Users/shame/OneDrive/Desktop/ear_stuff/BlazeEar")
PROJECT = Path("C:/Users/shame/OneDrive/Desktop/ear_stuff/Ear Landmarker")

LANDMARKER_INPUT_SIZE = 192
DETECTOR_INPUT_SIZE = 128
ROI_EXPAND = 1.3  # expand detected bbox by 30% for context
NMS_IOU_THRESH = 0.3  # suppress duplicate detections on same ear


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
        """
        detections = self.model.process(frame_rgb)
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        return detections


# ---------------------------------------------------------------------------
# EarLandmarker inference wrapper
# ---------------------------------------------------------------------------

class LandmarkPredictor:
    """Wraps EarLandmarker for inference."""

    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = EarLandmarker(num_landmarks=55)

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix from Lightning checkpoint keys
        state = {k.removeprefix("model."): v for k, v in state.items()
                 if k.startswith("model.")} or state
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Predict 55 landmarks on a cropped ear image.

        Args:
            crop_rgb: (H, W, 3) uint8 RGB ear crop.

        Returns:
            (55, 2) float32 landmarks in pixel coords of the crop.
        """
        h, w = crop_rgb.shape[:2]
        tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0), size=(LANDMARKER_INPUT_SIZE, LANDMARKER_INPUT_SIZE),
                               mode="bilinear", align_corners=False)
        tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = tensor.to(self.device)

        out = self.model(tensor)  # (1, 110)
        lm = out.cpu().numpy().reshape(55, 2)
        lm[:, 0] *= w
        lm[:, 1] *= h
        return lm


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class EarLandmarkerPipeline:
    """Full pipeline: detect ears -> crop -> predict landmarks."""

    def __init__(
        self,
        detector_weights: str | Path,
        landmarker_weights: str | Path,
        device: str = "auto",
        detector_confidence: float | None = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = EarDetector(detector_weights, device, detector_confidence)
        self.landmarker = LandmarkPredictor(landmarker_weights, device)

    def __call__(self, frame_rgb: np.ndarray) -> List[dict]:
        """Run full pipeline on an RGB frame.

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

        results = []
        for det in detections:
            ymin, xmin, ymax, xmax, conf = det[:5]

            # Expand bbox for context
            bw, bh = xmax - xmin, ymax - ymin
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            side = max(bw, bh) * ROI_EXPAND
            x1 = max(0, int(cx - side / 2))
            y1 = max(0, int(cy - side / 2))
            x2 = min(w, int(cx + side / 2))
            y2 = min(h, int(cy + side / 2))

            if x2 - x1 < 16 or y2 - y1 < 16:
                continue

            crop = frame_rgb[y1:y2, x1:x2]
            lm_crop = self.landmarker.predict(crop)

            # Map landmarks back to full frame coords
            lm_frame = lm_crop.copy()
            lm_frame[:, 0] += x1
            lm_frame[:, 1] += y1

            results.append({
                "bbox": np.array([ymin, xmin, ymax, xmax]),
                "confidence": float(conf),
                "landmarks": lm_frame,
            })

        return results

    @staticmethod
    def _nms(detections: np.ndarray, iou_thresh: float = NMS_IOU_THRESH) -> np.ndarray:
        """Standard greedy NMS to remove duplicate detections on the same ear."""
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

            order = rest[iou <= iou_thresh]

        return detections[keep]


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

LINESTRIP_RANGES = [(0, 20), (20, 35), (35, 50), (50, 55)]
LINESTRIP_COLORS = [
    (0, 255, 0),    # helix - green
    (255, 128, 0),  # antihelix - orange
    (0, 128, 255),  # concha - blue
    (255, 0, 128),  # tragus - pink
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
