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

from checkpoint_utils import checkpoint_score
from model.ear_landmarker import EarLandmarker, EarLandmarkerHeatmap
from model.smoothing import EarTracker

# BlazeEar lives in a sibling directory
BLAZEEAR_DIR = Path(os.environ.get("BLAZEEAR_DIR", Path(__file__).resolve().parents[1] / "BlazeEar"))
PROJECT = Path(__file__).resolve().parent

LANDMARKER_INPUT_SIZE = 192
DETECTOR_INPUT_SIZE = 256   # must match the checkpoint; see DETECTOR_WEIGHTS
ROI_EXPAND = 1.3  # expand detected bbox by 30% for context
NMS_IOU_THRESH = 0.3  # suppress duplicate detections on same ear
NMS_IOMIN_THRESH = 0.35  # also suppress when overlap covers this much of the SMALLER box
# The right detector gate is not a property of the model, it is a property of
# who consumes the output -- but NOT for the reason first written here, and the
# correction matters enough to record in full.
#
# The original argument was that a false positive is "55 landmarks drawn
# confidently on a cheek", invisible to an automated consumer and therefore
# worth paying recall to avoid. **That premise is empirically false on this
# corpus.** Cropping and looking at 64 sampled false positives across both
# score bands (2026-10-15, scripts/fp_audit.py): 64 of 64 are REAL EARS with
# good landmark fits -- second ears on frontal faces that master.csv's human
# rows do not label. Not one was a cheek, hair or background. A handful of
# genuine non-ear detections do exist (one jacket/face box seen at 0.71 in a
# separate pass) but they are a small minority.
#
# Three consequences:
#   1. The precision column measures LABEL INCOMPLETENESS as much as error, so
#      "FP/image 0.315 -> 0.127" is mostly correct detections being discarded.
#   2. The bias is not uniform across gates -- a lower gate emits more
#      unlabelled-ear detections, so it is penalised harder. Correcting it
#      would move the optimum DOWN, not up. The F1 argmax (0.76) and the F0.5
#      argmax (0.86) both inherit this and should not be treated as optima.
#   3. A higher gate therefore buys FEWER DETECTIONS, not better ones. It
#      discards real ears on both sides of the ledger: the ones it drops as
#      low-confidence, and the ones it was being credited for "removing".
#
# So the profiles below are a choice about HOW MANY detections you want, not
# how accurate they are. Raise the gate only if you have your own reason --
# throughput, review cost, a downstream stage that cannot take duplicates --
# and sweep it on your own corpus, because nothing here supports a higher gate
# on quality grounds.
#
# Recall cost, measured on 6,000 images / 7,075 ears (README, "Choosing the
# stage-2 gate"), and this part is unaffected by the labelling bias because it
# is scored against labelled ears only:
#
#   gate  recall   | 0-32 px band      80+ px band
#   0.70  79.02%   | 33.51%            92.78%
#   0.76  76.38%   | 29.83% (-11.0%)   90.96% ( -2.0%)
#   0.86  64.83%   | 18.17% (-45.8%)   81.16% (-12.5%)
#
# Raising the gate is NOT a neutral trade: it costs small ears ~3.7x harder in
# relative terms, so a high gate biases against small-in-frame subjects --
# children, distant and profile views. That tilt is invisible in any pooled
# metric, so re-measure PER BAND before re-pointing a name.
DETECTOR_PROFILES = {
    # Shipped default and the only gate with evidence behind it. Keeps the most
    # ears, including the small ones the higher gates drop disproportionately.
    # Right for anything a human looks at, and for annotation assist and corpus
    # onboarding, where a spurious box is one click to reject while a missing
    # ear has to be noticed first.
    "interactive": 0.70,
    # A mild reduction in detection count. Was the F1 argmax; that label is
    # withdrawn, since F1 here inherits the broken precision term.
    "fewer": 0.76,
    # A large reduction in detection count, at -14.2 pp of recall and -45.8%
    # of the under-32px band. Only for a consumer with its own reason to want
    # fewer boxes. NOT a quality setting.
    "fewest": 0.86,
}
DETECTOR_PROFILE_DEFAULT = "interactive"

# Unchanged shipped value; `interactive` is the default profile precisely so
# that naming the choice did not alter existing behaviour.
DETECTOR_CONFIDENCE = DETECTOR_PROFILES[DETECTOR_PROFILE_DEFAULT]


def resolve_confidence(value: float | str | None = None) -> float:
    """Turn a profile name, a raw threshold, or None into a gate.

    `None` gives the default profile. A string is looked up in
    DETECTOR_PROFILES. Anything else is taken as an explicit threshold, so a
    caller who has swept their own corpus is never forced through a name.
    """
    if value is None:
        return DETECTOR_PROFILES[DETECTOR_PROFILE_DEFAULT]
    if isinstance(value, str):
        try:
            return DETECTOR_PROFILES[value]
        except KeyError:
            raise ValueError(
                f"unknown detector profile {value!r}; choose from "
                f"{', '.join(sorted(DETECTOR_PROFILES))}, or pass a float "
                f"threshold if you have swept your own corpus"
            ) from None
    gate = float(value)
    if not 0.0 < gate <= 1.0:
        raise ValueError(f"detector gate must be in (0, 1], got {gate}")
    return gate

# BlazeEar v2 ships a two-stage pipeline and the ear model is trained on FACE
# CROPS, not full frames. runs/checkpoints/ still holds the pre-v2 single-stage
# model; it loads through a different backbone and different anchors, so the
# path decides both. Override with --detector-weights.
#
# BlazeEar now ships a 256 px model and its DEFAULT_INPUT_SIZE moved to 256. A
# checkpoint does NOT record the size it was trained for -- the 128 px and
# 256 px files have byte-identical state dict shapes -- so loading one at the
# wrong size does not raise, it degrades: the 128 px model scores 0.7426
# mAP@0.5 at 128 and 0.4577 at 256.
#
# We were never in that state: runs/checkpoints_crop/BlazeEar_best.pth is
# byte-identical to model_weights/BlazeEar_best_256px.pth (md5 50f740f7...),
# because the crop model was retrained in place at 256. The path is now the
# explicitly sized one and input_size is passed rather than inherited, so a
# future retrain at another size cannot pair them silently.
DETECTOR_WEIGHTS = "model_weights/BlazeEar_best_256px.pth"

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
TRAIN_OCCUPANCY = 0.777   # ear extent / crop side. Was measured over the 5,870-sample
                          # iBUG-derived corpus; the shipped model trains on data/manual,
                          # which ingest_manual.py targets at the same mean (achieved
                          # 0.775, sd 0.093), so the constant is unchanged.
ROI_OCC_TOL = 0.06        # skip refinement when occupancy is already this close
ROI_SATURATED = 0.88      # above this the ear is clipped, so measured extent under-reads
ROI_SAT_BOOST = 1.25      # ...so grow more aggressively than the measurement implies
ROI_MAX_REFINE = 1        # refinement passes; 1 lands within 1% of the fixed point

# Average each prediction with the un-mirrored prediction on the mirrored crop.
# ON by default. Measured on the 861-row cross-source test split with the
# shipped fixed_conv_s42 checkpoint:
#   nme_normal        -1.83%  (p=0.009)
#   points over 0.05  -0.57pp (p=1.4e-09)
#   points over 0.10  -0.14pp (p=5.0e-03)
#   mean worst point  -1.73%  (p=1.1e-03)
# Batched, so it is one forward of batch 2 rather than two passes.
#
# WHEN TO TURN IT OFF. The gain is in the mean and the body of the distribution,
# NOT in the far tail. On this checkpoint the rate of rows whose worst point
# exceeds 0.20 went the wrong way, 3.48% -> 4.30% (p=0.039 nominal, and not
# significant after correcting for the five thresholds tested). Ear_3D_Lifter
# measured the same sign independently on a >10mm gross-error gate on renders.
# Two weak signals in different domains agreeing on direction is worth more than
# either p-value, so a consumer whose metric IS the far tail should measure
# before trusting this, and pass flip_fusion=False if it does not pay there.
# On the LEGACY manual_occ_s42 checkpoint fusion helped at every threshold; the
# tail cost appears only once the handedness bias is gone, which is consistent
# with fusion no longer having a systematic offset to cancel.
FLIP_FUSION_DEFAULT = True

# Also run the DETECTOR on the mirrored frame and merge the two box sets.
# OFF by default. This is a different lever from FLIP_FUSION_DEFAULT: that one
# mirrors the crop tensor inside the landmarker, so the detector has already
# run and its output is fixed. Measured on 1200 full frames with human ear
# ground truth, detector only, at IoU>=0.3:
#     as-is frame     80.00%
#     mirrored frame  79.58%
#     union of both   83.75%     <- +3.75 pp
# Neither presentation is better; they are DIFFERENTLY wrong. Per-box agreement
# across presentations is median IoU 0.813, with 6.4% of boxes disagreeing below
# IoU 0.1 -- found in one presentation, essentially absent in the other. No
# amount of averaging inside the landmarker recovers a crop that was never made.
#
# COSTS A SECOND FULL DETECTOR PASS, and the detector is ~68% of pipeline wall
# clock, so expect roughly +68% per frame. That is why it is off by default.
#
# NOT the whole of what a frame-level mirror buys. Ear_3D_Lifter mirrors the
# entire frame, so their ROI refinement loop sees the mirror too and compounds a
# better first pass; refinement fires on 81% of their render views. On
# photographs it never fires (1.000 landmarker calls per ear), so that route is
# worth nothing here and is not implemented.
DETECTOR_FRAME_TTA_DEFAULT = False
# Hard bounds on the expansion, as a multiple of the detector box extent.
# Without these the refinement is a positive feedback loop: a landmarker that
# reports a saturated extent (confused by motion blur, an occluded ear, or a
# false-positive box) grows the crop, the grown value is cached, and the next
# frame seeds from it -- measured at ~1.53x per frame, reaching 38x in eight
# frames with no recovery, since the enlarged crop makes the ear smaller still.
# The upper bound also keeps the crop inside the band where over-wide framing
# was measured to be cheap: drift is ~0 at 1.7x but 17.7% by 2.5x.
ROI_EXPAND_MIN = 1.0
ROI_EXPAND_MAX = 2.5


# ---------------------------------------------------------------------------
# BlazeEar detector wrapper (lightweight, avoids importing full BlazeEar pkg)
# ---------------------------------------------------------------------------

def _as_boxes(detections) -> np.ndarray:
    """BlazeEar.process() output -> (N, 5) [ymin, xmin, ymax, xmax, conf].

    process() returns (N, 17): box in columns 0-3, six BlazeFace-style keypoints
    in 4-15, and confidence in the LAST column. Reading column 4 as the
    confidence -- as this file used to -- picks up a keypoint x-coordinate, in
    the hundreds of pixels rather than [0, 1], which made NMS sort by a keypoint
    and report nonsense scores.
    """
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
    detections = np.asarray(detections, dtype=np.float64)
    if detections.size == 0:
        return np.zeros((0, 5))
    detections = np.atleast_2d(detections)
    if detections.shape[1] > 5:
        detections = np.column_stack([detections[:, :4], detections[:, -1]])
    return detections


class EarDetector:
    """BlazeEar's two-stage detector: BlazeFace on the frame, ears on each crop.

    Asking one 128x128 detector to find an ear in a whole frame gives it a
    median 14-pixel target; cropping to a face first makes it 32. Measured on
    500 annotated full scenes, recall at IoU>=0.3 goes 47.6% -> 78.6%.

    The cost is a recall ceiling -- an ear whose face is missed never reaches
    the second stage, 1.4% of images here -- which BlazeEar measured and found
    not worth a full-frame fallback, since those ears are a median 4.4px.

    Model construction, anchor choice and crop geometry are all deferred to
    BlazeEar's own helpers rather than reimplemented. That matters: v2 replaced
    the folded backbone with a trainable-BatchNorm one AND fitted new ear
    anchors, so a hand-rolled `BlazeEar()` + `generate_anchors(anchor_options)`
    now either fails to load or silently decodes boxes at the wrong scale.

    Pass two_stage=False to run the ear model on the full frame, which is what
    this did before and is only correct for a pre-v2 checkpoint.
    """

    def __init__(self, weights_path: str | Path, device: str = "cpu",
                 confidence: float | str | None = None, two_stage: bool = True,
                 face_weights: str | Path | None = None,
                 legacy_anchors: bool | None = None,
                 input_size: int = DETECTOR_INPUT_SIZE) -> None:
        sys.path.insert(0, str(BLAZEEAR_DIR))
        import make_face_crops                                  # type: ignore
        from evaluate_two_stage import load_ear_model          # type: ignore
        from make_face_crops import crop_window, load_face_detector  # type: ignore
        from utils.config import (                              # type: ignore
            FACE_CROP_EXPAND, FACE_CROP_MAX_FACES, FACE_CROP_THRESHOLD,
        )

        # BlazeEar's scripts run with their repo as the working directory, so
        # its face weights path is relative. Absolutise it rather than chdir,
        # which would be a process-wide side effect from a constructor.
        if not Path(make_face_crops.FACE_WEIGHTS).is_absolute():
            make_face_crops.FACE_WEIGHTS = str(BLAZEEAR_DIR / make_face_crops.FACE_WEIGHTS)

        self.device = torch.device(device)
        self._crop_window = crop_window
        self.expand = FACE_CROP_EXPAND
        self.max_faces = FACE_CROP_MAX_FACES

        # A pre-v2 checkpoint was trained against the original square anchors;
        # decoding it through the fitted ear priors yields wrong-scale boxes
        # rather than an error, so infer it from the path unless told.
        if legacy_anchors is None:
            p = str(weights_path)
            legacy_anchors = ("checkpoints_crop" not in p
                              and "checkpoints_v2" not in p
                              and "model_weights/BlazeEar_best_" not in p)
        self.model = load_ear_model(
            str(weights_path),
            resolve_confidence(confidence),
            str(self.device), legacy_anchors=legacy_anchors,
            input_size=input_size)

        self.two_stage = two_stage
        self.face = load_face_detector(
            score_threshold=FACE_CROP_THRESHOLD, device=str(self.device),
        ) if two_stage else None
        if face_weights is not None and self.face is not None:
            self.face.min_score_thresh = float(FACE_CROP_THRESHOLD)

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Detect ears in an RGB frame.

        Returns:
            (N, 5) array of [ymin, xmin, ymax, xmax, confidence] in pixel coords.
        """
        if not self.two_stage:
            return _as_boxes(self.model.process(frame_rgb))

        faces = self.face.process(frame_rgb)
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        faces = np.atleast_2d(np.asarray(faces)) if np.size(faces) else np.zeros((0, 17))

        out = []
        for face in faces[: self.max_faces]:
            x0, y0, side = self._crop_window(face, self.expand, frame_rgb.shape)
            patch = frame_rgb[y0:y0 + side, x0:x0 + side]
            if patch.size == 0:
                continue
            det = _as_boxes(self.model.process(patch))
            if not len(det):
                continue
            det[:, :4] += np.array([y0, x0, y0, x0], dtype=np.float64)
            out.append(det)
        if not out:
            return np.zeros((0, 5))
        # Overlapping face crops can each report the same ear; the pipeline's
        # own NMS (IoU + IoMin) runs downstream and handles it.
        return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# EarLandmarker inference wrapper
# ---------------------------------------------------------------------------

class LandmarkPredictor:
    """Wraps EarLandmarker for inference."""

    def __init__(self, weights_path: str | Path, device: str = "cpu",
                 flip_fusion: bool = FLIP_FUSION_DEFAULT) -> None:
        self.device = torch.device(device)
        self.flip_fusion = flip_fusion

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
    def predict(self, crop_rgb: np.ndarray, with_confidence: bool = False,
                flip_fusion: bool | None = None):
        """Predict 55 landmarks on a cropped ear image.

        Args:
            crop_rgb: (H, W, 3) uint8 RGB ear crop.
            with_confidence: Also return per-point confidence. Only the heatmap
                architecture can produce it; the GAP head returns None.
            flip_fusion: Average the prediction with the un-mirrored prediction
                on the mirrored crop. None uses the instance default
                (FLIP_FUSION_DEFAULT). See the class docstring for why this is
                on by default and when to turn it off.

        Returns:
            (55, 2) float32 landmarks in pixel coords of the crop, or a
            (landmarks, confidence) tuple when `with_confidence` is set.
        """
        h, w = crop_rgb.shape[:2]
        tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0), size=(LANDMARKER_INPUT_SIZE, LANDMARKER_INPUT_SIZE),
                               mode="bilinear", align_corners=False)
        # [-1, 1], matching data/dataset.py: to_tensor() -> [0,1], then
        # normalize(mean=0.5, std=0.5). Feeding [0,1] instead costs 15.6% test
        # NME (0.0288 -> 0.0341 on 400 held-out samples) and raises no error.
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(self.device)

        fuse = self.flip_fusion if flip_fusion is None else flip_fusion
        if fuse:
            # Both arms in ONE batched forward, so the second costs a fraction
            # of a pass rather than a whole one. Mirroring here -- after the
            # resize, on the normalised tensor -- keeps the two arms pixel-exact
            # mirrors of each other; mirroring the uint8 crop and re-resizing
            # does not, because the resize is not flip-commutative at the edges.
            tensor = torch.cat([tensor, torch.flip(tensor, dims=[3])], dim=0)

        conf = None
        if with_confidence and hasattr(self.model, "predict_with_confidence"):
            out, c = self.model.predict_with_confidence(tensor)
            conf = c.cpu().numpy().reshape(-1, 55).mean(0)
        else:
            out = self.model(tensor)  # (B, 110)

        if fuse:
            arms = out.cpu().numpy().reshape(2, 55, 2)
            # Un-mirror is exactly 1-x: the decoder's x grid is (i+0.5)/w
            # (model/ear_landmarker.py), and data/dataset.py trains the flip as
            # landmarks[:, 0] = 1.0 - landmarks[:, 0] with NO index permutation,
            # so point i is the same anatomical point in both arms. An index
            # remap here would be wrong, and MMPose's shift_heatmap would inject
            # a half-pixel error.
            arms[1, :, 0] = 1.0 - arms[1, :, 0]
            lm = arms.mean(0)
        else:
            lm = out.cpu().numpy().reshape(55, 2)
        lm[:, 0] *= w
        lm[:, 1] *= h
        return (lm, conf) if with_confidence else lm


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def unmirror_boxes(detections: np.ndarray, width: int) -> np.ndarray:
    """Map boxes detected on a mirrored frame back to the original frame.

    Boxes are [ymin, xmin, ymax, xmax, score, ...]. Mirroring sends a column x
    to width - x, which REVERSES the order of the two x edges: the mirrored
    box's xmin corresponds to the original's xmax. Assigning the two columns
    without the swap yields boxes with xmin > xmax, whose area computes
    negative and which then silently survive NMS and match nothing.

    Returns a copy; the input is not modified.
    """
    detections = np.asarray(detections, dtype=np.float64)
    if not len(detections):
        return detections
    out = detections.copy()
    out[:, 1] = width - detections[:, 3]
    out[:, 3] = width - detections[:, 1]
    return out


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
        detector_confidence: float | str | None = None,
        smooth: bool = True,
        smoothing_kwargs: dict | None = None,
        refine_roi: bool = True,
        flip_fusion: bool = FLIP_FUSION_DEFAULT,
        detector_frame_tta: bool = DETECTOR_FRAME_TTA_DEFAULT,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = EarDetector(detector_weights, device, detector_confidence)
        self.landmarker = LandmarkPredictor(landmarker_weights, device,
                                            flip_fusion=flip_fusion)
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
        # See DETECTOR_FRAME_TTA_DEFAULT. Costs a second detector pass.
        self.detector_frame_tta = detector_frame_tta
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

        if self.detector_frame_tta:
            # The detector is presentation-sensitive: the two arms are not
            # better and worse, they fail on different boxes. Merging them is
            # what recovers the 3.75 pp; see DETECTOR_FRAME_TTA_DEFAULT.
            mirrored = unmirror_boxes(
                self.detector.detect(np.ascontiguousarray(frame_rgb[:, ::-1])), w)
            if len(mirrored):
                detections = (np.concatenate([detections, mirrored], axis=0)
                              if len(detections) else mirrored)

        # Suppress duplicate boxes (secondary NMS on denormalized detections).
        # This is also what merges the two presentations when frame TTA is on:
        # a box both arms found is deduplicated, a box only one arm found
        # survives, which is exactly the union the measurement scored.
        detections = self._nms(detections)

        # Smooth the detector boxes before cropping. The jitter benchmark showed
        # box wobble is ~90% of frame-to-frame jitter, so this is where the win
        # is -- a steady crop means the landmarker sees a consistent input.
        track_ids: List[int] = []
        if self.tracker is not None:
            t = time.perf_counter() if timestamp is None else timestamp
            # One call does association and box smoothing (BlazeEar's
            # BoxTracker, vendored). Only matched tracks come back, so a
            # coasting track keeps its filter and landmark state alive across a
            # short dropout without reporting an ear on a frame that had none.
            tracked = self.tracker.update(detections, t)
            track_ids = [tid for tid, _ in tracked]
            detections = [det for _, det in tracked]

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
            expand = min(max(expand, ROI_EXPAND_MIN), ROI_EXPAND_MAX)
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
                side = float(np.clip(refine_roi_side(side, extent),
                                     max(bw, bh) * ROI_EXPAND_MIN,
                                     max(bw, bh) * ROI_EXPAND_MAX))
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

# The run the shipped model comes from. Checkpoint scores are NOT comparable
# across runs -- v5_tang* was trained with a tangential-weighted loss and
# monitored val_nme_normal, so it posts a lower val_nme while scoring WORSE on
# test (0.0296 against 0.0292). Ranking every run together therefore picks a
# model that should not ship, which is exactly what an earlier version of
# find_best_checkpoint() did once it learned to search recursively.
#
# Was v6_persp65, trained on the iBUG-derived corpus. Now manual_occ_s42,
# trained only on the commissioned set (data/manual), which is what lets the
# shipped weights carry Apache-2.0 -- see NOTICE.
#
# It scores WORSE on the old benchmark (0.0347 against 0.0292) and that is not
# the reason to distrust it: 55% of that test split is collectionB, whose
# annotation convention v6_persp65 was trained on and this model was not.
# Excluding collectionB the gap is +3.1%, and on audioear2d +0.9%. Judged
# without any ground truth -- 455 real full frames, contour placement scored by
# image-gradient response -- three seeds of this model beat three seeds of
# v6_persp65 with complete separation (4.0668 vs 3.8886, p=0.05 exact, the
# floor for 3v3). See RESULTS.md.
#
# Now fixed_conv_s42, same recipe and the same data/manual-only provenance, so
# the Apache-2.0 position in NOTICE is unchanged. It differs from
# manual_occ_s42 by one thing: the sub-pixel augmentation fix in
# data/dataset.py (LEGACY_SUBPIXEL_SHIFT). That removes a handedness bias which
# was real and large -- paired within-row, the R-L antisymmetry on the
# handedness-balanced audioear2d source was +17.05% of nme_normal at p=2.7e-19,
# and is now +1.01% at p=0.46. The legacy model was better on RIGHT-presented
# ears and worse on left; side labels come from the canal (point 39) being
# anterior, not from the rim's signed area, whose polarity is not self-evident. Accuracy is a wash (val_nme 0.01452 against
# 0.01487 is one seed on data/manual, the cut that flatters the fix; the
# cross-source split reverses it slightly), so the bias removal is the reason,
# not accuracy.
#
# NOT A STRICT UPGRADE FOR EVERY CONSUMER. On Ear_3D_Lifter's >10mm
# gross-error gate over renders this checkpoint is worse as-is than
# manual_occ_s42, +0.275pp at p=3.2e-04, while their mirrored arm improves by
# about as much. Their corpus presents every ear in left form, where an
# anti-equivariant offset is a constant of fixed sign rather than a handedness
# bias -- so the legacy model carried a constant offset that a single-handed
# pipeline can be benefiting from, and which would reverse for the opposite
# presentation. Which presentation it favours in THAT pipeline's own frame is
# for that pipeline to measure; it does not follow from the side names here.
# A consumer in that position should measure before adopting.
SHIPPED_RUN = "fixed_conv_s42"


def find_best_checkpoint(run: str | None = SHIPPED_RUN) -> Path:
    """Best landmarker checkpoint for one run, by the score inside the file.

    Scoped to `run` by default, because scores are not comparable between runs
    (see SHIPPED_RUN). Pass run=None to search every run, which is only
    meaningful for runs that share a training objective.

    Ranked by the score ModelCheckpoint stores inside the file, not by the nme=
    field in the name: that is rounded to four decimals and every run here has
    two or three checkpoints tied at it, so ranking by name resolved the tie on
    directory order and could return a checkpoint other than the best.
    """
    ckpt_dir = PROJECT / "runs" / "checkpoints"
    search_dir = ckpt_dir / run if run else ckpt_dir
    if run and not search_dir.is_dir():
        raise FileNotFoundError(
            f"Run {run!r} not found under {ckpt_dir}. Pass --checkpoint "
            f"explicitly, or find_best_checkpoint(run=None) to search all runs.")

    candidates = sorted(c for c in search_dir.rglob("*.ckpt") if "nme=" in c.name)
    if not candidates:
        last = search_dir / "last.ckpt"
        if last.exists():
            return last
        raise FileNotFoundError(f"No checkpoints found under {search_dir}")

    return min(candidates, key=checkpoint_score)


def run_image(args: argparse.Namespace) -> None:
    """Run on a single image and display / save result."""
    detector_weights = args.detector_weights or BLAZEEAR_DIR / DETECTOR_WEIGHTS
    landmarker_weights = args.landmarker_weights or find_best_checkpoint()

    pipeline = EarLandmarkerPipeline(
        detector_weights, landmarker_weights,
        device=args.device,
        detector_confidence=(args.confidence if args.confidence is not None
                             else args.profile),
        flip_fusion=args.flip_fusion,
        detector_frame_tta=args.detector_frame_tta,
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
    detector_weights = args.detector_weights or BLAZEEAR_DIR / DETECTOR_WEIGHTS
    landmarker_weights = args.landmarker_weights or find_best_checkpoint()

    pipeline = EarLandmarkerPipeline(
        detector_weights, landmarker_weights,
        device=args.device,
        detector_confidence=(args.confidence if args.confidence is not None
                             else args.profile),
        flip_fusion=args.flip_fusion,
        detector_frame_tta=args.detector_frame_tta,
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
        # Name the consumer, or give a number. Mutually exclusive, because
        # silently letting one win is how a caller ends up running a gate they
        # did not pick.
        gate = p.add_mutually_exclusive_group()
        gate.add_argument("--profile", type=str, default=None,
                          choices=sorted(DETECTOR_PROFILES),
                          help="Who consumes the output, which sets the "
                               "detector gate: "
                               + ", ".join(f"{k}={v:.2f}" for k, v in
                                           sorted(DETECTOR_PROFILES.items()))
                               + f" (default: {DETECTOR_PROFILE_DEFAULT}). "
                                 "A higher gate trades recall for precision "
                                 "and costs small ears disproportionately; "
                                 'see README "Choosing the stage-2 gate".')
        gate.add_argument("--confidence", type=float, default=None,
                          help="Explicit detector gate, overriding --profile. "
                               f"Default: the {DETECTOR_PROFILE_DEFAULT} "
                               f"profile ({DETECTOR_CONFIDENCE:.2f}).")
        p.add_argument("--detector-weights", type=str, default=None,
                       help=f"BlazeEar weights (default: BlazeEar/{DETECTOR_WEIGHTS})")
        p.add_argument("--landmarker-weights", type=str, default=None,
                       help="EarLandmarker weights (default: best checkpoint)")
        p.add_argument("--detector-frame-tta", dest="detector_frame_tta",
                       action="store_true", default=DETECTOR_FRAME_TTA_DEFAULT,
                       help="Also run the detector on the mirrored frame and "
                            "merge the boxes. Worth +3.75pp of detector recall "
                            "and costs a second detector pass (~+68%% per "
                            "frame). Off by default")
        p.add_argument("--no-flip-fusion", dest="flip_fusion",
                       action="store_false", default=FLIP_FUSION_DEFAULT,
                       help="Predict from the crop only, instead of averaging "
                            "with the mirrored crop. Fusion is on by default "
                            "and costs one batched forward; turn it off if your "
                            "metric is the far error tail (see "
                            "FLIP_FUSION_DEFAULT)")

    args = parser.parse_args()
    if args.mode == "image":
        run_image(args)
    elif args.mode == "webcam":
        run_webcam(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
