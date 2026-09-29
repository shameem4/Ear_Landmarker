"""Temporal smoothing for the live pipeline: landmarks and detector boxes.

Why this is the right lever: the jitter benchmark showed that ~90-95% of
frame-to-frame jitter comes from crop-window (detector bbox) wobble, and that no
landmarker-side change moved it much (-8.8% end to end). Smoothing is what
actually attacks it.

Why One Euro rather than a plain EMA: a fixed-alpha EMA trades jitter for lag at
a single fixed ratio, so the only way to kill jitter is to make the output lag
visibly during real motion. The One Euro filter adapts its cutoff to the observed
speed -- heavy smoothing when the subject is still, light smoothing when moving --
which is the standard approach for real-time landmark streams and what MediaPipe
uses for the same job.

Reference: Casiez, Roussel & Vogel, "1 euro filter: a simple speed-based
low-pass filter for noisy input in interactive systems", CHI 2012.

Both filters are timestamp-driven, so they behave correctly at variable frame
rates rather than assuming a fixed dt.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OneEuroFilter", "LandmarkSmoother", "BoxSmoother", "EarTracker"]


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    """Low-pass smoothing factor for a given cutoff frequency and timestep."""
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Speed-adaptive low-pass filter, applied elementwise over an array.

    Args:
        min_cutoff: Cutoff frequency (Hz) at zero speed. Lower = smoother when
            still, but slower to respond.
        beta: Speed coefficient. Higher = follows fast motion more eagerly
            (less lag) at the cost of passing more jitter during motion.
        d_cutoff: Cutoff for the derivative estimate itself.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None

    def __call__(
        self,
        x: np.ndarray,
        t: float,
        cutoff_scale: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Filter one observation.

        Args:
            x: Observation, any shape (filtered elementwise).
            t: Timestamp in seconds (monotonic).
            cutoff_scale: Per-element multiplier on min_cutoff. Values < 1 smooth
                harder. Used to smooth low-confidence landmarks more aggressively.

        Returns:
            The filtered array, same shape as x.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        dt = t - self._t_prev
        if dt <= 0:                      # duplicate/out-of-order frame
            return self._x_prev.copy()

        # Low-pass the derivative, then set the cutoff from the smoothed speed.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff * np.asarray(cutoff_scale) + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat.copy()


class LandmarkSmoother:
    """One Euro over (N, 2) landmarks, optionally weighted by per-point confidence.

    Confidence comes from the heatmap head's spatial softmax: a peaked
    distribution means the model is sure, a diffuse one means it is guessing.
    Uncertain points are smoothed harder, which is the whole reason to prefer the
    heatmap architecture for a live demo.

    Args:
        min_cutoff, beta, d_cutoff: See OneEuroFilter.
        conf_strength: How much confidence modulates the cutoff. 0 disables it;
            1 means a zero-confidence point gets `conf_floor` of the base cutoff.
        conf_floor: Lower bound on the confidence multiplier, so a low-confidence
            point is still allowed to move eventually.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
        conf_strength: float = 1.0,
        conf_floor: float = 0.25,
    ) -> None:
        self._f = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.conf_strength = float(conf_strength)
        self.conf_floor = float(conf_floor)

    def reset(self) -> None:
        self._f.reset()

    def __call__(
        self,
        landmarks: np.ndarray,
        t: float,
        confidence: np.ndarray | None = None,
    ) -> np.ndarray:
        """Smooth (N, 2) landmarks; `confidence` is (N,) in [0, 1] or None."""
        scale: np.ndarray | float = 1.0
        if confidence is not None and self.conf_strength > 0:
            c = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
            s = 1.0 - self.conf_strength * (1.0 - c)
            scale = np.clip(s, self.conf_floor, 1.0)[:, None]  # broadcast over (x, y)
        return self._f(landmarks, t, scale)


class BoxSmoother:
    """One Euro over a box as [ymin, xmin, ymax, xmax].

    Smoothing the box is what actually attacks the dominant jitter term: a stable
    crop means the landmarker sees a consistent input instead of a slightly
    different framing every frame.

    Boxes are filtered in centre/size parameterisation rather than corner
    coordinates, so smoothing cannot make the box drift out of shape -- position
    and scale are smoothed independently, and scale can be smoothed harder than
    position (scale wobble is almost always noise, whereas position often is not).
    """

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.03,
        size_min_cutoff: float = 0.4,
        size_beta: float = 0.01,
    ) -> None:
        self._pos = OneEuroFilter(min_cutoff, beta)
        self._size = OneEuroFilter(size_min_cutoff, size_beta)

    def reset(self) -> None:
        self._pos.reset()
        self._size.reset()

    def __call__(self, box: np.ndarray, t: float) -> np.ndarray:
        ymin, xmin, ymax, xmax = np.asarray(box, dtype=np.float64)
        centre = np.array([(ymin + ymax) / 2.0, (xmin + xmax) / 2.0])
        size = np.array([ymax - ymin, xmax - xmin])

        centre = self._pos(centre, t)
        size = np.maximum(self._size(size, t), 1.0)

        return np.array([
            centre[0] - size[0] / 2.0, centre[1] - size[1] / 2.0,
            centre[0] + size[0] / 2.0, centre[1] + size[1] / 2.0,
        ])


class EarTracker:
    """Associates detections across frames and owns one filter pair per track.

    Smoothing only means anything if frame N's box is filtered against the *same*
    ear's box from frame N-1. Without association, two ears in frame would swap
    filter states and the smoothing would actively corrupt the output.

    Association is nearest-centre within a fraction of box size, which is ample
    for the 1-2 well-separated ears this pipeline sees; it is not a general MOT
    solution.

    Args:
        max_dist_frac: Match radius as a fraction of the mean box size.
        max_missed: Drop a track after this many frames without a detection.
    """

    def __init__(
        self,
        max_dist_frac: float = 0.6,
        max_missed: int = 5,
        landmark_kwargs: dict | None = None,
        box_kwargs: dict | None = None,
    ) -> None:
        self.max_dist_frac = float(max_dist_frac)
        self.max_missed = int(max_missed)
        self._lm_kwargs = landmark_kwargs or {}
        self._box_kwargs = box_kwargs or {}
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    @staticmethod
    def _centre(box: np.ndarray) -> np.ndarray:
        ymin, xmin, ymax, xmax = box
        return np.array([(ymin + ymax) / 2.0, (xmin + xmax) / 2.0])

    def assign(self, boxes: list[np.ndarray]) -> list[int]:
        """Match each box to an existing track id, creating tracks as needed.

        Greedy nearest-centre, each track claimed at most once.
        """
        taken: set[int] = set()
        ids: list[int] = []
        for box in boxes:
            c = self._centre(box)
            size = max(float(np.mean([box[2] - box[0], box[3] - box[1]])), 1.0)
            best_id, best_d = None, self.max_dist_frac * size
            for tid, tr in self._tracks.items():
                if tid in taken:
                    continue
                d = float(np.linalg.norm(c - tr["centre"]))
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = {
                    "box_f": BoxSmoother(**self._box_kwargs),
                    "lm_f": LandmarkSmoother(**self._lm_kwargs),
                    "centre": c,
                    "missed": 0,
                }
            taken.add(best_id)
            ids.append(best_id)
        self._expire(taken)
        return ids

    def _expire(self, seen: set[int]) -> None:
        for tid in list(self._tracks):
            if tid in seen:
                self._tracks[tid]["missed"] = 0
            else:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_missed:
                    del self._tracks[tid]

    def smooth_box(self, tid: int, box: np.ndarray, t: float) -> np.ndarray:
        tr = self._tracks[tid]
        out = tr["box_f"](box, t)
        tr["centre"] = self._centre(out)
        return out

    def smooth_landmarks(
        self, tid: int, landmarks: np.ndarray, t: float,
        confidence: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._tracks[tid]["lm_f"](landmarks, t, confidence)
