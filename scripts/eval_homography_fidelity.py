"""Does the perspective AUGMENTATION look like a real head turn?

data/dataset.py simulates yaw/pitch with a homography and notes it "models
foreshortening but not 3D parallax or self-occlusion". Its measured benefit
plateaus at 65 deg. Two readings fit: the model saturates, or the warp stops
resembling a rotation. This separates them with NO model and NO landmarks --
render a real ear mesh at true yaw, apply dataset.py's own homography for the
same nominal yaw to the pose-zero render, and measure the disagreement.

Result on 12 AudioEar3D ears: agreement falls smoothly from NCC 0.80 at 10 deg
to 0.47 at 30, 0.15 at 65 and negative at 80. There is no cliff at 65 -- the warp
was never a close match. Combined with eval_pose_consistency.py finding no
measurable yaw benefit from the augmentation (-3.2%, t=-0.99) while pitch does
benefit (-18.0%, t=-4.00), the picture is that the homography helps where self-
occlusion is absent and cannot help where it dominates.

Usage:
    python scripts/eval_homography_fidelity.py --ears 12

SUPERSEDED STACK -- READ BEFORE TRUSTING ANY NUMBER FROM THIS FILE.

This script predates ear3d/ and still runs the old 3D path: render3d_ears'
analytic project()/unproject_rays() with a field-of-view camera, and a
RaycastingScene for back-projection. The current path (ear3d/, driven by
scripts/view_backprojection.py and scripts/ingest_render3d.py) differs in ways
that change results:

  - the camera is now one explicit (K, E) handed to both the renderer and the
    projection, instead of a FOV form paired with a separate analytic formula
  - back-projection reads the depth buffer of the render that produced the
    pixels, instead of casting rays into a second acceleration structure
  - ears are found through a MediaPipe head frame, and labelled by the shipped
    EarLandmarkerPipeline on the whole frame rather than a hand-made crop

Numbers produced here were measured before the Blender camera faults, the
renderer split and the framing fixes were found, so they are not comparable with
anything the current path reports. Migrate it to ear3d before using it again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render3d_ears import ears, load_ear, project, render   # noqa: E402
from data.dataset import EarLandmarkDataset                 # noqa: E402

SIZE = 600
OUT = 192
TRAIN_OCC = 0.777


def framed(mesh, yaw, pitch, out=OUT, size=SIZE):
    img = render(mesh, yaw, pitch, size=size)[:, :, :3]
    from render3d_ears import rot
    uv = project(np.asarray(mesh.vertices) @ rot(yaw, pitch).T, size)
    side = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / TRAIN_OCC
    cx = (uv[:, 0].min() + uv[:, 0].max()) / 2
    cy = (uv[:, 1].min() + uv[:, 1].max()) / 2
    l, t, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    if s < 32:
        return None
    cv = Image.new("RGB", (s, s), (128, 128, 128))
    src = Image.fromarray(img)
    sx0, sy0 = max(0, l), max(0, t)
    sx1, sy1 = min(size, l + s), min(size, t + s)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    cv.paste(src.crop((sx0, sy0, sx1, sy1)), (sx0 - l, sy0 - t))
    return cv.resize((out, out), Image.BILINEAR)


def homography_warp(img, yaw, pitch):
    """Exactly the warp data/dataset.py applies for this yaw/pitch."""
    h = EarLandmarkDataset._perspective_matrix(yaw, pitch)
    n = img.size[0]
    scale = np.array([[n, 0, 0], [0, n, 0], [0, 0, 1]])
    hp = scale @ h @ np.linalg.inv(scale)
    hi = np.linalg.inv(hp)
    hi = hi / hi[2, 2]
    return img.transform((n, n), Image.PERSPECTIVE, hi.flatten()[:8].tolist(),
                         resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def compare(a, b):
    """NCC over pixels that are ear in EITHER image, plus silhouette IoU."""
    A = np.asarray(a.convert("L"), dtype=np.float64)
    B = np.asarray(b.convert("L"), dtype=np.float64)
    ma = np.abs(np.asarray(a.convert("RGB")).astype(int) - 128).max(2) > 6
    mb = np.abs(np.asarray(b.convert("RGB")).astype(int) - 128).max(2) > 6
    m = ma | mb
    if m.sum() < 50:
        return np.nan, np.nan
    x, y = A[m] - A[m].mean(), B[m] - B[m].mean()
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return (float((x * y).sum() / den) if den > 0 else np.nan,
            float((ma & mb).sum() / max(m.sum(), 1)))


def main() -> None:
    p = argparse.ArgumentParser(description="Homography vs true 3D rotation")
    p.add_argument("--ears", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--yaws", type=int, nargs="*", default=[10, 20, 30, 40, 50, 65, 80])
    args = p.parse_args()

    mem = ears()
    rng = np.random.default_rng(args.seed)
    mem = [mem[i] for i in sorted(rng.choice(len(mem), min(args.ears, len(mem)), replace=False))]
    acc = {a: {"ncc": [], "iou": []} for a in args.yaws}
    ok = 0
    for m in mem:
        try:
            mesh = load_ear(m)
        except Exception:
            continue
        if mesh is None:
            continue
        base = framed(mesh, 0, 0)
        if base is None:
            continue
        ok += 1
        for a in args.yaws:
            t = framed(mesh, a, 0)
            if t is None:
                continue
            c, iou = compare(t, homography_warp(base, a, 0))
            if np.isfinite(c):
                acc[a]["ncc"].append(c)
                acc[a]["iou"].append(iou)
    print(f"\n{ok} ears\n")
    print(f"{'yaw':>6s} {'NCC(true 3D, homography)':>26s} {'silhouette IoU':>16s}")
    for a in args.yaws:
        if acc[a]["ncc"]:
            print(f"{a:>4d}deg {np.mean(acc[a]['ncc']):26.3f} {np.mean(acc[a]['iou']):16.3f}")


if __name__ == "__main__":
    main()
