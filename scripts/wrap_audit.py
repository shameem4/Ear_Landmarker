#!/usr/bin/env python3
"""Audit a wrapped corpus: find the bad fits, and ask whether one template is enough.

Three checks, none of which a fitting metric gives you:

  DISTANCE FROM THE MEAN. Every fit is in the same parameterisation, so they can
  be averaged. A fit that slid onto the wrong anatomy is far from the mean in a
  way a good fit of an unusual ear is not -- it is wrong in a direction no real
  ear goes.

  LEFT AGAINST RIGHT. Most subjects contribute both ears, and after mirroring
  they should be similar: the two sides of one person agree far more than two
  people do. When they do not, one of the two fits is wrong, and this says
  WHICH SUBJECT to look at without any ground truth at all.

  IS THE RESIDUAL ONE MODE OR TWO. If the distances from the mean are unimodal,
  one template serves the corpus. If they split, that is a real shape clustering
  and template variants earn their place. This is the measurement that settles
  it, rather than guessing from a couple of ears.

Outliers get rendered so they can be looked at, because a number saying a fit is
odd is not the same as seeing how.

    python scripts/wrap_audit.py --wrapped wrapped/ --out audit/
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import load_mesh, rigid_fit               # noqa: E402


def procrustes_stack(V, iters=3):
    """Align every fit to their running mean. Returns (aligned, mean)."""
    A = V.copy()
    mean = A[0]
    for _ in range(iters):
        for i in range(len(A)):
            R, t, s = rigid_fit(A[i], mean)
            A[i] = s * A[i] @ R.T + t
        mean = A.mean(0)
    return A, mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wrapped", nargs="+", required=True)
    p.add_argument("--out", default=None, help="directory for outlier renders")
    p.add_argument("--worst", type=int, default=8)
    a = p.parse_args()

    files = []
    for d in a.wrapped:
        files += sorted(pathlib.Path(d).glob("*.obj"))
    if not files:
        sys.exit("no wrapped meshes found")
    V0, F = load_mesh(files[0])
    V = np.empty((len(files), len(V0), 3))
    names = []
    for i, f in enumerate(files):
        Vi, Fi = load_mesh(f)
        if len(Vi) != len(V0):
            sys.exit(f"{f.name}: {len(Vi)} verts against {len(V0)} -- not one topology")
        V[i] = Vi
        names.append(f.stem)
    print(f"{len(files)} fits, {len(V0)} vertices each")

    A, mean = procrustes_stack(V)
    size = float(np.ptp(mean, axis=0).max())
    d = np.linalg.norm(A - mean, axis=2).mean(axis=1) / size * 100
    order = np.argsort(-d)

    print(f"\ndistance from the corpus mean, % of ear size:")
    print(f"  median {np.median(d):.2f}  p90 {np.percentile(d,90):.2f}  "
          f"worst {d.max():.2f}")
    print(f"\nthe {a.worst} furthest from the mean -- most likely to be bad fits:")
    for i in order[:a.worst]:
        print(f"  {names[i]:<24s} {d[i]:5.2f}%")

    # one mode or two? a split means template variants are earning their place
    hist, edges = np.histogram(d, bins=24)
    print("\ndistribution (each block is a bin, left to right):")
    peak = hist.max()
    print("  " + "".join("#" if h > 0.5 * peak else ("+" if h > 0.15 * peak
          else ("." if h else " ")) for h in hist))
    print(f"  {edges[0]:.1f}%{' ' * 16}{edges[-1]:.1f}%")
    lo, hi = d < np.median(d), d >= np.median(d)
    sep = (d[hi].mean() - d[lo].mean()) / d.std()
    print(f"  spread {d.std():.2f}, split statistic {sep:.2f} "
          f"(a clean two-mode split runs well above 2)")

    # left against right, which needs no ground truth
    pairs = {}
    for i, n in enumerate(names):
        if n.endswith("_left") or n.endswith("_right"):
            pairs.setdefault(n.rsplit("_", 1)[0], {})[n.rsplit("_", 1)[1]] = i
    both = {k: v for k, v in pairs.items() if len(v) == 2}
    if both:
        lr = []
        for k, v in both.items():
            R, t, s = rigid_fit(A[v["right"]], A[v["left"]])
            e = np.linalg.norm(A[v["left"]] - (s * A[v["right"]] @ R.T + t), axis=1)
            lr.append((k, 100 * float(np.median(e)) / size))
        lr.sort(key=lambda x: -x[1])
        vals = np.array([x[1] for x in lr])
        print(f"\nleft vs right within a subject ({len(lr)} subjects), "
              f"% of ear size:")
        print(f"  median {np.median(vals):.2f}  p90 {np.percentile(vals,90):.2f}")
        print(f"  worst -- one of the two fits is probably wrong:")
        for k, v in lr[:a.worst]:
            print(f"    {k:<22s} {v:5.2f}%")

    if a.out:
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        _render(A, F, mean, [int(i) for i in order[:a.worst]], names,
                out / "worst_fits.png")
        print(f"\nwrote {out / 'worst_fits.png'} (top row is the corpus mean)")


def _render(A, F, mean, idx, names, path):
    import open3d as o3d
    from PIL import Image
    from ear3d.camera import camera_ke, orbit_extrinsic
    from ear3d.config import DEFAULTS
    from ear3d.render import MATERIAL, light_scene
    cfg = dict(DEFAULTS)
    size = 0.55 * float(np.ptp(mean, axis=0).max())
    tiles = []
    for V in [mean] + [A[i] for i in idx]:
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector((V - mean.mean(0)) / size),
            o3d.utility.Vector3iVector(F))
        m.compute_vertex_normals()
        r = o3d.visualization.rendering.OffscreenRenderer(300, 300)
        r.scene.add_geometry("m", m, MATERIAL)
        light_scene(r.scene, (0.5, 0.5, 0.5))
        K, E = camera_ke(300, cfg, orbit_extrinsic(-90.0, 0.0, cfg))
        r.setup_camera(K, E, 300, 300)
        tiles.append(np.asarray(r.render_to_image())[:, :, :3])
        del r
    w = 5
    rows = [np.hstack(tiles[i:i + w]) for i in range(0, len(tiles), w)]
    wid = min(r.shape[1] for r in rows)
    Image.fromarray(np.vstack([r[:, :wid] for r in rows])).save(path)


if __name__ == "__main__":
    main()
