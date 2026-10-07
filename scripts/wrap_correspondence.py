#!/usr/bin/env python3
"""Does vertex N land on the same anatomy in every wrapped ear?

That is the only property a shape model actually needs, and no fitting metric
reports it. Surface distance is driven to zero by the projection step whether
or not the template found the right fold; agreement with another tool only says
two fits agree. This asks the question directly.

A high-frequency checkerboard is defined on the FIRST mesh's geometry and then
painted onto the others BY VERTEX INDEX. If the correspondence holds, the
squares land on the same features everywhere -- the row along the helix rim,
the patch in the concha. If it slid, the squares drift off the anatomy they sit
on in the first, and a smooth gradient would have hidden it.

    python scripts/wrap_correspondence.py a.obj b.obj [c.obj ...] -o out.png
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import load_mesh, rigid_fit            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("meshes", nargs="+", help="wrapped meshes, all one topology")
    p.add_argument("-o", "--out", default="correspondence.png")
    p.add_argument("--freq", type=int, default=9, help="checker squares per axis")
    p.add_argument("--size", type=int, default=430)
    a = p.parse_args()

    V0, F = load_mesh(a.meshes[0])
    sets = [(pathlib.Path(a.meshes[0]).stem, V0)]
    for m in a.meshes[1:]:
        V, Fi = load_mesh(m)
        if len(V) != len(V0) or not np.array_equal(Fi, F):
            sys.exit(f"{m}: {len(V)} verts / {len(Fi)} faces against "
                     f"{len(V0)} / {len(F)} -- not the same topology, so there "
                     f"is no correspondence to check")
        # bring it alongside the first USING the correspondence, so what is left
        # on screen is shape difference rather than where the scan happened to sit
        R, t, s = rigid_fit(V, V0)
        sets.append((pathlib.Path(m).stem, s * V @ R.T + t))

    size = float(np.ptp(V0, axis=0).max())
    for name, V in sets[1:]:
        d = np.linalg.norm(V - V0, axis=1)
        print(f"{name:>28s} vs {sets[0][0]}: median {100*np.median(d)/size:5.2f}% "
              f"p90 {100*np.percentile(d,90)/size:5.2f}% of ear size "
              f"(between-subject shape difference; it cannot be zero)")

    import open3d as o3d
    from PIL import Image
    from ear3d.camera import camera_ke, orbit_extrinsic
    from ear3d.config import DEFAULTS
    from ear3d.render import light_scene
    cfg = dict(DEFAULTS)

    P = (V0 - V0.min(0)) / np.ptp(V0, axis=0)
    chk = (np.floor(P[:, 1] * a.freq) + np.floor(P[:, 2] * a.freq)) % 2
    col = np.where(chk[:, None] > 0, np.array([0.95, 0.35, 0.10]),
                   np.array([0.10, 0.30, 0.80]))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    rows = []
    for name, V in sets:
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector((V - V0.mean(0)) / (0.55 * size)),
            o3d.utility.Vector3iVector(F))
        m.vertex_colors = o3d.utility.Vector3dVector(col)
        m.compute_vertex_normals()
        tiles = []
        for yaw in (-90.0, -55.0, -125.0):
            r = o3d.visualization.rendering.OffscreenRenderer(a.size, a.size)
            r.scene.add_geometry("m", m, mat)
            light_scene(r.scene, (0.5, 0.5, 0.5))
            K, E = camera_ke(a.size, cfg, orbit_extrinsic(yaw, 0.0, cfg))
            r.setup_camera(K, E, a.size, a.size)
            tiles.append(np.asarray(r.render_to_image())[:, :, :3])
            del r
        rows.append(np.hstack(tiles))
    Image.fromarray(np.vstack(rows)).save(a.out)
    print(f"wrote {a.out}  (rows: {', '.join(n for n, _ in sets)})")


if __name__ == "__main__":
    main()
