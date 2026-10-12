#!/usr/bin/env python3
"""Wrap the template onto a POINT CLOUD, filling the gaps with the template.

Faceform Wrap needs meshes, so the AudioEar3D clouds are out of reach of it --
but nothing in non-rigid registration requires faces. Correspondence is a
nearest-neighbour query, and the only other thing the mesh was giving us was
normals, which these clouds carry per point. So the project's own NICP does
this directly.

The division of labour is the point. Where the cloud HAS points, they drive the
fit, exactly as a mesh would. Where it does not -- holes, the shadow behind the
helix, the scan's own edge -- every correspondence is rejected and those
vertices are decided by the stiffness term alone, which is to say by the
template's shape, smoothly blended into the parts that were measured. That is
warping, not registration, and it is what fills a hole with something ear-shaped
instead of with a dent.

Rejecting the cloud's EDGE points is what makes the difference. Template
vertices over a hole still find neighbours -- the points around its rim -- at
sensible distances and with agreeing normals, so without that test they pass
every check and drag the template down into the hole.

    python scripts/wrap_cloud.py --cloud AudioEar3D/001/left.ply \\
        --template templates/ear_mean_v4.obj --out fit_001_left.obj
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy.spatial import cKDTree                                 # noqa: E402

from ear3d.wrap import (boundary_points, boundary_vertices, load_mesh,  # noqa: E402
                        nicp, rigid_icp, vertex_normals)


def load_cloud(path):
    """Points and normals from a PLY. trimesh keeps normals in the raw block."""
    import trimesh
    c = trimesh.load(path, process=False)
    V = np.asarray(c.vertices, float)
    raw = c.metadata["_ply_raw"]["vertex"]["data"]
    N = np.asarray(raw[["nx", "ny", "nz"]].tolist(), float)
    if len(N) != len(V):
        sys.exit(f"{path}: {len(V)} points but {len(N)} normals")
    return V, N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cloud", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-dist", type=float, default=0.06,
                   help="correspondence limit, as a fraction of TEMPLATE size. "
                        "Never of the cloud: see the note in ear3d.wrap.nicp")
    p.add_argument("--min-cos", type=float, default=0.5)
    p.add_argument("--mirror", action="store_true",
                   help="right ear -> left form. Mirrors about Z, not X; "
                        "see wrap_corpus.mirror_to_left")
    p.add_argument("--render", default=None)
    a = p.parse_args()

    Vc, Nc = load_cloud(a.cloud)
    if a.mirror:
        Vc = Vc * np.array([1.0, 1.0, -1.0])
        Nc = Nc * np.array([1.0, 1.0, -1.0])
    Vt, Ft = load_mesh(a.template)
    size = float(np.ptp(Vt, axis=0).max())
    print(f"cloud {len(Vc)} points, extent {np.ptp(Vc, axis=0).round(1)} mm")
    print(f"template {len(Vt)} verts, extent {np.ptp(Vt, axis=0).round(1)} mm")

    edge = boundary_points(Vc, Nc)
    print(f"  {int(edge.sum())} cloud points ({100*edge.mean():.1f}%) are on an "
          f"edge -- a hole rim or the scan's own border; these are refused as "
          f"correspondences")

    # PRE-ALIGN before ICP. The two are already in the same frame, but they sit
    # in different places at different sizes, and ICP from that start matches
    # helix to concha on its first pass and never recovers -- it came back with
    # a 0.75 scale factor, shrinking the template inside the ear.
    k0 = float(np.ptp(Vc, axis=0).max()) / size
    Vp = (Vt - Vt.mean(0)) * k0 + Vc.mean(0)
    print(f"  pre-align: scale {k0:.3f}, centroid shift "
          f"{np.linalg.norm(Vc.mean(0) - Vt.mean(0)):.1f} mm")
    # NO SCALE in the ICP. The template is a CUT ear whose border runs out onto
    # scalp and cheek; the cloud is the pinna alone. Those border vertices have
    # no data under them, so a scale-free fit is pulled inward by them --
    # measured, it came back at 0.879 and left the template 54.8 mm tall inside
    # a 60.6 mm ear. The pre-align above sets the size from the extents, which
    # is the honest estimate, and NICP takes up what is left.
    V0, (R, t, s) = rigid_icp(Vp, Ft, Vc, Ns=Nc, max_dist=0.3 * size,
                              with_scale=False, verbose=True)
    V = nicp(V0, Ft, Vc, Ns=Nc, max_dist=a.max_dist * size,
             min_cos=a.min_cos, verbose=True)

    # Which vertices were MEASURED and which were invented by the template.
    # This is the number that matters on a cloud, and no distance-to-surface
    # check reports it: the filled parts sit at zero distance from nothing.
    tree = cKDTree(Vc)
    d, idx = tree.query(V)
    Nv = vertex_normals(V, Ft)
    Ncn = Nc / np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)
    far = d >= a.max_dist * size
    on_rim = ~far & edge[idx]
    bad_n = ~far & ~on_rim & (np.einsum("ij,ij->i", Nv, Ncn[idx]) <= a.min_cos)
    cut = np.zeros(len(V), bool)
    cut[boundary_vertices(Ft)] = True
    supported = ~far & ~on_rim & ~bad_n & ~cut
    print(f"\n{int(supported.sum())}/{len(V)} template vertices "
          f"({100*supported.mean():.1f}%) sit on measured points; the rest are "
          f"carried by the template:")
    for lab, m in (("the template's own cut border", cut & ~far),
                   ("no cloud point in range (holes, behind the ear)", far),
                   ("nearest point is a hole rim", on_rim),
                   ("normal disagrees", bad_n)):
        print(f"    {int(m.sum()):4d}  {100*m.mean():5.1f}%  {lab}")
    if far.any():
        print(f"  the filled vertices bridge {np.median(d[far]):.1f} mm of empty "
              f"cloud on average, {d[far].max():.1f} mm at worst")
    # The gap where measured is near zero BY CONSTRUCTION -- the last stiffness
    # stage is loose enough to interpolate the points -- so it measures nothing
    # about whether the fit found the right anatomy. The support fraction above
    # is the number that carries information.
    print(f"  where measured: median gap {np.median(d[supported]):.3f} mm "
          f"(near zero by construction; not a quality measure)")

    import trimesh
    trimesh.Trimesh(vertices=V, faces=Ft, process=False).export(a.out)
    print(f"wrote {a.out}")
    np.save(str(pathlib.Path(a.out).with_suffix(".support.npy")), supported)

    if a.render:
        _render(Vc, V, Ft, supported, a.render)
        print(f"wrote {a.render}")


def _render(Vc, V, F, supported, path):
    import open3d as o3d
    from PIL import Image, ImageDraw
    from ear3d.camera import camera_ke, orbit_extrinsic
    from ear3d.config import DEFAULTS
    from ear3d.render import MATERIAL, light_scene
    cfg, S = dict(DEFAULTS), 700
    c = V.mean(0)
    k = 0.55 * float(np.ptp(V, axis=0).max())
    nm = lambda X: (X - c) / k

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(nm(V)),
                                     o3d.utility.Vector3iVector(F))
    mesh.compute_vertex_normals()
    col = np.where(supported[:, None], [0.70, 0.72, 0.75], [0.95, 0.35, 0.15])
    mesh.vertex_colors = o3d.utility.Vector3dVector(col)
    plain = o3d.geometry.TriangleMesh(mesh)
    plain.paint_uniform_color([0.62, 0.64, 0.68])
    plain.compute_vertex_normals()

    step = max(1, len(Vc) // 60000)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(nm(Vc[::step])))
    pc.paint_uniform_color([0.25, 0.75, 0.95])
    pm = o3d.visualization.rendering.MaterialRecord()
    pm.shader, pm.point_size = "defaultUnlit", 2.5

    tiles, labels = [], []
    for name, geoms in (("the cloud", [("p", pc, pm)]),
                        ("fit over the cloud", [("p", pc, pm), ("m", plain, MATERIAL)]),
                        ("orange = filled by template", [("m", mesh, MATERIAL)])):
        for yaw in (-90.0, -50.0):
            r = o3d.visualization.rendering.OffscreenRenderer(S, S)
            for n, g, mat in geoms:
                r.scene.add_geometry(n, g, mat)
            light_scene(r.scene, (0.12, 0.12, 0.14))
            K, E = camera_ke(S, cfg, orbit_extrinsic(yaw, 0.0, cfg))
            r.setup_camera(K, E, S, S)
            tiles.append(np.asarray(r.render_to_image())[:, :, :3])
            labels.append(f"{name}   yaw {yaw:+.0f}")
            del r
    sheet = Image.fromarray(np.vstack([np.hstack(tiles[0:2]),
                                       np.hstack(tiles[2:4]),
                                       np.hstack(tiles[4:6])]))
    d = ImageDraw.Draw(sheet)
    for i, lab in enumerate(labels):
        d.text((12 + S * (i % 2), 10 + S * (i // 2)), lab, fill=(225, 225, 225))
    sheet.save(path)


if __name__ == "__main__":
    main()
