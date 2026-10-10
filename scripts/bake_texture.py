#!/usr/bin/env python3
"""Bake an OBJ's texture map into per-vertex colours, and write a PLY.

WHY: the 3D path loads heads with o3d.io.read_triangle_mesh and then paints
them with a synthetic skin shader, because the meshes it was built for
(HUTUBS, SONICOM) carry no colour at all. A photogrammetry scan DOES carry
colour -- a real photograph of real skin -- and discarding it throws away the
one thing that makes a render look like a photograph, which is exactly what
the landmarker is weakest on.

Open3D cannot be used for this here: read_triangle_mesh(enable_post_processing
=True) aborts the process on this file with "Image has unrecognized
bytes_per_channel", and the .mtl names the map `BJ002_0.jpg` while the file on
disk is `BJ002_0.JPG`, which is a dangling reference on a case-sensitive
filesystem. Both are avoided by parsing the OBJ and sampling the texture here.

Per-vertex colour is a real approximation: a 4096x4096 map carries far more
detail than 99k vertices can hold, so pores and fine grain are lost and what
survives is the large-scale colouring. That is still much closer to a
photograph than a uniform skin tone.

    python scripts/bake_texture.py --obj supersecret/BJ002/BJ002.OBJ \
        --out supersecret/BJ002/BJ002_vcol.ply
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--texture", default="", help="defaults to the OBJ's map_Kd, "
                                                 "matched case-insensitively")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from PIL import Image
    import open3d as o3d

    # A 16k photogrammetry atlas is 268M pixels and PIL refuses anything over
    # 179M as a possible decompression bomb. That guard is for untrusted
    # input; these files are local scans whose size is known from their own
    # metadata, so it is raised deliberately rather than by a blanket None.
    Image.MAX_IMAGE_PIXELS = 400_000_000

    obj = pathlib.Path(a.obj)
    V, VT, faces = [], [], []
    for ln in open(obj, errors="ignore"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("vt "):
            VT.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith("f "):
            c = []
            for tok in ln.split()[1:]:
                bits = tok.split("/")
                vi = int(bits[0]) - 1
                ti = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1
                c.append((vi, ti))
            for i in range(1, len(c) - 1):          # fan-triangulate
                faces.append((c[0], c[i], c[i + 1]))
    V = np.asarray(V, float)
    VT = np.asarray(VT, float) if VT else np.zeros((0, 2))
    if not len(VT):
        sys.exit("this OBJ carries no texture coordinates")
    print(f"{len(V)} vertices, {len(VT)} uvs, {len(faces)} triangles")

    tex = a.texture
    if not tex:
        mtl = next((l.split(maxsplit=1)[1].strip()
                    for l in open(obj.with_suffix(".MTL"), errors="ignore")
                    if l.lower().startswith("map_kd")), None)
        if mtl is None:
            sys.exit("no map_Kd in the .MTL and no --texture given")
        # case-insensitive match, because the .mtl and the file disagree
        cands = [q for q in obj.parent.iterdir()
                 if q.name.lower() == pathlib.Path(mtl).name.lower()]
        if not cands:
            sys.exit(f"the .MTL names {mtl!r}, which is not in {obj.parent}")
        tex = cands[0]
    img = np.asarray(Image.open(tex).convert("RGB"), np.float32) / 255.0
    th, tw = img.shape[:2]
    print(f"texture {pathlib.Path(tex).name}  {tw}x{th}")

    tri = np.asarray([[c[0] for c in f] for f in faces])      # vertex indices
    tti = np.asarray([[c[1] for c in f] for f in faces])      # uv indices

    # Average the uv-sampled colour over every corner that references a vertex.
    # Vertices on a UV seam get the mean of both sides, which is the right
    # answer for a vertex colour and is a blur of at most one texel.
    acc = np.zeros((len(V), 3))
    cnt = np.zeros(len(V))
    flat_v, flat_t = tri.ravel(), tti.ravel()
    ok = flat_t >= 0
    uv = VT[flat_t[ok]]
    px = np.clip((uv[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(int)
    py = np.clip((1.0 - (uv[:, 1] % 1.0)) * (th - 1), 0, th - 1).astype(int)
    col = img[py, px]
    np.add.at(acc, flat_v[ok], col)
    np.add.at(cnt, flat_v[ok], 1.0)
    missing = cnt == 0
    cnt[missing] = 1.0
    C = acc / cnt[:, None]
    if missing.any():
        C[missing] = C[~missing].mean(0)
        print(f"{int(missing.sum())} vertices had no uv; given the mean colour")

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(tri)
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    m.compute_vertex_normals()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), m)
    print(f"wrote {out}  (mean colour {C.mean(0).round(3).tolist()})")


if __name__ == "__main__":
    main()
