#!/usr/bin/env python3
"""Render real heads with landmarks whose x, y AND z come from the geometry.

This is what the whole template-fitting track was for. The 55 marks were placed
by hand on the mean ear, propagated to every fitted ear, and carried back onto
the original head by the transform cut_ears records -- so no detector and no
landmarker is involved anywhere in the label. ingest_render3d.py, by contrast,
lifts the MODEL's own predictions through a depth buffer, which is why its
labels are model-seeded and cannot be used to judge the model.

THE HEAD IS RENDERED, NOT THE EAR ALONE. A floating ear teaches a model that a
hard silhouette marks the boundary, which is the lesson crop_to_occupancy
already records; and depth read off an isolated ear has no surrounding surface
to be relative to.

REAL ROTATION, NOT A HOMOGRAPHY. data/dataset.py simulates yaw by warping, which
cannot produce parallax or self-occlusion -- measured, it correlates with a true
rotation at 0.47 at 30 degrees and 0.15 at 65. For a 2D label that is a
weakness; for a DEPTH label it is fatal, because the warp leaves the image
looking turned while z stays frontal. Here the geometry is rotated and
re-rendered, so z is right at every pose.

    python scripts/ingest_depth3d.py --fits corpus/faceform2_hutubs \\
        --transforms corpus/tf_hutubs --marks mean_v4_marks.npz \\
        --template ear_mean.obj --out data/depth3d
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.spatial import cKDTree                               # noqa: E402

from ear3d.camera import Camera, camera_ke, orbit_extrinsic     # noqa: E402
from ear3d.config import DEFAULTS                               # noqa: E402
from ear3d.frames import load_head                              # noqa: E402
from ear3d.render import MATERIAL, light_scene                  # noqa: E402
from ear3d.scheme import STRIPS                                 # noqa: E402

# sun intensity x skin luminance, calibrated so a mid tone lands near the
# 0.33 mean image brightness the old fixed-tone corpus had
EXPOSURE = 14000.0
from ear3d.wrap import base_topology, load_mesh                 # noqa: E402


def _flush(out, rows, lms, viss):
    """Write what we have so far, so a crash costs one chunk and not the run."""
    np.save(out / "landmarks.npy", np.asarray(lms, np.float32))
    np.save(out / "visible.npy", np.asarray(viss, bool))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "image_file", "ear", "source_mesh", "yaw", "pitch",
                    "occupancy", "n_visible"])
        w.writerows(rows)


def ear_frame(M):
    """Rotation taking the ear to the renderer's canonical pose.

    The template faces -x in the cut frame and runs head-to-lobe along y, so the
    cut transform's rotation gives both directions on the head. orbit_extrinsic
    puts the camera on +z at yaw 0, so the ear's outward direction becomes +z.
    """
    A = M[:3, :3]
    out = A @ np.array([-1.0, 0.0, 0.0])
    up = A @ np.array([0.0, 1.0, 0.0])
    out = out / np.linalg.norm(out)
    up = up - out * (up @ out)
    up = up / np.linalg.norm(up)
    right = np.cross(up, out)
    return np.stack([right, up, out])          # rows: x, y, z of the new frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", nargs="+", required=True)
    p.add_argument("--transforms", nargs="+", required=True)
    p.add_argument("--marks", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--yaws", type=float, nargs="*",
                   default=[-40, -25, -12, 0, 12, 25, 40])
    p.add_argument("--pitches", type=float, nargs="*", default=[-15, 0, 15])
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--render-size", type=int, default=640)
    p.add_argument("--occ", type=float, nargs=2, default=(0.55, 0.80),
                   help="how much of the crop the ear fills, sampled per view")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="keep the ears already in the output and carry on")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--vary-light", action="store_true",
                   help="draw sun direction, colour, intensity and background "
                        "per view instead of the single fixed setup the old "
                        "corpus used")
    p.add_argument("--random-tone", action="store_true",
                   help="draw a Fitzpatrick I-VI tone per ear instead of the "
                        "single fixed config tone the old corpus used")
    p.add_argument("--subdiv-to", type=int, default=0,
                   help="midpoint-subdivide every head until it has at least "
                        "this many faces, so vertex-colour texture is sampled "
                        "at the same rate whatever the source; 300000 brings "
                        "SONICOM up to HUTUBS")
    a = p.parse_args()

    import open3d as o3d
    from PIL import Image
    from skin import apply_skin
    from ingest_render3d import crop_to_occupancy

    out = pathlib.Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    m = np.load(a.marks)
    tri, bary = m["triangle"], m["barycentric"]
    Vt, Ft = load_mesh(a.template)
    cfg = dict(DEFAULTS)
    rng = np.random.default_rng(a.seed)

    jobs = []
    for fd, td in zip(a.fits, a.transforms):
        for f in sorted(pathlib.Path(fd).glob("*.obj")):
            t = pathlib.Path(td) / f"{f.stem}.npz"
            if t.exists():
                jobs.append((f, t))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} ears x {len(a.yaws)}x{len(a.pitches)} views", flush=True)

    rows, lms, viss, skipped, n = [], [], [], [], 0
    done_ears = set()
    if a.resume and (out / "manifest.csv").exists():
        import csv as _csv
        prev = list(_csv.DictReader(open(out / "manifest.csv")))
        if prev:
            rows = [[r[k] for k in ("idx", "image_file", "ear", "source_mesh",
                                    "yaw", "pitch", "occupancy", "n_visible")]
                    for r in prev]
            lms = list(np.load(out / "landmarks.npy"))
            viss = list(np.load(out / "visible.npy"))
            done_ears = {r["ear"] for r in prev}
            n = max(int(r["idx"]) for r in prev) + 1
            print(f"resuming: {len(done_ears)} ears and {n} views already done",
                  flush=True)
    for f, tp in jobs:
        if f.stem in done_ears:
            continue
        z = np.load(tp, allow_pickle=True)
        M, src = z["matrix"], str(z["source"])
        V, F = load_mesh(f)
        V, F = base_topology(V, F, Vt, Ft, f.name)
        if f.stem.endswith("_right"):
            V = V * np.array([1.0, 1.0, -1.0])      # fits are stored in left form
        L = np.einsum("ij,ijk->ik", bary, V[F[tri]])
        L = L @ M[:3, :3].T + M[:3, 3]              # onto the head

        # LOAD THE SOURCE THE WAY ITS TRANSFORM EXPECTS. cut_hutubs works on
        # load_head's normalised copy; cut_sonicom works on the raw STL. Using
        # load_head for both put the SONICOM landmarks 95 units from a mesh
        # normalised to unit radius, and the renders came back as empty grey
        # with the marks floating on nothing.
        if str(z["dataset"]) == "hutubs":
            head = load_head(src)
        else:
            import open3d as o3d_
            head = o3d_.io.read_triangle_mesh(src)
            if len(head.vertices):
                head.compute_vertex_normals()

        # EQUALISE THE TWO SOURCES' MESH RESOLUTION, or the renders announce
        # which dataset they came from. Measured on the previous corpus, a
        # logistic regression on eleven global image statistics identified the
        # source at 87% on a subject-disjoint split, driven by gradient mean at
        # d = 1.24. HUTUBS heads carry ~337k faces against SONICOM's ~36k, a 9x
        # difference in surface detail that blur does not touch (0.887 -> 0.884
        # at sigma <= 0.8) and that merging SONICOM's unshared STL vertices
        # makes WORSE, not better (0.870 -> 0.996): smooth-shading a coarse
        # mesh only widens the gap. The resolution itself has to go.
        #
        # Both sources get the SAME treatment -- merge, decimate to a common
        # target, recompute normals -- so neither is processed in a way the
        # other is not. SONICOM is already near the target, so it is close to
        # a no-op there; the detail being discarded is detail SONICOM never
        # had, and the renders carry synthetic skin texture anyway.
        # EQUALISE THE TEXTURE SAMPLING RATE, not the geometry. skin.py paints
        # grain as VERTEX COLOURS, so the finest texture a mesh can carry is
        # set by its vertex density -- HUTUBS heads have ~337k faces and
        # SONICOM ~36k, so the same grain is resolved by one and aliased away
        # by the other. Measured, that let a logistic regression on eleven
        # global image statistics name the source at 87% on a subject-disjoint
        # split.
        #
        # Three things that did NOT fix it, all measured: blur (0.887 ->
        # 0.884), decimating both to a common face count (0.870 -> 0.903, and
        # it visibly smears the skin), and merging SONICOM's unshared STL
        # vertices to match HUTUBS' shading (0.870 -> 0.996).
        #
        # MIDPOINT subdivision is used, not Loop: it inserts vertices at edge
        # midpoints and leaves the surface exactly where it was, so this
        # raises the sampling rate for texture without inventing geometry.
        if a.subdiv_to and head is not None and len(head.vertices):
            head = head.remove_duplicated_vertices()
            head = head.remove_duplicated_triangles()
            head = head.remove_degenerate_triangles()
            guard = 0
            while len(head.triangles) < a.subdiv_to and guard < 4:
                head = head.subdivide_midpoint(1)
                guard += 1
            head.compute_vertex_normals()
        if head is None or not len(head.vertices):
            skipped.append((f.stem, "the source mesh would not load"))
            continue

        # AND CHECK IT, rather than trusting the branch above. The marks came
        # off this very surface, so they must sit on it; anything else means
        # the frames disagree and every label from this ear would be fiction.
        Vsrc = np.asarray(head.vertices)
        gap = float(np.median(cKDTree(Vsrc).query(L)[0]))
        ear = float(np.ptp(L, axis=0).max())
        if gap > 0.05 * ear:
            skipped.append((f.stem, f"landmarks sit {gap / ear:.2f} ear-widths "
                                    f"off the source surface"))
            continue
        Q = ear_frame(M)
        c = L.mean(0)
        Vh = (np.asarray(head.vertices) - c) @ Q.T
        scale = float(np.ptp(L, axis=0).max())
        Vh = Vh / scale                             # ear ~1 unit across
        Lc = ((L - c) @ Q.T) / scale
        g = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vh),
                                      head.triangles)
        g.compute_vertex_normals()
        # TONE IS DRAWN PER EAR, not held at the config default. The whole
        # previous corpus was rendered at tone=10 -- a single fixed value, and
        # one that skin.py itself documents as "deeper than Fitzpatrick VI,
        # i.e. past real skin ... not for training images". A corpus with no
        # skin-colour variation is its own artefact: the model never sees any,
        # and the renders are already the weak link against real photographs
        # (4.26% error on them against 1.47% on real). Here it costs nothing,
        # because these labels come from GEOMETRY -- marks carried through the
        # cut transform -- so no label depends on what colour the skin is.
        tone = (None if a.random_tone else cfg["tone"])
        apply_skin(g, seed=int(rng.integers(1 << 30)), tone=tone,
                   ao_strength=cfg["ao_strength"], n_rays=cfg["ao_rays"])
        # AUTO-EXPOSE, so image brightness does not encode skin tone. The old
        # corpus rendered every ear at tone 10, luminance 0.099, under a fixed
        # 95000 sun. Drawing tones from the Fitzpatrick I-VI pool, mean
        # luminance 0.549, left the same sun 5.5x too strong: measured, the
        # mean image went from 0.332 to 0.708. A camera would not do that --
        # it meters the scene -- and a corpus where pale skin is bright and
        # dark skin is dim teaches brightness as a feature of the subject.
        skin_lum = (float((np.asarray(g.vertex_colors)
                           @ [0.2126, 0.7152, 0.0722]).mean())
                    if len(g.vertex_colors) else 0.5)
        sun = EXPOSURE / max(skin_lum, 0.05)

        # ONE RENDERER PER EAR, not per view. Filament holds a GPU context per
        # engine, and creating 9996 of them leaked until the driver refused --
        # the run died at 6600 views with no traceback, mid-write. The geometry
        # only changes per ear; the camera is all that moves between views.
        r = o3d.visualization.rendering.OffscreenRenderer(
            a.render_size, a.render_size)
        r.scene.add_geometry("h", g, MATERIAL)
        light_scene(r.scene, (0.5, 0.5, 0.5))
        for yaw in a.yaws:
            for pitch in a.pitches:
                # LIGHT VARIES PER VIEW. light_scene is documented as "the one
                # lighting setup", and it was: one sun direction, one colour,
                # one intensity, one background for every image in the corpus.
                # That matters more for depth than for anything else, because
                # SHADING IS A DEPTH CUE -- under a single fixed light a model
                # can learn "this gradient means this z", which is true of the
                # corpus and not of a photograph. Nothing in the labels
                # depends on it: they come from geometry, and `visible` is
                # read from the depth buffer.
                if a.vary_light:
                    v = rng.normal(size=3)
                    v[2] = -abs(v[2]) - 0.25        # keep the sun in front
                    v /= np.linalg.norm(v)
                    warm = rng.uniform(0.90, 1.0, 3)
                    warm[rng.integers(3)] = 1.0
                    r.scene.scene.set_sun_light(
                        list(v), list(warm),
                        float(sun * rng.uniform(0.65, 1.5)))
                    bg = float(rng.uniform(0.30, 0.70))
                    r.scene.set_background([bg, bg, bg, 1.0])
                elif a.random_tone:
                    r.scene.scene.set_sun_light(
                        [-0.3, -0.4, -0.9], [1.0, 1.0, 1.0], float(sun))
                E = orbit_extrinsic(yaw, pitch, cfg)
                K, E = camera_ke(a.render_size, cfg, E)
                r.setup_camera(K, E, a.render_size, a.render_size)
                img = np.asarray(r.render_to_image())[:, :, :3]
                depth = np.asarray(r.render_to_depth_image(z_in_view_space=True))
                cam = Camera(K, E, img.shape[:2], a.render_size)
                uv = cam.project(Lc)
                zc = (Lc @ cam.R.T + cam.t)[:, 2]          # camera-space depth

                # VISIBLE means the surface at that pixel is not in front of the
                # landmark. Occluded points still get a label -- a depth model
                # should predict the far side of the helix -- but the flag lets a
                # loss ignore them if that turns out to hurt.
                ui = np.clip(np.round(uv[:, 0]).astype(int), 0, img.shape[1] - 1)
                vi = np.clip(np.round(uv[:, 1]).astype(int), 0, img.shape[0] - 1)
                vis = depth[vi, ui] >= zc - 0.02
                inside = ((uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1])
                          & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0]))
                if vis.sum() < 30 or not inside.all():
                    continue
                occ = float(rng.uniform(*a.occ))
                crop, box = crop_to_occupancy(img, uv, occ, out=a.size)
                if crop is None:
                    continue
                l, t, s, o = box
                xy = (uv - np.array([l, t])) / s            # into [0, 1]

                # z IN THE SAME UNITS AS x AND y, which is what makes it
                # learnable from a crop: the crop spans s pixels, and at the
                # ear's distance one pixel is zc/fx of world, so the crop is
                # s*zbar/fx wide. Depth is root-relative; absolute distance is
                # not recoverable from a crop and pretending otherwise would
                # teach the model to guess the camera.
                zbar = float(zc.mean())
                span = s * zbar / cam.fx
                zn = (zc - zbar) / span
                if not (np.isfinite(xy).all() and np.isfinite(zn).all()):
                    continue
                name = f"{n:06d}.jpg"
                crop.save(out / "images" / name, quality=a.quality)
                rows.append([n, f"images/{name}", f.stem, src, f"{yaw:.0f}",
                             f"{pitch:.0f}", f"{occ:.3f}", int(vis.sum())])
                lms.append(np.c_[xy, zn])
                viss.append(vis)
                n += 1
        del r
        done_ears.add(f.stem)
        # FLUSH AS WE GO. The first version accumulated everything and wrote at
        # the end, so a crash at 6600 of 9800 views threw away forty minutes.
        if len(done_ears) % 20 == 0 or f is jobs[-1][0]:
            _flush(out, rows, lms, viss)
            print(f"  {n} views, {len(done_ears)} ears done", flush=True)

    _flush(out, rows, lms, viss)
    print(f"\n{n} views -> {out}")
    if skipped:
        print(f"  {len(skipped)} ears skipped:")
        for nm, why in skipped[:10]:
            print(f"    {nm}: {why}")
    A = np.asarray(lms)
    if len(A):
        print(f"  x,y in [{A[:, :, :2].min():.3f}, {A[:, :, :2].max():.3f}]")
        print(f"  z  in [{A[:, :, 2].min():.3f}, {A[:, :, 2].max():.3f}], "
              f"sd {A[:, :, 2].std():.3f}")
        print(f"  visible landmarks: median {np.median(np.asarray(viss).sum(1)):.0f}/55")


if __name__ == "__main__":
    main()
