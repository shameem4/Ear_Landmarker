#!/usr/bin/env python3
"""A window onto every stage of the multi-view lift, not just its answer.

WHAT THIS IS FOR. The pipeline now has several places a landmark can go
wrong, and a single number at the end cannot say which one did. A per-view
estimate can read the scalp; a set of views can disagree; the consensus can
sit off the surface. Each of those looks identical in a z_skill table and
completely different on screen, so the stages are made steppable instead of
being collapsed into a score.

FIVE STAGES, cycled with the number keys:

  1  one view        that view's render with its 2D landmarks, beside the 3D
                     those same points lifted to. LEFT/RIGHT step views, so
                     a bad view is found by walking the sweep.
  2  all views       every view's lift at once, each view its own colour.
                     This is the convergence seen as a cloud: tight where
                     the views agree, a smear where they do not.
  3  one landmark    one landmark's estimate from every view, with a line
                     from each to the consensus, inliers solid and outliers
                     dark. UP/DOWN step landmarks. This is the only view
                     that shows WHY a point was accepted or rejected.
  4  consensus       the RANSAC result, coloured by how many views agreed.
  5  against truth   the consensus with the nearest surface point beside it
                     and a stick between them, which is the error with a
                     direction rather than a number.
  7  against the wrap   the consensus against the wrap's OWN landmark for the
                     same index -- the only reference here that has
                     correspondence. Stage 5 cannot see a point that slides
                     along the ear; this one can. [G] overlays the same
                     landmarks on any other stage.

STAGE 5 AND STAGE 7 DISAGREE, AND STAGE 7 IS THE ONE TO BELIEVE. Over 21
views of hutubs pp10_right the two rank the views with a correlation of
+0.35. The reference is the ear's own surface -- the wrap fits it to 0.002 mm
and the landmarks land 0.07 mm from the scan -- so stage 7's number is OUR
error outright, with nothing to deduct.

THE REFERENCE IS THE RE-MARKED SCHEME, mark_mean_v6_spacing: the hand-marked
curve with the shipped landmarker's spacing along it. The hand marks put the
same points at different places along each contour, which cost 2.5 mm of pure
convention; RANSAC on pp10_right reads 6.08 mm against them and 3.61 against
these. --gt-marks takes any other marks file, so the two can be compared on
screen.

EVERY GEOMETRY IS BUILT BEFORE THE WINDOW OPENS. An OffscreenRenderer cannot
be created while the GUI holds the Filament context -- it fails in the driver
and takes the process down -- so nothing here may render on demand. That is
also why the 2D overlays are drawn with PIL up front rather than by the
renderer.

    python scripts/view_pipeline.py --mv supersecret/mv/ear0 \
        --mesh supersecret/BJ002/BJ002.OBJ
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import pathlib
import sys

# MUST COME BEFORE `import open3d`, and it is not optional on a Wayland
# session. Open3D's interactive window is the legacy GLFW/GLEW path; on
# Wayland it fails outright and the window never appears. GLFW has to be
# pushed onto X11, which XWayland serves, and BOTH of these are needed --
# clearing WAYLAND_DISPLAY alone still fails. Setting them after a renderer
# has initialised is too late, hence module scope above the import. Copied
# deliberately from scripts/view_backprojection.py, where it was worked out.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np                                               # noqa: E402
import open3d as o3d                                             # noqa: E402
from PIL import Image, ImageDraw                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.draw import spheres                                   # noqa: E402
from ear3d.render import LINE_MATERIAL, MATERIAL, light_scene    # noqa: E402
from ear3d.scheme import (STRIPS, STRIP_COLOURS,                  # noqa: E402
                          STRIP_NAMES)

CONSENSUS_C = (0.20, 0.95, 0.65)
WEAK_C = (1.00, 0.45, 0.10)
TRUTH_C = (0.15, 0.50, 1.00)
STICK_C = (1.00, 0.75, 0.00)
# The wrap's landmarks are NOT the blue nearest-surface point and must not
# read as it: they are a different kind of claim about the same index.
GT_C = (0.90, 0.25, 0.95)
GT_STICK_C = (0.62, 0.12, 0.72)


def view_colour(i, n):
    """Distinct hue per view, so a cloud can be read back to its cameras."""
    return colorsys.hsv_to_rgb((i / max(n, 1)) * 0.85, 0.80, 0.95)


def lineset(pairs, colour):
    pts, idx, col = [], [], []
    for a, b in pairs:
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        idx.append([len(pts), len(pts) + 1])
        pts += [np.asarray(a, float), np.asarray(b, float)]
        col.append(colour)
    ls = o3d.geometry.LineSet()
    if not pts:
        return ls
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(idx))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(col, float))
    return ls


def strip_of(k):
    return next(i for i, (a, b) in enumerate(STRIPS) if a <= k < b)


def overlay_image(png, lm, note):
    """The view's render with its 2D landmarks on it, as an o3d Image.

    Drawn with PIL rather than the renderer because the renderer is not
    available once the window exists -- see the module docstring.
    """
    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, (a, b) in enumerate(STRIPS):
        c = tuple(int(255 * v) for v in STRIP_COLOURS[i])
        pts = [tuple(lm[k]) for k in range(a, b) if np.all(np.isfinite(lm[k]))]
        if len(pts) > 1:
            d.line(pts, fill=c, width=2)
    for k in range(len(lm)):
        if not np.all(np.isfinite(lm[k])):
            continue
        x, y = lm[k]
        c = tuple(int(255 * v) for v in STRIP_COLOURS[strip_of(k)])
        d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=c)
    d.text((8, 8), note, fill=(255, 255, 255))
    return o3d.geometry.Image(np.ascontiguousarray(np.asarray(im)))


def depth_panels(true_d, mg, mask, lm, mm_per_unit, pad=0.15):
    """MoGe's map, the renderer's exact buffer, and their difference.

    THE AFFINE IS FITTED OUTSIDE THE EAR. MoGe is free in global scale and
    offset, so the two maps cannot be differenced until one is put in the
    other's units -- but solving that on the ear would absorb the very relief
    the difference panel exists to show, and every model would look perfect.
    So scale and offset come from the valid pixels beyond the grown landmark
    box and are applied to the ear unchanged. Same choice as
    moge_lift_check.py, for the same reason, so the picture and that script's
    numbers mean the same thing.

    The two depth panels share one colour scale, because the whole point is
    whether the same structure appears in both; scaling each to its own range
    would make a flat map look detailed.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm

    H, W = true_d.shape
    valid = np.isfinite(true_d) & np.isfinite(mg) & mask
    (x0, y0), (x1, y1) = lm.min(0), lm.max(0)
    ew, eh = x1 - x0, y1 - y0
    yy, xx = np.mgrid[0:H, 0:W]
    grown = ((xx >= x0 - pad * ew) & (xx <= x1 + pad * ew) &
             (yy >= y0 - pad * eh) & (yy <= y1 + pad * eh))
    ear = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    fit = valid & ~grown
    if fit.sum() < 200 or (valid & ear).sum() < 200:
        return None, {}
    A = np.stack([mg[fit], np.ones(fit.sum())], 1)
    sc, sh = np.linalg.lstsq(A, true_d[fit], rcond=None)[0]
    maff = sc * mg + sh

    # colour scale from the EAR's true depths, so the ear is what the ramp
    # resolves rather than the whole head's range
    lo, hi = np.percentile(true_d[valid & ear], [2, 98])
    def ramp(a, m):
        n = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = (cm.viridis(n)[:, :, :3] * 255).astype(np.uint8)
        rgb[~m] = 0
        return rgb
    # SUBTRACT ONLY WHERE BOTH MAPS EXIST. The renderer writes inf for
    # background and MoGe's mask leaves inf where it declines to predict, so
    # a plain difference evaluates inf - inf and yields nan. The stats below
    # index down to valid pixels afterwards, so the result was already
    # correct -- but arithmetic on infinities is how a nan eventually reaches
    # something that does not mask, and it buried the real output in warnings.
    diff = np.zeros_like(true_d)
    np.subtract(maff, true_d, out=diff, where=valid)
    diff *= mm_per_unit
    lim = float(np.percentile(np.abs(diff[valid & ear]), 95)) or 1.0
    dn = np.clip(diff / (2 * lim) + 0.5, 0, 1)
    drgb = (cm.coolwarm(dn)[:, :, :3] * 255).astype(np.uint8)
    drgb[~valid] = 0

    panels = [ramp(true_d, valid), ramp(maff, valid), drgb]
    gap = np.zeros((H, 6, 3), np.uint8)
    comp = np.concatenate(
        [panels[0], gap, panels[1], gap, panels[2]], axis=1)
    im = Image.fromarray(comp)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(("renderer (exact)", "MoGe", "MoGe - truth")):
        d.text((8 + i * (W + 6), 8), t, fill=(255, 255, 255))
    d.text((8, H - 16), f"{lo * mm_per_unit:.0f}..{hi * mm_per_unit:.0f} mm "
                        f"shared ramp", fill=(255, 255, 255))
    d.text((8 + 2 * (W + 6), H - 16), f"+-{lim:.1f} mm", fill=(255, 255, 255))
    st = dict(
        resid=float(np.std(diff[fit])),
        ear_rms=float(np.sqrt(np.mean(diff[valid & ear] ** 2))),
        relief_t=float(np.std(true_d[valid & ear]) * mm_per_unit),
        relief_m=float(np.std(maff[valid & ear]) * mm_per_unit),
        scale=float(sc))
    return o3d.geometry.Image(np.ascontiguousarray(np.asarray(im))), st


def load(mv, mesh_path):
    mv = pathlib.Path(mv)
    tags = json.loads((mv / "views.json").read_text())
    C = np.load(mv / "consensus.npz", allow_pickle=True)
    kept = [str(t) for t in C["tags"]]
    cons, nin, est = C["cons"], C["nin"], C["est"]
    # est is indexed by the FULL view list, the consensus by the kept subset;
    # keeping both straight matters because a view with no depth map is in
    # one and not the other, and mixing them silently mislabels every colour.
    rows = [tags.index(t) for t in kept]
    views = []
    for t, r in zip(kept, rows):
        d = mv / f"view_{t}"
        F = np.load(d / "frame.npz", allow_pickle=True)
        views.append(dict(tag=t, row=r, lm=F["lm"], png=d / "render.png",
                          yaw=float(F["yaw"]), pitch=float(F["pitch"]),
                          depth=F["depth"].astype(float),
                          scale=float(F["scale"]), moge=d / "moge.npz"))
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    m.compute_vertex_normals()
    # render units -> scan units, the same chain moge_multiview inverts: a
    # length picks up the view's scale and the mesh normalisation, and
    # nothing else.
    Vm = np.asarray(m.vertices)
    s0 = float(np.abs(Vm - Vm.mean(0)).max())
    # MILLIMETRES PER SCAN UNIT, measured from the head rather than assumed.
    # BJ002 is in mm and the HUTUBS heads are in METRES, and with mm assumed
    # this window reported every distance as 0.00 mm and would have drawn
    # half-metre spheres. A head is 150-350 mm tall by any measure, so an
    # extent under 10 cannot be millimetres. Same rule as
    # moge_multiview.py, deliberately -- two different answers for one scan
    # would be worse than either.
    mmpu = 1000.0 if float(np.ptp(Vm, axis=0).max()) < 10.0 else 1.0
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    ok = np.isfinite(cons).all(1)
    surf = np.full_like(cons, np.nan)
    if ok.any():
        surf[ok] = sc.compute_closest_points(
            o3d.core.Tensor(cons[ok].astype(np.float32)))["points"].numpy()
    dist = np.linalg.norm(cons - surf, axis=1) * mmpu   # mm
    # WHICH ESTIMATOR AND WHICH TOLERANCE, from the file rather than assumed.
    # The combine step can export a median, a geometric median or a RANSAC
    # consensus, and a viewer that hard-codes one of those names will happily
    # label a median "RANSAC" -- which it did until this was read through.
    est_name = str(C["estimator"]) if "estimator" in C.files else "unknown"
    tol = float(C["tol"]) if "tol" in C.files else 2.0
    return dict(views=views, cons=cons, nin=nin, est=est, surf=surf,
                dist=dist, mesh=m, K=len(cons), spread=C["spread"],
                est_name=est_name, tol=tol, s0=s0, mmpu=mmpu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mv", required=True,
                   help="an output directory from moge_multiview.py --combine")
    p.add_argument("--mesh", required=True, help="the scan, in SCAN units")
    p.add_argument("--radius", type=float, default=0.55,
                   help="sphere radius IN MILLIMETRES, converted to scan "
                        "units from the head's measured size")
    p.add_argument("--size", type=int, default=1100)
    p.add_argument("--gt-subject", default="",
                   help="e.g. pp10 -- enables stage 7 from that ear's wrap "
                        "landmarks. --mesh must be the same head, in the "
                        "head's own frame.")
    p.add_argument("--gt-ear", default="right", choices=["left", "right"])
    p.add_argument("--gt-marks", default="",
                   help="a .wrap or .npz of template marks; the default is "
                        "mark_mean_v6_spacing, the re-marked scheme")
    p.add_argument("--gt-dataset", default="hutubs")
    p.add_argument("--stage", type=int, default=1,
                   choices=[1, 2, 3, 4, 5, 6, 7],
                   help="which stage to open on")
    p.add_argument("--landmark", type=int, default=0,
                   help="which landmark stage 3 opens on")
    p.add_argument("--selftest", action="store_true",
                   help="build every stage's geometry and exit without "
                        "opening a window, so the logic can be checked where "
                        "a GUI cannot run")
    a = p.parse_args()

    D = load(a.mv, a.mesh)
    V, K = D["views"], D["K"]
    rad = a.radius / D["mmpu"]          # mm -> scan units
    tol_scan = D["tol"] / D["mmpu"]
    nv = len(V)
    print(f"{nv} views, {K} landmarks, mesh "
          f"{len(D['mesh'].vertices):,} verts")
    f = np.isfinite(D["dist"])
    print(f"units:  {D['mmpu']:.0f} mm per scan unit\n"
          f"consensus ({D['est_name']}, tol {D['tol']:.1f} mm): median "
          f"{np.median(D['dist'][f]):.2f} mm to the surface, "
          f"{(D['nin'] >= 3).sum()}/{K} with 3+ supporting views")

    # ---- geometry for every stage, all of it up front --------------------
    g, depth_stats = {}, {}
    for i, v in enumerate(V):
        P = D["est"][v["row"]]
        col = [view_colour(i, nv)] * K
        g[f"view{i}_lm"] = (spheres(P, rad, col), MATERIAL)
        # kept as a (thing, material) pair like every geometry, with no
        # material, so one loop can walk the whole table
        if v["moge"].exists():
            M = np.load(v["moge"])
            img, st6 = depth_panels(v["depth"], M["depth"].astype(float),
                                    M["mask"].astype(bool), v["lm"],
                                    v["scale"] * D["s0"] * D["mmpu"])
            if img is not None:
                g[f"view{i}_depth"] = (img, None)
                depth_stats[i] = st6
        g[f"view{i}_img"] = (overlay_image(
            v["png"], v["lm"],
            f"view {v['tag']}  yaw {v['yaw']:+.0f} pitch {v['pitch']:+.0f}"),
            None)
    cons, nin, surf = D["cons"], D["nin"], D["surf"]
    g["cons"] = (spheres(cons, rad,
                         [CONSENSUS_C if nin[k] >= 3 else WEAK_C
                          for k in range(K)]), MATERIAL)
    g["truth"] = (spheres(surf, rad * 0.8, [TRUTH_C] * K), MATERIAL)
    g["sticks"] = (lineset(list(zip(cons, surf)), STICK_C), LINE_MATERIAL)

    # ---- the wrap's own landmarks, when a subject is named ---------------
    gt = gt_err = None
    if a.gt_subject:
        from gt_landmark_check import gt_landmarks
        G_mm, _, gt_src = gt_landmarks(a.gt_dataset, a.gt_subject,
                                       a.gt_ear, K,
                                       marks=a.gt_marks or None)
        # THE SAME HEAD, CHECKED. The wrap records which scan it was cut
        # from, and comparing a consensus with a different person's ear
        # would look like a plain accuracy failure rather than a mistake.
        if pathlib.Path(gt_src).resolve() != pathlib.Path(a.mesh).resolve():
            print(f"  WARNING the wrap was cut from {gt_src}\n"
                  f"          but --mesh is {a.mesh} -- same head?")
        gt = G_mm / D["mmpu"]           # head mm -> scan units
        gt_err = np.linalg.norm(cons - gt, axis=1) * D["mmpu"]
        g["gt"] = (spheres(gt, rad * 0.85, [GT_C] * K), MATERIAL)
        g["gt_sticks"] = (lineset(list(zip(cons, gt)), GT_STICK_C),
                          LINE_MATERIAL)
        e = gt_err[np.isfinite(gt_err)]
        print(f"wrap reference ({a.gt_subject}_{a.gt_ear}): paired median "
              f"{np.median(e):.2f} mm, p90 {np.percentile(e, 90):.2f}, "
              f"max {e.max():.2f}")
    rows = [v["row"] for v in V]
    for k in range(K):
        # one estimate per KEPT view, in the same order the colours use, so a
        # sphere's hue identifies its camera in every stage
        pts = np.asarray([D["est"][r, k, :] for r in rows])
        c = cons[k]
        # an inlier here means what RANSAC meant by it: within tol of the
        # consensus it ended up with. Recomputed rather than stored so the
        # picture cannot disagree with the number beside it.
        inl = (np.isfinite(pts).all(1)
               & (np.linalg.norm(pts - c, axis=1) <= tol_scan)
               ) if np.isfinite(c).all() else np.zeros(len(pts), bool)
        cl = [view_colour(j, nv) if inl[j] else (0.25, 0.25, 0.25)
              for j in range(nv)]
        g[f"lmk{k}_est"] = (spheres(pts, rad * 0.9, cl), MATERIAL)
        g[f"lmk{k}_cons"] = (spheres(np.asarray([c]), rad * 1.5,
                                     [CONSENSUS_C]), MATERIAL)
        g[f"lmk{k}_fan"] = (lineset([(q, c) for q in pts], (0.7, 0.7, 0.7)),
                            LINE_MATERIAL)

    if a.selftest:
        bad = [k for k, (geo, _) in g.items()
               if hasattr(geo, "vertices") and len(geo.vertices) == 0]
        empt = [k for k, (geo, _) in g.items()
                if hasattr(geo, "lines") and len(geo.lines) == 0]
        print(f"\nselftest: {len(g)} geometries built")
        print(f"  meshes with no vertices: {bad if bad else 'none'}")
        print(f"  linesets with no points: {len(empt)} "
              f"({'ok -- a landmark every view agreed on has no outliers' if empt else 'none'})")
        for st, keys in (("1 one view", ["view0_lm", "view0_img"]),
                         ("2 all views", [f"view{i}_lm" for i in range(nv)]),
                         ("3 one landmark", ["lmk0_est", "lmk0_cons",
                                             "lmk0_fan"]),
                         ("4 consensus", ["cons"]),
                         ("5 against truth", ["cons", "truth", "sticks"]),
                         ("6 depth map", ["view0_depth"]),
                         ("7 against wrap",
                          ["cons", "gt", "gt_sticks"] if gt is not None
                          else ["cons"])):
            miss = [k for k in keys if k not in g]
            print(f"  stage {st:<16s} {'OK' if not miss else 'MISSING ' + str(miss)}")
        print(f"\ninlier counts: min {nin.min()}, median "
              f"{np.median(nin):.0f}, max {nin.max()} of {nv}")
        return

    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering
    app = gui.Application.instance
    app.initialize()
    win = app.create_window("ear pipeline", a.size, int(a.size * 0.72))
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(win.renderer)
    widget.scene.add_geometry("mesh", D["mesh"], MATERIAL)
    # LIT FOR LOOKING AT, which is not how the renders are lit. light_scene
    # is a single hard sun, correct for the offscreen renders because it is
    # what the landmarker and the depth model were measured under -- changing
    # it there would move every number. Here it left whole sides of the head
    # black, and a landmark cannot be judged against a surface that is not
    # drawn. So the background comes from light_scene and the lighting is
    # then replaced by an image-based profile, which fills the shadow side
    # without inventing a second hard light that would read as geometry.
    light_scene(widget.scene, (0.18, 0.18, 0.20))
    widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS,
                              (-0.3, -0.4, -0.9))
    widget.scene.scene.set_indirect_light_intensity(45000)
    bbox = D["mesh"].get_axis_aligned_bounding_box()
    widget.setup_camera(50.0, bbox, cons[np.isfinite(cons).all(1)].mean(0)
                        if np.isfinite(cons).any() else bbox.get_center())

    info = gui.Label("")
    panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
    panel.background_color = gui.Color(0, 0, 0, 0.72)
    panel.add_child(info)
    img_w = gui.ImageWidget(g["view0_img"][0])
    win.add_child(widget)
    win.add_child(panel)
    win.add_child(img_w)

    st = {"stage": a.stage, "view": 0, "lmk": a.landmark % max(K, 1),
          "mesh": True, "gt": False, "drawn": []}

    def on_layout(ctx):
        r = win.content_rect
        widget.frame = r
        pref = panel.calc_preferred_size(ctx, gui.Widget.Constraints())
        panel.frame = gui.Rect(r.x, r.y, min(r.width, pref.width),
                               min(r.height, pref.height))
        # The 2D render sits bottom-right while stage 1 is showing. HIDDEN
        # with .visible, not collapsed to a 1x1 rect: the rect trick left a
        # degenerate primitive in the scene, and Filament then reported
        # "missing required attributes (0xd), declared=0x5" on the first
        # stage change -- one line, no crash, and nothing to say which
        # geometry it meant. A hidden widget is not submitted at all.
        # stage 6 is three panels side by side, so it gets a wide strip
        # across the bottom instead of the square thumbnail stage 1 uses
        img_w.visible = st["stage"] in (1, 6)
        if st["stage"] == 6:
            w = int(r.width * 0.96)
            h = int(w / 3.03)
            img_w.frame = gui.Rect(r.x + (r.width - w) // 2,
                                   r.get_bottom() - h - 8, w, h)
        elif img_w.visible:
            side = int(min(r.width, r.height) * 0.42)
            img_w.frame = gui.Rect(r.get_right() - side - 8,
                                   r.get_bottom() - side - 8, side, side)
    win.set_on_layout(on_layout)

    def show(keys):
        for n in st["drawn"]:
            widget.scene.remove_geometry(n)
        st["drawn"] = []
        for n in keys:
            geo, mat = g[n]
            if hasattr(geo, "points") and len(geo.points) == 0:
                continue
            if hasattr(geo, "vertices") and len(geo.vertices) == 0:
                continue
            widget.scene.add_geometry(n, geo, mat)
            st["drawn"].append(n)

    def redraw():
        s, vi, k = st["stage"], st["view"], st["lmk"]
        widget.scene.show_geometry("mesh", st["mesh"])
        if s == 1:
            show([f"view{vi}_lm"])
            v = V[vi]
            img_w.update_image(g[f"view{vi}_img"][0])
            txt = (f"STAGE 1  one view\n"
                   f"view {vi + 1}/{nv}   {v['tag']}   "
                   f"yaw {v['yaw']:+.0f}  pitch {v['pitch']:+.0f}\n"
                   f"LEFT/RIGHT change view")
        elif s == 2:
            show([f"view{i}_lm" for i in range(nv)])
            txt = (f"STAGE 2  all {nv} views at once\n"
                   f"one colour per view -- tight where they agree,\n"
                   f"a smear where they do not")
        elif s == 3:
            show([f"lmk{k}_fan", f"lmk{k}_est", f"lmk{k}_cons"])
            txt = (f"STAGE 3  one landmark's views\n"
                   f"landmark {k}  ({STRIP_NAMES[strip_of(k)]})\n"
                   f"{int(nin[k])}/{nv} views within {D['tol']:.1f} mm   "
                   f"spread {D['spread'][k]:.2f} mm\n"
                   f"grey spheres are outside it, big green is the "
                   f"{D['est_name']}\n"
                   f"UP/DOWN change landmark")
        elif s == 4:
            show(["cons"])
            txt = (f"STAGE 4  consensus ({D['est_name']})\n"
                   f"green = 3 or more views within {D['tol']:.1f} mm "
                   f"({(nin >= 3).sum()}/{K})\n"
                   f"orange = fewer ({(nin < 3).sum()}/{K})")
        elif s == 6:
            show([f"view{vi}_lm"])
            v = V[vi]
            key = f"view{vi}_depth"
            if key in g:
                img_w.update_image(g[key][0])
                t6 = depth_stats[vi]
                txt = (f"STAGE 6  the depth map itself\n"
                       f"view {vi + 1}/{nv}   {v['tag']}\n"
                       f"affine fitted OUTSIDE the ear (scale "
                       f"{t6['scale']:.3f}), residual {t6['resid']:.2f} mm\n"
                       f"relief across the ear: truth {t6['relief_t']:.2f} mm,"
                       f" MoGe {t6['relief_m']:.2f} mm "
                       f"(x{t6['relief_m'] / max(t6['relief_t'], 1e-9):.2f})\n"
                       f"rms over the ear {t6['ear_rms']:.2f} mm\n"
                       f"LEFT/RIGHT change view")
            else:
                txt = (f"STAGE 6  the depth map itself\n"
                       f"view {v['tag']} has no usable depth map")
        elif s == 7:
            if gt is None:
                show(["cons"])
                txt = ("STAGE 7  against the wrap\n"
                       "no --gt-subject was given, so there is no reference")
            else:
                show(["cons", "gt", "gt_sticks"])
                e = gt_err[np.isfinite(gt_err)]
                sd = D["dist"][np.isfinite(D["dist"])]
                txt = (f"STAGE 7  {D['est_name']} against the WRAP's "
                       f"landmarks\n"
                       f"violet = the wrap's own point for the SAME index\n"
                       f"median {np.median(e):.2f} mm   p90 "
                       f"{np.percentile(e, 90):.2f}   max {e.max():.2f}\n"
                       f"stage 5 says {np.median(sd):.2f} mm for the same "
                       f"points -- that metric cannot see a point\n"
                       f"that slid along the ear, and this one can\n"
                       f"the reference is the ear's own surface, so this "
                       f"is our error with nothing to deduct\n"
                       f"marks: {pathlib.Path(a.gt_marks).name if a.gt_marks else 'mark_mean_v6_spacing'}")
        else:
            show(["cons", "truth", "sticks"])
            d = D["dist"][np.isfinite(D["dist"])]
            txt = (f"STAGE 5  {D['est_name']} against the scan\n"
                   f"blue = nearest surface point, amber = the error\n"
                   f"median {np.median(d):.2f} mm   p90 "
                   f"{np.percentile(d, 90):.2f}   max {d.max():.2f}")
        # the reference as an OVERLAY, so convergence in stage 2 or a single
        # landmark's fan in stage 3 can be read against where the point
        # should have been, not just against where the others went
        if st["gt"] and gt is not None and s != 7:
            geo, mat = g["gt"]
            widget.scene.add_geometry("gt_overlay", geo, mat)
            st["drawn"].append("gt_overlay")
            txt += "\n[G] violet = the wrap's landmarks"
        info.text = txt + ("\n\n[1-7] stage   [M] mesh   [G] wrap   "
                           "[Q] quit")
        win.set_needs_layout()

    def on_key(e):
        if e.type != gui.KeyEvent.Type.DOWN:
            return False
        # The digit KeyNames ARE their ASCII codes, which is what makes this
        # readable; the arrows and letters have proper names.
        if ord("1") <= e.key <= ord("7"):
            st["stage"] = e.key - ord("0")
        elif e.key == gui.KeyName.RIGHT:
            st["view"] = (st["view"] + 1) % nv
            st["stage"] = 1
        elif e.key == gui.KeyName.LEFT:
            st["view"] = (st["view"] - 1) % nv
            st["stage"] = 1
        elif e.key == gui.KeyName.DOWN:
            st["lmk"] = (st["lmk"] + 1) % K
            st["stage"] = 3
        elif e.key == gui.KeyName.UP:
            st["lmk"] = (st["lmk"] - 1) % K
            st["stage"] = 3
        elif e.key == gui.KeyName.M:
            st["mesh"] = not st["mesh"]
        elif e.key == gui.KeyName.G:
            st["gt"] = not st["gt"]
        elif e.key in (gui.KeyName.Q, gui.KeyName.ESCAPE):
            gui.Application.instance.quit()
            return True
        else:
            return False
        redraw()
        return True
    win.set_on_key(on_key)

    redraw()
    app.run()


if __name__ == "__main__":
    main()
