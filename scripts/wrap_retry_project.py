#!/usr/bin/env python3
"""Build a Faceform Wrap project that steps through the fits that failed.

Wrap reads a FRAME SEQUENCE: a LoadGeom whose filename contains #### loads
scan_0001.obj, scan_0002.obj and so on as you scrub the timeline, and
SelectPointPairs keeps a separate set of picks per frame. So the scans that the
automatic pipeline could not handle become one project: place control points on
each frame, compute, done, and the results come back named by frame.

The failures are CHOSEN BY MEASUREMENT, not by hand. Three signals, any of which
condemns a fit:

  left against right   the two ears of one subject should agree closely; when
                       they do not, one of the two fits is wrong
  distance from mean   a fit that slid is far from the corpus average in a
                       direction no real ear goes
  stretch              how far the template had to distort to get there

Scans that never produced a cut at all are included too, as whole heads.

The project is CLONED from a working one rather than written from scratch:
Wrap's format carries settings this tool has no business inventing, and a
hand-built file is rejected by the format converter.

    python scripts/wrap_retry_project.py --base ear_wrap_pp12_left.wrap \\
        --out retry/ --wrapped corpus/wrapped_hutubs corpus/wrapped_sonicom \\
        --ears corpus/ears_hutubs corpus/ears_sonicom
"""
import argparse
import csv
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import load_mesh, rigid_fit                  # noqa: E402


def flag_failures(wrapped_dirs, lr_pct, mean_pct):
    """Return {name: reason} for fits that fail any of the measured checks."""
    files, V, names = [], [], []
    for d in wrapped_dirs:
        files += sorted(pathlib.Path(d).glob("*.obj"))
    if not files:
        return {}
    V0, _ = load_mesh(files[0])
    V = np.empty((len(files), len(V0), 3))
    for i, f in enumerate(files):
        Vi, _ = load_mesh(f)
        if len(Vi) != len(V0):
            raise SystemExit(f"{f.name}: topology differs from {files[0].name}")
        V[i] = Vi
        names.append(f.stem)

    A = V.copy()
    mean = A[0]
    for _ in range(3):
        for i in range(len(A)):
            R, t, s = rigid_fit(A[i], mean)
            A[i] = s * A[i] @ R.T + t
        mean = A.mean(0)
    size = float(np.ptp(mean, axis=0).max())
    dist = np.linalg.norm(A - mean, axis=2).mean(axis=1) / size * 100

    bad = {}
    for i, n in enumerate(names):
        if dist[i] > mean_pct:
            bad[n] = f"{dist[i]:.1f}% from the corpus mean"

    pairs = {}
    for i, n in enumerate(names):
        if "_" in n:
            subj, side = n.rsplit("_", 1)
            if side in ("left", "right"):
                pairs.setdefault(subj, {})[side] = i
    for subj, v in pairs.items():
        if len(v) != 2:
            continue
        R, t, s = rigid_fit(A[v["right"]], A[v["left"]])
        e = np.linalg.norm(A[v["left"]] - (s * A[v["right"]] @ R.T + t), axis=1)
        lr = 100 * float(np.median(e)) / size
        if lr > lr_pct:
            for side in ("left", "right"):
                bad.setdefault(f"{subj}_{side}", f"left/right disagree by {lr:.1f}%")
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="a working .wrap to clone")
    p.add_argument("--out", required=True)
    p.add_argument("--wrapped", nargs="*", default=[])
    p.add_argument("--ears", nargs="+", required=True,
                   help="where the cut ears live, to source the scans from")
    p.add_argument("--extra", nargs="*", default=[],
                   help="meshes to include outright, e.g. scans that never cut")
    p.add_argument("--lr-pct", type=float, default=5.0)
    p.add_argument("--mean-pct", type=float, default=6.0)
    p.add_argument("--template", default=None,
                   help="template mesh for the clone's LoadGeom. The base "
                        "project's path is usually stale, and it must be a "
                        "template already in the CUT FRAME -- the original "
                        "ear_L_template.obj has its axes the other way round "
                        "and would start 90 degrees out.")
    p.add_argument("--done", nargs="*", default=[],
                   help="ears already wrapped by hand; they are dropped from the "
                        "project so it only ever contains work left to do")
    p.add_argument("--name", default="retry")
    p.add_argument("--target-size", type=float, default=80.0,
                   help="each frame is scaled to this max extent, in whatever "
                        "units the template uses")
    a = p.parse_args()

    out = pathlib.Path(a.out)
    seq = out / "scans"
    seq.mkdir(parents=True, exist_ok=True)

    bad = flag_failures(a.wrapped, a.lr_pct, a.mean_pct) if a.wrapped else {}
    print(f"{len(bad)} fits flagged by measurement")
    for d in a.done:
        if bad.pop(d, None) is not None:
            print(f"  {d}: already wrapped by hand, dropped")

    # resolve each flagged name to the cut ear it came from
    frames = []
    for name, why in sorted(bad.items()):
        src = None
        for d in a.ears:
            c = pathlib.Path(d) / f"{name}.obj"
            if c.exists():
                src = c
                break
        if src is None:
            print(f"  {name}: no cut ear found, skipping")
            continue
        frames.append((name, src, why))
    for e in a.extra:
        e = pathlib.Path(e)
        frames.append((e.stem, e, "never produced a cut"))

    if not frames:
        sys.exit("nothing to retry")

    # Wrap numbers frames from the pattern, so the scans are copied to a
    # contiguous ####-named sequence and a manifest records what each one is.
    #
    # Every frame is also rescaled to one size. A sequence shares a single
    # LoadGeom scale, and these scans do not share units -- HUTUBS heads are in
    # metres, its cut ears in normalised head units, SONICOM in millimetres --
    # so without this the template would be the wrong size on most frames and
    # the point pairs would be placed against a speck or a wall. The factor is
    # recorded per frame, so the fitted result maps straight back.
    import trimesh
    rows = []
    for i, (name, src, why) in enumerate(frames, start=1):
        m = trimesh.load(src, process=False, force="mesh")
        V, F = np.asarray(m.vertices), np.asarray(m.faces)
        # RIGHT EARS ARE MIRRORED INTO LEFT FORM, as the automatic fitting does
        # internally. Without this the template is the opposite hand from half
        # the frames and no amount of control points will make it fit -- the
        # first version of this project shipped that way and only the left-ear
        # frames could be wrapped. The axis is z; see wrap_corpus.mirror_to_left.
        mirrored = name.endswith("_right")
        if mirrored:
            V = V * np.array([1.0, 1.0, -1.0])
            F = F[:, ::-1]                 # or the normals point inward
        # target_size <= 0 means the meshes are already in a common, anatomical
        # scale (bisected halves and SONICOM heads both are), and rescaling them
        # by mesh EXTENT would undo that -- a whole head normalised to the size
        # of an ear leaves the ear a few units across.
        k = (a.target_size / float(np.ptp(V, axis=0).max())
             if a.target_size > 0 else 1.0)
        trimesh.Trimesh(vertices=V * k, faces=F, process=False).export(
            seq / f"scan_{i:04d}.obj")
        rows.append([i, name, str(src), round(k, 6), "yes" if mirrored else "no", why])
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "name", "source", "scale_applied", "mirrored", "why"])
        w.writerows(rows)

    d = json.load(open(a.base))
    nodes = list(d["nodes"].values())
    rig = next((n for n in nodes if n["nodeType"] == "RigidAlignment"), None)
    scan_id = rig["params"]["geomFixed"]["connectedNodeId"]
    tpl_id = rig["params"]["geomFloating"]["connectedNodeId"]
    for n in nodes:
        if n["nodeType"] == "LoadGeom" and n["nodeId"] == tpl_id and a.template:
            n["params"]["fileName"]["value"] = str(pathlib.Path(a.template).resolve())
            n["params"]["scale"]["value"] = 1
        if n["nodeType"] == "LoadGeom" and n["nodeId"] == scan_id:
            n["params"]["fileName"]["value"] = str(seq / "scan_####.obj")
            n["params"]["scale"]["value"] = 1
        if n["nodeType"] == "SaveGeom":
            n["params"]["fileName"]["value"] = str(out / "wrapped_####.obj")
        if n["nodeType"] == "SelectPointPairs":
            # the picks belong to the old scan; they are placed per frame now
            n["params"]["pointsRight"]["value"] = []
    d["timeline"] = {"current": 1, "min": 1, "max": len(frames)}

    proj = out / f"{a.name}.wrap"
    json.dump(d, open(proj, "w"), indent=4)
    print(f"\nwrote {proj}")
    print(f"  {len(frames)} frames, scans in {seq}")
    print(f"  results will be written as {out / 'wrapped_####.obj'}")
    print(f"  frame -> subject mapping in {out / 'manifest.csv'}")
    for r in rows:
        print(f"    frame {r[0]:3d}  {r[1]:<18s} x{r[3]:<10.4g} "
              f"{'mirrored' if r[4] == 'yes' else '        '}  {r[5]}")


if __name__ == "__main__":
    main()
