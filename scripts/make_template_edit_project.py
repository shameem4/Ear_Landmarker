#!/usr/bin/env python
"""Build a Faceform Wrap project for marking template polygons to exclude.

The deliverable is the SAVED OBJ, not a selection file: SaveGeom writes the
trimmed template, and the kept vertices are recovered afterwards by exact
coordinate match (scripts/ingest_trimmed_template.py).  That keeps this
independent of SavePolygons' parameter schema, and it survives an exporter
that reorders vertices -- the one thing that would otherwise cost us the
55 marks, the fitted corpus and the landmark-vertex numbering.

Reference overlays, not wired into the chain:
  landmarks   octahedra at the 55 landmark vertices -- do not cut here
  suggestion  the faces the automatic dot < -0.6 rule would remove
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ear3d.wrap import load_mesh, vertex_normals        # noqa: E402

WRAP = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")


def write_obj(path, V, F=None):
    with open(path, "w") as fh:
        for p in V:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        if F is not None:
            for t in F:
                fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def blobs(P, r):
    """Octahedra at P, as one mesh -- visible in Wrap's viewport."""
    o = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * r
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    V = np.vstack([p + o for p in P])
    F = np.vstack([f + 6 * i for i in range(len(P))])
    return V, F


def node(nid, ntype, params):
    return {"isEnabled": True, "nodeId": nid, "nodeType": ntype,
            "params": params}


def fn(p):
    return {"fileName": {"value": str(pathlib.Path(p).resolve())}}


# Wrap validates a project against the full parameter set and refuses the file
# if any are absent, so every node carries its defaults.  WrapCmd names the
# missing ones, which is how these were established rather than guessed.
LOADGEOM_DEFAULTS = {
    "rotation": {"value": {"x": 0, "y": 0, "z": 0}},
    "translation": {"value": {"x": 0, "y": 0, "z": 0}},
    "scale": {"value": 1},
    "texture": {},
}
SAVEGEOM_DEFAULTS = {
    "dracoCompression": {"value": 0}, "dracoNormalQuant": {"value": 8},
    "dracoTexCoordQuant": {"value": 10}, "dracoVertexQuant": {"value": 11},
}


def visual(r, g, b):
    return {"visualParam": {"value": {
        "colorFront": {"r": r, "g": g, "b": b},
        "colorBack": {"r": 23, "g": 23, "b": 23},
        "colorWire": {"r": 17, "g": 26, "b": 34},
        "isLightingEnabled": True, "isSurfaceEnabled": True,
        "isWireframeEnabled": True, "surfaceFillType": 1}}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=str(WRAP / "templates/ear_lmv_v3.obj"))
    p.add_argument("--n-landmarks", type=int, default=55)
    p.add_argument("--suggest-dot", type=float, default=-0.6,
                   help="write the faces this normal test would remove, as a "
                        "reference overlay only")
    p.add_argument("--out", default=str(WRAP / "template_edit"))
    p.add_argument("--selection", default=None,
                   help="JSON list of polygon indices to preload into "
                        "SelectPolygons. Valid across templates that share "
                        "connectivity, which replaying the promotion recipe "
                        "guarantees.")
    p.add_argument("--reference", default=str(WRAP / "ear_wrap_pp12_left.wrap"),
                   help="a project this Wrap build loads; its formatVersion "
                        "and settings are copied rather than hardcoded -- a "
                        "wrong formatVersion makes Wrap attempt a format "
                        "conversion and refuse the file")
    a = p.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    V, F = load_mesh(a.template)
    k = a.n_landmarks
    scale = float(np.ptp(V, axis=0).max())
    print(f"template {pathlib.Path(a.template).name}: {len(V)}v {len(F)}f; "
          f"landmarks are vertices 0..{k - 1}")

    # landmark markers -- vertices 0..k-1 ARE the landmarks in ear_lmv_v1
    Lv, Lf = blobs(V[:k], 0.008 * scale)
    write_obj(out / "landmarks.obj", Lv, Lf)

    # the automatic suggestion, for reference
    N = vertex_normals(V, F)
    f = N[:k].mean(0)       # landmarks ARE vertices 0..k-1 in ear_lmv_v1
    f /= np.linalg.norm(f)
    dot = N @ f
    back = (dot <= a.suggest_dot)[F].any(1)
    write_obj(out / "suggestion_back_faces.obj", V, F[back])
    print(f"suggestion overlay: {back.sum()}/{len(F)} faces at dot <= "
          f"{a.suggest_dot}; landmarks inside it: "
          f"{int((dot[:k] <= a.suggest_dot).sum())}")

    preset = sorted(int(x) for x in json.load(open(a.selection))) if a.selection else []
    if preset:
        print(f"preloaded selection: {len(preset)} polygons")
    saved = out / (pathlib.Path(a.template).stem + "_front.obj")
    nodes = {
        "template":     node(0, "LoadGeom",
                             {**fn(a.template), **LOADGEOM_DEFAULTS,
                              **visual(200, 200, 200)}),
        "landmarks":    node(1, "LoadGeom",
                             {**fn(out / "landmarks.obj"), **LOADGEOM_DEFAULTS,
                              **visual(40, 220, 40)}),
        "suggestion":   node(2, "LoadGeom",
                             {**fn(out / "suggestion_back_faces.obj"),
                              **LOADGEOM_DEFAULTS, **visual(230, 60, 60)}),
        "SelectPolygons": node(3, "SelectPolygons",
                               {"source": {"value": 0},
                                "geom": {"connectedNodeId": 0},
                                "displayMode": {"value": 0},
                                "fileName": {"value": ""},
                                "invisiblePolygonIndices": {"value": []},
                                "selection": {"value": preset}}),
        "DeletePolygons": node(4, "DeletePolygons",
                               {"geom": {"connectedNodeId": 0},
                                "polygons": {"connectedNodeId": 3},
                                "delete": {"value": 0}}),
        "SaveGeom":     node(5, "SaveGeom",
                             {"geom": {"connectedNodeId": 4},
                              "createDirectories": {"value": True},
                              "includeTransform": {"value": True},
                              "saveNormals": {"value": False},
                              **SAVEGEOM_DEFAULTS, **fn(saved)}),
    }
    xy = [(-40, 20), (220, 20), (220, 110), (-40, 140),
          (-40, 250), (-40, 360)]
    ref = json.load(open(a.reference))
    proj = {
        "commonDir": "",
        "formatVersion": ref["formatVersion"],
        # the GUI validates layout more strictly than WrapCmd does and needs
        # layoutFormatVersion, so the layout envelope is copied too
        "layout": {
            **{k: v for k, v in ref["layout"].items() if k != "items"},
            "items": [
                {"itemId": i, "itemType": "node", "nodeId": n["nodeId"],
                 "x": xy[i][0], "y": xy[i][1]}
                for i, n in enumerate(nodes.values())],
            # 3D view: the selection node, plus the two reference overlays
            "shown3d": [3, 1, 2],
        },
        "metadata": {},
        "nodes": nodes,
        "pythonCode": "",
        "settings": ref.get("settings", {}),
        "timeline": ref.get("timeline", {"current": 0, "max": 0, "min": 0}),
    }
    wp = out / "template_edit.wrap"
    json.dump(proj, open(wp, "w"), indent=1)

    (out / "README.txt").write_text(f"""Template polygon exclusion
==========================

Open:  {wp}

Nodes
  template     {pathlib.Path(a.template).name}  ({len(V)}v {len(F)}f)  grey
  landmarks    green octahedra at the {k} landmark vertices -- DO NOT CUT HERE
  suggestion   red: the {int(back.sum())} faces an automatic normal test
               (dot <= {a.suggest_dot}) would remove.  Reference only; it is
               crude in both directions -- it eats concha wall that merely
               faces inward, and it keeps posterior surface that happens to
               face sideways.
  SelectPolygons   <-- mark the polygons to EXCLUDE here
  DeletePolygons   removes them
  SaveGeom         writes {saved.name}

What to do
  1. Hide 'suggestion' once you have the idea of it (it overlaps the
     template and will fight it in the viewport).
  2. In SelectPolygons, select the polygons to exclude -- the posterior
     surface and anywhere hair bridges to the head.
  3. Keep clear of the green markers.  A landmark left ON a free boundary
     is unconstrained during the fit and flaps; that is what ruled out the
     earlier trim 2.  Leave a couple of rings of polygons around each.
  4. Compute SaveGeom.

Constraint that matters
  DELETE FACES ONLY.  No re-meshing, no decimation, no welding, no moving
  vertices, no subdivision.  The kept vertices are then recovered by exact
  coordinate match, which preserves:
    - the 55 marks and the landmark-vertex numbering (vertex j == landmark j)
    - the 476-ear fitted corpus (remapped, not re-fitted)
    - the PCA basis and base_topology
  If vertices get merged or nudged, that mapping is gone and the corpus has
  to be re-fit from scratch.

Then
  {pathlib.Path(__file__).parent}/ingest_trimmed_template.py \\
      --template {a.template} --trimmed {saved}
  It verifies face-only editing, reports which landmarks survive and their
  margin to the new boundary, and writes the keep-vertex/keep-face maps.
""")
    print(f"wrote {wp}")
    print(f"      {out / 'README.txt'}")
    print(f"save target: {saved}")


if __name__ == "__main__":
    main()
