"""Fit the canonical ear SHAPE to a photo with Gemini, all four contours.

Seeding the model with a row's own annotation would destroy the only
property worth having: a model that anchors on what it is shown blesses a
bad contour, which is the exact failure this work exists to catch. So the
seed is the canonical shape from qa_build_shape_template.py -- what an ear
looks like in general, never what THIS ear was annotated as. The output
stays independent of the annotation being judged, while template point j
still corresponds to landmark j, which a blind polygon cannot offer.

It asks for all four strips at once, so the concha border comes along with
the outer helix. The concha is the structure classical CV could never
reach (best of five channels, 1.03), so it is the real prize here.

Thinking is off by default: measured identical box quality at an eighth of
the output tokens. This task is harder than a box, so --think 1 is worth a
comparison run before trusting the default corpus-wide.

    python scripts/qa_gemini_refine.py OUT 852,113,155 --think 0
"""
import argparse
import csv
import io
import json
import pathlib
import re
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
# scheme.py directly, not via the package: ear3d/__init__ imports open3d,
# which the google-genai interpreter does not have.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_scheme", ROOT / "ear3d/scheme.py")
_scheme = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_scheme)
STRIPS, STRIP_NAMES = _scheme.STRIPS, _scheme.STRIP_NAMES

UPSCALE = 448
DESC = {
    "outer_helix": "the outer rim, from the top of the helix round the back"
                   " and down to the bottom of the earlobe",
    "inner_helix": "the inner edge of the helix fold, just inside the outer rim",
    "concha_border": "the boundary of the concha bowl: tragus, then the canal"
                     " opening, then the antitragus, then the concha floor,"
                     " then up the inferior crus",
    "superior_crus": "the superior crus of the antihelix, the short upper fork",
}


GT_PROMPT = """This photo shows one human ear.

Below is an EXISTING trace of its four contours, as [y, x] normalised to
0-1000. It is a trace OF THIS EAR, but it may be wrong: points may sit off
the ear, a contour may cut a corner or wander onto hair, cheek or jaw.

{pts}

Correct the placement. Move each point onto the structure it is meant to
mark, and leave it where it is if it is already right.

Rules:
- Return every point, in the same order and the same count per contour.
- Make each contour SMOOTH: no kinks or zigzags between neighbours, and
  keep the points roughly evenly spaced along it.
- Where hair, an earring, a hand or deep shadow hides a contour,
  EXTRAPOLATE where it would run and place the points there. Do not follow
  the occluder.

Reply with JSON only, coordinates normalised to 0-1000, y first:
{{"outer_helix": [[y,x], ...], "inner_helix": [...],
  "concha_border": [...], "superior_crus": [...]}}"""

ONE_PROMPT = """The FIRST image is a photo of one human {side} ear."""[:0] + """The FIRST image is a photo of one human {side} ear.

The SECOND image is a template: the average shape of a {side} ear's four
contours, with the point indices labelled. It is a template of what a
{side} ear looks like in general. It is NOT this ear.

Fit it to the photo: translate, scale, rotate and deform each contour so
it lies on the matching structure of THIS ear.

Rules:
- Return every point, in the same order and the same count per contour:
  outer_helix 20, inner_helix 15, concha_border 15, superior_crus 5.
- Point j must land on the structure the template labels j, so the
  contours keep their correspondence.
- Make each contour SMOOTH: no kinks or zigzags between neighbours, and
  keep the points roughly evenly spaced along it.
- Where hair, an earring, a hand or deep shadow hides a contour,
  EXTRAPOLATE where it would run and place the points there. Do not follow
  the occluder and do not bunch the points up at the edge of it.

Reply with JSON only, coordinates normalised to 0-1000, y first:
{{"outer_helix": [[y,x], ...], "inner_helix": [...],
  "concha_border": [...], "superior_crus": [...]}}"""

GTTPL_PROMPT = """The FIRST image is a photo of one human {side} ear.

The SECOND image is a template: the average shape of a {side} ear's four
contours, with the point indices labelled. It is what a {side} ear looks
like in general. It is NOT this ear.

Below is an EXISTING trace of THIS ear's four contours, as [y, x]
normalised to 0-1000. It may be wrong: points may sit off the ear, a
contour may cut a corner or wander onto hair, cheek or jaw.

{pts}

Correct the trace so that point j marks the same structure the template
labels j. Where the trace already does that, LEAVE THE POINT WHERE IT IS.
The template gives the correspondence and the general shape; the photo
decides the exact position. The template is not this ear's shape, so do
not pull the trace towards the template where the photo disagrees.

Rules:
- Return every point, in the same order and the same count per contour:
  outer_helix 20, inner_helix 15, concha_border 15, superior_crus 5.
- Make each contour SMOOTH: no kinks or zigzags between neighbours, and
  keep the points roughly evenly spaced along it.
- Where hair, an earring, a hand or deep shadow hides a contour,
  EXTRAPOLATE where it would run and place the points there. Do not follow
  the occluder and do not bunch the points up at the edge of it.

Reply with JSON only, coordinates normalised to 0-1000, y first:
{{"outer_helix": [[y,x], ...], "inner_helix": [...],
  "concha_border": [...], "superior_crus": [...]}}"""


IMG_PROMPT = """The FIRST image is a photo of one human ear.

The SECOND and THIRD images are templates: the average shape of an ear's
four contours, drawn for a RIGHT ear and for a LEFT ear, with the point
indices labelled. They are templates of what an ear looks like in general.
They are NOT this ear.

Pick whichever template matches the orientation of the ear in the photo,
then fit it: translate, scale, rotate and deform each contour so it lies
on the matching structure of THIS ear.

Rules:
- Return every point, in the same order and the same count per contour:
  outer_helix 20, inner_helix 15, concha_border 15, superior_crus 5.
- Point j must land on the structure the template labels j, so the
  contours keep their correspondence.
- Make each contour SMOOTH: no kinks or zigzags between neighbours, and
  keep the points roughly evenly spaced along it.
- Where hair, an earring, a hand or deep shadow hides a contour,
  EXTRAPOLATE where it would run and place the points there. Do not follow
  the occluder and do not bunch the points up at the edge of it.

Reply with JSON only, coordinates normalised to 0-1000, y first, and say
which template you used:
{"side": "right" or "left",
 "outer_helix": [[y,x], ...], "inner_helix": [...],
 "concha_border": [...], "superior_crus": [...]}"""


def build_prompt(tpl):
    lines = []
    for nm in STRIP_NAMES:
        s = tpl["strips"][nm]
        lines.append(f'"{nm}" (points {s["lo"]}-{s["hi"]-1}), {DESC[nm]}:\n'
                     f'  {json.dumps(s["points"])}')
    return f"""This photo shows one human ear.

Below is a CANONICAL ear: the average shape of each of its four contours,
as [y, x] with coordinates normalised to 0-1000. It is a template of what
an ear looks like in general. It is NOT this ear and its coordinates are
NOT where these contours lie in this photo.

{chr(10).join(lines)}

Fit this template to the ear in the photo. Translate, scale, rotate and
deform each contour so it lies on the matching structure of THIS ear. The
template is drawn for a right ear seen from the side; if this is the other
ear, mirror it horizontally.

Rules:
- Return every point, in the same order and the same count per contour.
- Point j must land on the structure the template puts at point j, so the
  contours keep their correspondence.
- Make each contour SMOOTH: no kinks or zigzags between neighbours, and
  keep the points roughly evenly spaced along it.
- Where hair, an earring, a hand or deep shadow hides a contour,
  EXTRAPOLATE where it would run and place the points there. Do not follow
  the occluder and do not bunch the points up at the edge of it.

Reply with JSON only, coordinates normalised to 0-1000, y first:
{{"outer_helix": [[y,x], ...], "inner_helix": [...],
  "concha_border": [...], "superior_crus": [...]}}"""


def parse(txt):
    out = np.full((55, 2), np.nan)
    for (lo, hi), nm in zip(STRIPS, STRIP_NAMES):
        m = re.search(rf'"{nm}"\s*:\s*\[(.*?)\]\s*(?:,\s*"|\}})', txt, re.S)
        if not m:
            continue
        p = re.findall(r'\[\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\]', m.group(1))
        if len(p) < hi - lo:
            continue
        v = np.array([[float(x), float(y)] for y, x in p[:hi - lo]]) / 1000.0
        out[lo:hi] = v
    return out


def main():
    from google import genai
    from google.genai import types
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("rows")
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--template", default="runs/shape_template/template.json")
    ap.add_argument("--seed",
                    choices=["numbers", "images", "both", "one", "gt", "gttpl"],
                    default="numbers")
    # The gt seed is NOT an audit -- it shows the model the answer. It is an
    # upper bound on achievable fit, and it is only interpretable next to
    # --corrupt, which breaks the seed in the way the real failures break:
    # the endpoint drift that started all this, plus a wander onto the cheek.
    ap.add_argument("--corrupt", action="store_true")
    a = ap.parse_args()
    OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    tpl = json.load(open(ROOT / a.template))
    tdir = (ROOT / a.template).parent
    TILE = {s_: (tdir / f"template_{s_}.png").read_bytes()
            for s_ in ("right", "left")}
    if a.seed in ("one", "gttpl"):
        prompt, tiles = None, None          # chosen per row below
    elif a.seed == "numbers":
        prompt, tiles = build_prompt(tpl), []
    else:
        tiles = [(tdir / f"template_{s_}.png").read_bytes()
                 for s_ in ("right", "left")]
        prompt = IMG_PROMPT if a.seed == "images" else (
            build_prompt(tpl) + "\n\nThe two images after the photo draw that"
            " same template, for a right ear and for a left ear, with the"
            " point indices labelled. Use whichever matches, and add"
            ' "side": "right" or "left" to your JSON.')
    pick = [int(x) for x in a.rows.split(",")]
    key = (pathlib.Path.home() / ".config/gemini/api-key").read_text().strip()
    cl = genai.Client(api_key=key)
    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    tc = None if a.think else types.ThinkingConfig(thinking_budget=0)

    store, tin, tout, n, sides = {}, 0, 0, 0, []
    SEEDS = {}
    print(f"{'row':>7}  " + "".join(f"{nm[:12]:>14}" for nm in STRIP_NAMES))
    for i in pick:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
        if max(im.size) < UPSCALE:
            k = UPSCALE / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)),
                           Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "PNG")
        g0 = L[i]
        gx, gy = g0[:20, 0], g0[:20, 1]
        area0 = 0.5 * np.sum(gx * np.roll(gy, -1) - np.roll(gx, -1) * gy)
        this_side = "right" if area0 < 0 else "left"
        if a.seed in ("gt", "gttpl"):
            seed = g0[:55].copy()
            if a.corrupt:
                span0 = float(g0[:20, 1].max() - g0[:20, 1].min())
                seed[16:20, 0] += 0.18 * span0 * (1 if i % 2 else -1)
                seed[16:20, 1] += 0.12 * span0
                seed[8:12, 0] += 0.10 * span0
                seed[43:47, 0] += 0.10 * span0      # push the concha floor
            blocks = []
            for (lo, hi), nm in zip(STRIPS, STRIP_NAMES):
                pts = [[round(float(y * 1000), 1), round(float(x * 1000), 1)]
                       for x, y in seed[lo:hi]]
                blocks.append(f'"{nm}" (points {lo}-{hi-1}):\n  '
                              + json.dumps(pts))
            if a.seed == "gttpl":
                # the correct-side template AND the trace: the template
                # supplies the index-to-structure correspondence, the photo
                # supplies the position. Asked explicitly NOT to pull the
                # trace towards the template, which is the failure the
                # numbers-only gt seed showed as an echo.
                use = [TILE[this_side]]
                body = GTTPL_PROMPT.format(side=this_side,
                                           pts="\n".join(blocks))
            else:
                use, body = [], GT_PROMPT.format(pts="\n".join(blocks))
            SEEDS[str(i)] = seed.tolist()
        elif a.seed == "one":
            use, body = [TILE[this_side]], ONE_PROMPT.format(side=this_side)
        else:
            use, body = tiles, prompt
        parts = [types.Part.from_bytes(data=buf.getvalue(),
                                       mime_type="image/png")]
        parts += [types.Part.from_bytes(data=t, mime_type="image/png")
                  for t in use]
        o = cl.models.generate_content(
            model=a.model,
            contents=parts + [body],
            config=types.GenerateContentConfig(temperature=0.0,
                                               thinking_config=tc))
        u = o.usage_metadata
        tin += u.prompt_token_count
        tout += u.total_token_count - u.prompt_token_count
        n += 1
        txt = o.text or ""
        got = parse(txt)
        ms = re.search(r'"side"\s*:\s*"(right|left)"', txt)
        said = ms.group(1) if ms else (this_side if a.seed in ("one", "gt", "gttpl")
                                       else "?")
        if np.isnan(got).all():
            print(f"{i:>7}  no points"); continue
        g = L[i]
        gx, gy = g[:20, 0], g[:20, 1]
        area = 0.5 * np.sum(gx * np.roll(gy, -1) - np.roll(gx, -1) * gy)
        truth = "right" if area < 0 else "left"   # matches the template build
        store[str(i)] = dict(points=got.tolist(), side=said, truth=truth)
        sides.append((said, truth))
        span = float(g[:20, 1].max() - g[:20, 1].min())
        cells = [f"{said[:1]}/{truth[:1]}  "]
        for (lo, hi), nm in zip(STRIPS, STRIP_NAMES):
            d = np.linalg.norm(got[lo:hi] - g[lo:hi], axis=1) / span * 100
            cells.append("          n/a" if np.isnan(d).all()
                         else f"{np.nanmedian(d):>13.1f}")
        print(f"{i:>7}  " + "".join(cells))
    tag = a.seed + ("_corrupt" if a.corrupt else "")
    json.dump(store, open(OUT / f"refine_{tag}_think{a.think}.json", "w"))
    if SEEDS:
        json.dump(SEEDS, open(OUT / f"seeds_{tag}.json", "w"))
    known = [(x, y) for x, y in sides if x != "?"]
    if known:
        agree = sum(x == y for x, y in known)
        print(f"\nhandedness: model named a side on {len(known)}/{len(sides)},"
              f" agreeing with the signed-area label on {agree}/{len(known)}")
    if n:
        print(f"\n{len(store)}/{n} ok   tokens {tin/n:.0f} in, {tout/n:.0f} out"
              f"   -> 10,535 rows = "
              f"${tin/n*10535/1e6*.75 + tout/n*10535/1e6*3.75:.2f}"
              f" on {a.model}")
        print("(cells are median distance from the ANNOTATION, % of ear height."
              "\n small on clean rows = agreement; large on flagged rows is the"
              "\n point, so render before reading them as error.)")


if __name__ == "__main__":
    main()
