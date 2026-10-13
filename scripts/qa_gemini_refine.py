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
    a = ap.parse_args()
    OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    tpl = json.load(open(ROOT / a.template))
    prompt = build_prompt(tpl)
    pick = [int(x) for x in a.rows.split(",")]
    key = (pathlib.Path.home() / ".config/gemini/api-key").read_text().strip()
    cl = genai.Client(api_key=key)
    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    tc = None if a.think else types.ThinkingConfig(thinking_budget=0)

    store, tin, tout, n = {}, 0, 0, 0
    print(f"{'row':>7}  " + "".join(f"{nm[:12]:>14}" for nm in STRIP_NAMES))
    for i in pick:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
        if max(im.size) < UPSCALE:
            k = UPSCALE / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)),
                           Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "PNG")
        o = cl.models.generate_content(
            model=a.model,
            contents=[types.Part.from_bytes(data=buf.getvalue(),
                                            mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(temperature=0.0,
                                               thinking_config=tc))
        u = o.usage_metadata
        tin += u.prompt_token_count
        tout += u.total_token_count - u.prompt_token_count
        n += 1
        got = parse(o.text or "")
        if np.isnan(got).all():
            print(f"{i:>7}  no points"); continue
        store[str(i)] = got.tolist()
        g = L[i]
        span = float(g[:20, 1].max() - g[:20, 1].min())
        cells = []
        for (lo, hi), nm in zip(STRIPS, STRIP_NAMES):
            d = np.linalg.norm(got[lo:hi] - g[lo:hi], axis=1) / span * 100
            cells.append("          n/a" if np.isnan(d).all()
                         else f"{np.nanmedian(d):>13.1f}")
        print(f"{i:>7}  " + "".join(cells))
    json.dump(store, open(OUT / f"refine_think{a.think}.json", "w"))
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
