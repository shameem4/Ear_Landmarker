"""Ask a VLM where the whole ear is, lobe included, even where hair hides it.

Text prompting SAM 3 does not fix the lobe shortfall: naming the lobe moves
the point-19 excess over mid-helix by less than the run-to-run spread, and
"auricle"/"pinna" are not in its concept vocabulary at all (0 masks on 156
rows). So the question becomes whether a VLM can supply the missing prior
as a BOX -- a region that reaches below the lobe -- which SAM 3 does accept
alongside the text.

Nothing here sees the landmarks, so the box is independent of the
annotation it will later be used to judge.

Backends (each needs its own interpreter, hence the two-step design --
this writes boxes, qa_sam3_boxed.py scores them):

  ollama  local qwen2.5vl:7b, nothing leaves the machine
  gemini  UPLOADS THE CROPS to Google; the user asked for it explicitly

    python scripts/qa_ear_box_vlm.py ollama runs/vlmbox ROWS.txt
"""
import base64
import csv
import io
import json
import pathlib
import re
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASK = ("Photo of a human ear. Report the bounding box of the ENTIRE outer"
       " ear: from the top of the helix down to the BOTTOM of the earlobe."
       " If hair, a hand or an earring hides part of the helix or the lobe,"
       " extrapolate the hidden outline and include it. Reply with JSON"
       " only, absolute pixel coordinates in the image as shown to you:"
       ' [{"bbox_2d": [x_left, y_top, x_right, y_bottom]}]')
# NB: the first version of this prompt carried "0.0 ... 1.0" as its format
# example and qwen2.5vl echoed it on every row -- 36 boxes, all of them the
# whole frame. Never put a usable value in a format example.
#
# Each backend is asked in ITS OWN documented convention. Asked for qwen's
# x-first "bbox_2d", Gemini silently answered y-first on some rows and
# x-first on others, which no parser can undo.
ASK_GEMINI = ASK.replace(
    'absolute pixel coordinates in the image as shown to you:'
    ' [{"bbox_2d": [x_left, y_top, x_right, y_bottom]}]',
    'as [{"box_2d": [ymin, xmin, ymax, xmax], "label": "ear"}] with the'
    ' coordinates normalised to 0-1000.')

UPSCALE = 448


def parse(txt, w, h, order="xyxy"):
    """Accept {"x0":..}, qwen's "bbox_2d" or gemini's y-first "box_2d"."""
    b = None
    m = re.search(r'"b?box_2d"\s*:\s*\[([^\]]*)\]', txt)
    if m:
        try:
            b = [float(v) for v in m.group(1).split(",")][:4]
        except Exception:
            b = None
    if b is None:
        m = re.search(r"\{[^{}]*\}", txt, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
            b = [float(d[k]) for k in ("x0", "y0", "x1", "y1")]
        except Exception:
            return None
    if len(b) != 4 or not all(v == v for v in b):
        return None
    if order == "yxyx":
        b = [b[1], b[0], b[3], b[2]]
    if max(b) > 1.5:
        # qwen2.5vl answers in a 0-1000 frame even when asked for pixels of
        # a 448px image, so anything past the true width must be /1000.
        if max(b) > max(w, h):
            b = [v / 1000.0 for v in b]
        else:
            b = [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
    b = [min(max(v, 0.0), 1.0) for v in b]
    if b[2] - b[0] < 0.05 or b[3] - b[1] < 0.05:
        return None
    return b


def prep(path):
    """Returns the PNG bytes and the size the VLM actually sees."""
    im = Image.open(path).convert("RGB")
    if max(im.size) < UPSCALE:
        k = UPSCALE / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue(), im.size


def ask_ollama(png):
    import requests
    r = requests.post("http://localhost:11434/api/generate", timeout=300,
                      json=dict(model="qwen2.5vl:7b", prompt=ASK, stream=False,
                                images=[base64.b64encode(png).decode()],
                                options=dict(temperature=0.0)))
    r.raise_for_status()
    return r.json().get("response", "")


def ask_gemini(png):
    from google import genai
    from google.genai import types
    global _CL
    try:
        _CL
    except NameError:
        key = (pathlib.Path.home() / ".config/gemini/api-key").read_text().strip()
        _CL = genai.Client(api_key=key)
    # Thinking is billed at the OUTPUT rate and is 86% of this task's output
    # tokens, for four integers. THINK=0 turns it off; measured identical
    # boxes to within a few parts in 1000, at 38 output tokens instead of 436.
    tc = None if THINK else types.ThinkingConfig(thinking_budget=0)
    o = _CL.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"),
                  ASK_GEMINI],
        config=types.GenerateContentConfig(temperature=0.0, thinking_config=tc))
    global TOK
    u = o.usage_metadata
    TOK[0] += u.prompt_token_count
    TOK[1] += u.total_token_count - u.prompt_token_count
    return o.text or ""


BACKEND = sys.argv[1]
OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
pick = [int(x) for x in pathlib.Path(sys.argv[3]).read_text().split()]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "gemini-3.8-flash"
THINK = int(sys.argv[5]) if len(sys.argv) > 5 else 1   # 0 = thinking off
TOK = [0, 0]
ask = dict(ollama=ask_ollama, gemini=ask_gemini)[BACKEND]

rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
box, bad = {}, 0
for c, i in enumerate(pick):
    try:
        png, (vw, vh) = prep(ROOT / "data/manual" / rows[i]["image_file"])
        txt = ask(png)
    except Exception as e:
        print(f"  {i}: {type(e).__name__} {e}"); bad += 1; continue
    b = parse(txt, vw, vh, "yxyx" if BACKEND == "gemini" else "xyxy")
    if b is None:
        bad += 1
        if bad < 4:
            print(f"  {i}: unparsable -> {txt[:120]!r}")
        continue
    box[str(i)] = b
    if (c + 1) % 20 == 0:
        print(f"  {c+1}/{len(pick)}  kept {len(box)}  bad {bad}", flush=True)

tag = BACKEND + ("" if THINK else "_nothink")
json.dump(box, open(OUT / f"boxes_{tag}.json", "w"))
print(f"\n{BACKEND}: {len(box)}/{len(pick)} boxes, {bad} unusable")
if TOK[0]:
    n = len(pick)
    print(f"  tokens: {TOK[0]/n:.0f} in, {TOK[1]/n:.0f} out per row"
          f"   -> 10,535 rows = "
          f"${TOK[0]/n*10535/1e6*0.75 + TOK[1]/n*10535/1e6*3.75:.2f} on"
          f" 3.8-flash, ${TOK[0]/n*10535/1e6*0.25 + TOK[1]/n*10535/1e6*1.50:.2f}"
          f" on 3.1-flash-lite")
if box:
    import statistics as st
    for k, j in (("width", 2), ("height", 3)):
        v = [box[s][j] - box[s][j - 2] for s in box]
        print(f"  median {k:7}{st.median(v):.3f}   "
              f"(1.0 = the whole crop, i.e. no information)")
