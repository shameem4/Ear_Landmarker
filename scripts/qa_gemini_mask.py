"""Get an ear SHAPE, not just a box, out of Gemini.

A box is four numbers; the QA question needs a contour. The obvious
conversion -- feed the box to SAM 3 -- inherits exactly the lobe shortfall
the box was meant to escape (qa_sam3_boxed.py: 1.51 -> 1.68, no change).
But Gemini will emit a segmentation mask directly, from the same model
whose boxes DO reach the lobe (median IoU 0.909 against annotated extent).

The mask arrives as COCO compressed RLE in the "mask" key, NOT the base64
PNG the older docs describe, so it is decoded here rather than with PIL.
pycocotools is not installed in `mocap` and this is twenty lines, so the
decoder is inline: counts are LEB128-style 5-bit groups, delta-coded
against the value two back, and the runs are column-major starting with
background.

    python scripts/qa_gemini_mask.py OUT_DIR row,row,row

Needs the google-genai interpreter, not mocap:
    /mnt/.../exrcise_trainer/.venv/bin/python
"""
import csv
import io
import json
import pathlib
import re
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASK = ("Give the segmentation masks for the human ear, including the earlobe."
       " Output a JSON list of segmentation masks where each entry contains"
       ' the 2D bounding box in the key "box_2d", the segmentation mask in'
       ' key "mask", and the text label in the key "label".')


def rle_counts(s):
    """COCO compressed-RLE string -> run lengths."""
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_to_mask(s, h, w):
    """Runs are column-major and start with background."""
    c = rle_counts(s)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for n in c:
        if n < 0:
            return None
        flat[pos:pos + n] = val
        pos += n
        val ^= 1
        if pos > h * w:
            return None
    return flat.reshape((w, h)).T          # column-major


def ask(cl, types, png, model):
    o = cl.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"), ASK],
        config=types.GenerateContentConfig(temperature=0.0))
    return o.text or "", o.usage_metadata


def main():
    from google import genai
    from google.genai import types
    OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    picks = [int(x) for x in sys.argv[2].split(",")]
    model = sys.argv[3] if len(sys.argv) > 3 else "gemini-3.8-flash"
    key = (pathlib.Path.home() / ".config/gemini/api-key").read_text().strip()
    cl = genai.Client(api_key=key)
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))

    store, tok = {}, 0
    for i in picks:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
        if max(im.size) < 448:
            k = 448 / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)),
                           Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "PNG")
        txt, u = ask(cl, types, buf.getvalue(), model)
        tok += u.total_token_count
        mm = re.search(r'"mask"\s*:\s*"([^"]*)"', txt)
        bb = re.search(r'"box_2d"\s*:\s*\[([^\]]*)\]', txt)
        if not mm or not bb:
            print(f"  {i}: no mask in reply"); continue
        b = [float(v) for v in bb.group(1).split(",")]
        y0, x0, y1, x1 = [v / 1000.0 for v in b]       # gemini is y-first
        store[str(i)] = dict(rle=mm.group(1), box=[x0, y0, x1, y1],
                             sent=list(im.size))
        print(f"  {i}: mask {len(mm.group(1))} chars, box "
              f"[{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}]")
    json.dump(store, open(OUT / "gemini_masks.json", "w"))
    print(f"\n{len(store)} masks, {tok} tokens "
          f"({tok/max(len(store),1):.0f}/row)")


if __name__ == "__main__":
    main()
