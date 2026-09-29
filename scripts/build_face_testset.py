"""SUPERSEDED by scripts/build_local_testset.py -- kept for the record.

BlazeEar's own data directory already holds 7,157 full-scene images with
ground-truth ear boxes (plus ~4,300 more across five other bundled datasets),
which is better than anything this downloads: real, diverse, local, and
annotated. This script yielded 14 detections from 56 images, of which only four
were usable modern ears -- the rest were marble busts and false positives on
painted faces. Use the local data instead.

Build a face test set spanning head rotations, and crop ears with BlazeEar.

Why: the ROI_EXPAND question could not be settled on the existing data. The
preprocessed test set is made of pre-cropped ear images, so it structurally
cannot evaluate wide crops, and the only real full frames available were four
shots of one subject at one distance. This assembles real full-frame faces at
varied head rotations so the crop-framing and stability analyses have a base
that is not one person.

SOURCING: Wikimedia Commons only, filtered to public-domain and CC licences, with
per-image attribution recorded in metadata.json. These are photographs of real,
often identifiable people, downloaded for local model testing. They are
gitignored and should not be redistributed; the licence field tells you what each
one actually permits if you ever want to.

NOTE: these images have NO ground-truth landmarks. They enable the
ground-truth-free analyses (prediction stability under crop jitter, detector
behaviour, duplicate rate) at proper scale and subject diversity. Settling crop
framing against true error still needs annotation.

Usage:
    python scripts/build_face_testset.py --download --limit 60
    python scripts/build_face_testset.py --detect        # run BlazeEar, crop ears
    python scripts/build_face_testset.py --download --detect
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "face_testset"
IMAGES = OUT / "images"
CROPS = OUT / "ear_crops"
META = OUT / "metadata.json"

UA = "EarLandmarkerResearch/1.0 (local model testing; contact: repo owner)"
API = "https://commons.wikimedia.org/w/api.php"

# Categories, not free-text search. A first attempt using text search returned
# overwhelmingly museum and historical material -- of 14 detections, six were
# marble Roman busts and several were false positives on painted or low-quality
# faces, leaving four usable modern ears. These categories are where photographs
# of living people with visible ears actually live.
CATEGORIES = [
    ("ear_closeup", "Human ears"),
    ("ear_context", "Earrings"),
    ("frontal", "Selfies"),
    ("varied", "Profile pictures"),
]

# Free-text searches kept as a supplement, biased toward modern photography.
QUERIES = [
    ("profile", "person profile photograph outdoors side view head"),
    ("three_quarter", "woman portrait photograph outdoors looking away"),
    ("three_quarter", "man portrait photograph outdoors looking away"),
]

ALLOWED_LICENCE = re.compile(
    r"(public domain|CC0|CC BY|CC-BY|Creative Commons)", re.IGNORECASE)
# Off: these images are only ever read locally for model testing and are
# gitignored. Licence and attribution are still recorded in metadata.json, so
# the provenance is there if any of this is ever shown outside the machine.
REQUIRE_OPEN_LICENCE = False

# The model is for photographs of living people. Marble, paint and print have
# the wrong texture and lighting entirely, and scanned book pages are not faces.
EXCLUDE_TITLE = re.compile(
    r"(museum|bust|statue|sculpture|marble|bronze|terracotta|relief|coin|medal|"
    r"engraving|etching|lithograph|painting|drawing|portrait of a|\.pdf|\.djvu|"
    r"camera work|plaster|mask|anatom|diagram|illustration|woodcut|fresco|mosaic|"
    r"figurine|effigy|memorial|tomb|manuscript|\bart\b)", re.IGNORECASE)


def _get(url: str, timeout: int = 40, tries: int = 5) -> bytes:
    """Fetch with exponential backoff. Commons rate-limits hard (HTTP 429) and
    a single unhandled 429 was aborting whole runs."""
    delay = 1.5
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 503):
                raise
        except Exception as e:
            last = e
        time.sleep(delay)
        delay *= 2
    raise last if last else RuntimeError("request failed")


def api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    return json.loads(_get(url))


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def search_files(query: str, limit: int) -> list[str]:
    d = api_get({
        "action": "query", "format": "json", "list": "search",
        "srsearch": query, "srnamespace": 6, "srlimit": limit,
    })
    return [x["title"] for x in d.get("query", {}).get("search", [])]


def category_files(category: str, limit: int) -> list[str]:
    """File titles in a Commons category (non-recursive)."""
    d = api_get({
        "action": "query", "format": "json", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmtype": "file", "cmlimit": limit,
    })
    return [x["title"] for x in d.get("query", {}).get("categorymembers", [])]


def file_info(titles: list[str]) -> list[dict]:
    """Image URL, size and licence for a batch of File: titles."""
    out = []
    for i in range(0, len(titles), 20):
        d = api_get({
            "action": "query", "format": "json", "titles": "|".join(titles[i:i + 20]),
            "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 900,
        })
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii.get("thumburl"):
                continue
            em = ii.get("extmetadata", {})
            out.append({
                "title": p.get("title", ""),
                "url": ii["thumburl"],
                "descriptionurl": ii.get("descriptionurl", ""),
                "width": ii.get("thumbwidth", 0),
                "height": ii.get("thumbheight", 0),
                "licence": strip_html(em.get("LicenseShortName", {}).get("value", "")),
                "artist": strip_html(em.get("Artist", {}).get("value", ""))[:120],
            })
        time.sleep(0.8)                      # be polite to the API
    return out


def download(limit: int) -> list[dict]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    seen, records = set(), []
    per_query = max(6, limit // len(QUERIES) + 4)

    sources = ([(p, c, "category") for p, c in CATEGORIES]
               + [(p, q, "search") for p, q in QUERIES])

    for pose, q, kind in sources:
        try:
            raw = (category_files(q, per_query) if kind == "category"
                   else search_files(q, per_query))
            titles = [t for t in raw if t not in seen and not EXCLUDE_TITLE.search(t)]
        except Exception as e:                # one bad query must not kill the run
            print(f"  {kind} failed ({q[:40]}...): {e}")
            continue
        for info in file_info(titles):
            if len(records) >= limit:
                break
            if REQUIRE_OPEN_LICENCE and not ALLOWED_LICENCE.search(info["licence"]):
                continue
            if info["width"] < 400 or info["height"] < 400:
                continue
            if info["title"] in seen or EXCLUDE_TITLE.search(info["title"]):
                continue
            seen.add(info["title"])

            name = re.sub(r"[^A-Za-z0-9]+", "_", info["title"][5:])[:60]
            ext = ".jpg" if info["url"].lower().endswith((".jpg", ".jpeg")) else ".png"
            path = IMAGES / f"{len(records):03d}_{pose}_{name}{ext}"
            try:
                data = _get(info["url"], timeout=60)
                if len(data) < 5000:
                    continue
                path.write_bytes(data)
            except Exception as e:
                print(f"  download failed {info['title'][:40]}: {e}")
                continue
            info.update({"pose_hint": pose, "file": str(path.relative_to(OUT))})
            records.append(info)
            print(f"  [{len(records):3d}] {pose:14s} {info['licence'][:28]:28s} {path.name[:48]}")
            time.sleep(0.5)
        if len(records) >= limit:
            break
    return records


def detect_and_crop(expand: float = 1.3, conf: float | None = None) -> None:
    """Run BlazeEar over the downloaded faces and save ear crops + boxes."""
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    blazeear_dir = Path(os.environ.get(
        "BLAZEEAR_DIR", ROOT.parent / "BlazeEar"))
    sys.path.insert(0, str(blazeear_dir))
    from blazeear import BlazeEar                       # type: ignore
    from utils.anchor_utils import anchor_options       # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlazeEar()
    if conf is not None:
        model.min_score_thresh = conf
    ckpt = torch.load(str(blazeear_dir / "runs/checkpoints/BlazeEar_best.pth"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.to(device).eval()
    model.generate_anchors(anchor_options)

    CROPS.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text()) if META.exists() else {"images": []}
    by_file = {m["file"]: m for m in meta["images"]}

    n_img = n_det = n_multi = 0
    for path in sorted(IMAGES.glob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        with torch.no_grad():
            dets = model.process(rgb)
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        dets = np.atleast_2d(np.asarray(dets)) if len(dets) else np.zeros((0, 17))
        n_img += 1
        if len(dets) > 1:
            n_multi += 1

        rec = by_file.get(str(path.relative_to(OUT)), {})
        rec["detections"] = []
        h, w = rgb.shape[:2]
        for k, d in enumerate(dets):
            ymin, xmin, ymax, xmax = [float(v) for v in d[:4]]
            c = float(d[-1])   # confidence is the LAST column of BlazeEar output
            bw, bh = xmax - xmin, ymax - ymin
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            side = max(bw, bh) * expand
            x1 = max(0, int(round(cx - side / 2)))
            y1 = max(0, int(round(cy - side / 2)))
            x2 = min(w, int(round(cx + side / 2)))
            y2 = min(h, int(round(cy + side / 2)))
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            crop_name = f"{path.stem}_ear{k}.png"
            Image.fromarray(rgb[y1:y2, x1:x2]).save(CROPS / crop_name)
            rec["detections"].append({
                "box": [xmin, ymin, xmax, ymax], "confidence": c,
                "crop": f"ear_crops/{crop_name}", "frame_size": [w, h],
            })
            n_det += 1
        by_file[str(path.relative_to(OUT))] = rec

    meta["images"] = list(by_file.values())
    meta["detect_summary"] = {
        "images": n_img, "detections": n_det,
        "images_with_multiple_detections": n_multi, "roi_expand": expand,
    }
    META.write_text(json.dumps(meta, indent=1))
    print(f"\n{n_img} images -> {n_det} ear detections "
          f"({n_multi} images had >1 detection)")
    print(f"crops in {CROPS}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a face/ear test set")
    p.add_argument("--download", action="store_true")
    p.add_argument("--detect", action="store_true")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--expand", type=float, default=1.3)
    p.add_argument("--confidence", type=float, default=None)
    args = p.parse_args()

    if not (args.download or args.detect):
        p.error("pass --download and/or --detect")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.download:
        print(f"downloading up to {args.limit} CC/PD face images from Wikimedia Commons")
        records = download(args.limit)
        meta = {"source": "Wikimedia Commons", "licence_filter": ALLOWED_LICENCE.pattern,
                "images": records}
        META.write_text(json.dumps(meta, indent=1))
        print(f"\n{len(records)} images -> {IMAGES}")
        print(f"attribution recorded in {META}")

    if args.detect:
        detect_and_crop(args.expand, args.confidence)


if __name__ == "__main__":
    main()
