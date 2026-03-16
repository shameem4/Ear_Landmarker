"""Unify ear landmark datasets into a single training-ready format.

Reads from 5 source datasets (3 formats), deduplicates by image hash,
normalizes 55 landmarks to [0,1], and writes fast-loading output.

Output (data/preprocessed/):
    images/          {idx:05d}.{ext} -- original-resolution ear crops
    landmarks.npy    (N, 55, 2) float32, normalized to [0,1]
    manifest.csv     provenance + metadata per sample

Usage:
    python data/preprocess.py
    python data/preprocess.py --skip-dedup   # keep all including hash-duplicates
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EAR_STUFF = Path("C:/Users/shame/OneDrive/Desktop/ear_stuff")
PROJECT = EAR_STUFF / "Ear Landmarker"
RAW_DIR = PROJECT / "data" / "raw"
PREP_DIR = PROJECT / "data" / "preprocessed"
IMG_DIR = PREP_DIR / "images"

NUM_LANDMARKS = 55

SOURCES = {
    "collectionA": {
        "path": str(EAR_STUFF / "scratch/old_earlandmarker/EarLandmarks/data/raw/collectionA"),
        "format": "pts",
        "description": "605 ear crops with 55-point .pts landmarks",
    },
    "collectionB": {
        "path": str(EAR_STUFF / "scratch/old_earlandmarker/EarLandmarks/data/raw/collectionB"),
        "format": "pts",
        "description": "6312 ear crops with 55-point .pts landmarks (includes flipped variants)",
    },
    "audioear2d": {
        "path": str(EAR_STUFF / "AudioEar2D.zip"),
        "format": "labelme_zip",
        "description": "2000 synthetic 299x299 ear crops from FFHQ, 55-point LabelMe JSON",
    },
    "audioear3d": {
        "path": str(EAR_STUFF / "AudioEar3D.zip"),
        "format": "labelme_zip",
        "description": "112 3D-scanned ear crops (187x186), 55-point LabelMe JSON",
    },
    "coco_keypoint": {
        "path": str(
            EAR_STUFF / "scratch/landmarker"
            / "Human ear with key point.v1i.coco"
        ),
        "format": "coco_kp",
        "description": "629 pre-cropped 224x224 ear images, 55 keypoints as 1x1 COCO bbox annotations",
    },
}

# ---------------------------------------------------------------------------
# Parsers (adapted from scratch/old_earlandmarker/shared/data/annotations.py)
# ---------------------------------------------------------------------------

def parse_pts_landmarks(filepath: Path) -> Optional[np.ndarray]:
    """Parse .pts file -> (55, 2) float32 pixel coords, or None."""
    coords: List[Tuple[float, float]] = []
    with open(filepath, "r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped[0] in {"{", "}"} or stripped.startswith(("version", "n_points")):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    arr = np.asarray(coords, dtype=np.float32)
    return arr if arr.shape == (NUM_LANDMARKS, 2) else None


def parse_labelme_landmarks(data: dict) -> Optional[np.ndarray]:
    """Parse LabelMe JSON shapes -> (55, 2) float32 pixel coords, or None."""
    shapes = sorted(data.get("shapes", []), key=lambda s: int(s.get("label", 0)))
    points: List[List[float]] = []
    for shape in shapes:
        points.extend(shape["points"])
    arr = np.asarray(points, dtype=np.float32)
    return arr if arr.shape == (NUM_LANDMARKS, 2) else None


def parse_coco_landmarks(
    image_id: int, annotations: List[Dict], num_landmarks: int = NUM_LANDMARKS
) -> Optional[np.ndarray]:
    """Parse landmarks from COCO annotations where each keypoint is a 1x1 bbox."""
    keypoints: Dict[int, Tuple[float, float]] = {}
    for ann in annotations:
        if ann.get("image_id") != image_id:
            continue
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if cat_id is None or bbox is None:
            continue
        x = bbox[0] + bbox[2] / 2.0
        y = bbox[1] + bbox[3] / 2.0
        idx = cat_id - 1  # category 1..55 -> index 0..54
        keypoints[idx] = (x, y)

    if len(keypoints) != num_landmarks:
        return None
    coords = np.zeros((num_landmarks, 2), dtype=np.float32)
    for i in range(num_landmarks):
        if i not in keypoints:
            return None
        coords[i] = keypoints[i]
    return coords


# ---------------------------------------------------------------------------
# Collector -- deduplicates, normalizes, writes images incrementally
# ---------------------------------------------------------------------------

class Collector:
    """Accumulates samples, deduplicates by image hash, writes images on add."""

    def __init__(self, dedup: bool = True):
        self.idx = 0
        self.manifest: List[Dict] = []
        self.landmarks: List[np.ndarray] = []
        self._seen: set = set()
        self._dedup = dedup
        self.dup_count = 0
        self.flip_skipped = 0

    def add(
        self,
        img_bytes: bytes,
        img_ext: str,
        lm_px: np.ndarray,
        width: int,
        height: int,
        source: str,
        original_path: str,
        is_flipped: bool = False,
        auto_crop: bool = False,
        crop_buffer: float = 0.15,
    ) -> bool:
        if is_flipped:
            self.flip_skipped += 1
            return False

        if self._dedup:
            h = hashlib.md5(img_bytes).hexdigest()
            if h in self._seen:
                self.dup_count += 1
                return False
            self._seen.add(h)

        # Auto-crop: when landmarks occupy a small fraction of the image,
        # crop to landmark bbox + buffer and re-normalize.
        if auto_crop:
            img_bytes, img_ext, lm_px, width, height = self._crop_to_landmarks(
                img_bytes, lm_px, width, height, crop_buffer,
            )
            if img_bytes is None:
                return False

        # Normalize landmarks to [0, 1]
        lm_norm = lm_px.copy()
        lm_norm[:, 0] /= max(width, 1)
        lm_norm[:, 1] /= max(height, 1)
        lm_norm = np.clip(lm_norm, 0.0, 1.0)

        fname = f"{self.idx:05d}{img_ext.lower()}"
        (IMG_DIR / fname).write_bytes(img_bytes)

        self.landmarks.append(lm_norm)
        self.manifest.append({
            "idx": self.idx,
            "image_file": f"images/{fname}",
            "source": source,
            "original_path": original_path,
            "width": width,
            "height": height,
        })
        self.idx += 1
        return True

    @staticmethod
    def _crop_to_landmarks(
        img_bytes: bytes, lm_px: np.ndarray, width: int, height: int, buffer: float
    ):
        """Crop image to landmark bounding box + buffer, return new bytes and landmarks."""
        xmin, ymin = lm_px.min(axis=0)
        xmax, ymax = lm_px.max(axis=0)
        bw, bh = xmax - xmin, ymax - ymin
        pad_x, pad_y = bw * buffer, bh * buffer
        x1 = max(0, int(xmin - pad_x))
        y1 = max(0, int(ymin - pad_y))
        x2 = min(width, int(xmax + pad_x))
        y2 = min(height, int(ymax + pad_y))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None, None, None, None, None
        img = Image.open(io.BytesIO(img_bytes))
        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=95)
        new_lm = lm_px.copy()
        new_lm[:, 0] -= x1
        new_lm[:, 1] -= y1
        return buf.getvalue(), ".jpg", new_lm, x2 - x1, y2 - y1

    def save(self) -> Tuple[int, ...]:
        lm = np.stack(self.landmarks).astype(np.float32)
        np.save(PREP_DIR / "landmarks.npy", lm)
        with open(PREP_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.manifest[0].keys()))
            writer.writeheader()
            writer.writerows(self.manifest)
        return lm.shape


# ---------------------------------------------------------------------------
# Source ingestors
# ---------------------------------------------------------------------------

def _find_image_for_stem(stem: str, directory: Path) -> Optional[Path]:
    """Find an image file matching a stem in the given directory."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = directory / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def ingest_pts_dir(c: Collector, name: str, path: Path, auto_crop: bool = False) -> None:
    """Ingest a directory of paired .pts + image files."""
    pts_files = sorted(path.glob("*.pts"))
    for pf in tqdm(pts_files, desc=name):
        lm = parse_pts_landmarks(pf)
        if lm is None:
            continue
        img_path = _find_image_for_stem(pf.stem, path)
        if img_path is None:
            continue
        img = Image.open(img_path)
        w, h = img.size
        c.add(
            img_path.read_bytes(), img_path.suffix, lm, w, h,
            name, str(img_path), is_flipped="_flipped" in pf.stem,
            auto_crop=auto_crop,
        )


def ingest_labelme_zip(c: Collector, name: str, zip_path: Path) -> None:
    """Ingest a zip of LabelMe JSON + image pairs."""
    with zipfile.ZipFile(zip_path) as zf:
        all_names = set(zf.namelist())
        jsons = sorted(n for n in all_names if n.endswith(".json"))
        for jf in tqdm(jsons, desc=name):
            with zf.open(jf) as f:
                data = json.load(f)
            lm = parse_labelme_landmarks(data)
            if lm is None:
                continue
            # Locate image in zip
            parent = str(Path(jf).parent)
            stem = Path(jf).stem
            img_entry = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = f"{parent}/{stem}{ext}" if parent != "." else f"{stem}{ext}"
                if cand in all_names:
                    img_entry = cand
                    break
            if img_entry is None:
                continue
            img_bytes = zf.read(img_entry)
            c.add(
                img_bytes, Path(img_entry).suffix, lm,
                data["imageWidth"], data["imageHeight"],
                name, f"{zip_path.name}::{jf}",
            )


def ingest_coco_kp(c: Collector, name: str, base: Path) -> None:
    """Ingest COCO-format keypoint annotations (each landmark = 1x1 bbox)."""
    json_files = list(base.rglob("_annotations.coco.json"))
    if not json_files:
        json_files = list(base.rglob("*.json"))

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            coco = json.load(f)
        if "annotations" not in coco:
            continue

        images = {img["id"]: img for img in coco["images"]}
        # Group annotations by image
        by_img: Dict[int, List[Dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("category_id", 0) >= 1:
                by_img[ann["image_id"]].append(ann)

        img_dir = jf.parent
        for img_id in tqdm(sorted(by_img.keys()), desc=f"{name}/{jf.parent.name}"):
            lm = parse_coco_landmarks(img_id, by_img[img_id])
            if lm is None:
                continue
            info = images.get(img_id)
            if info is None:
                continue
            img_file = img_dir / info["file_name"]
            if not img_file.exists():
                # Search subdirectories
                found = list(img_dir.rglob(info["file_name"]))
                img_file = found[0] if found else img_file
            if not img_file.exists():
                continue
            c.add(
                img_file.read_bytes(), img_file.suffix, lm,
                info["width"], info["height"],
                name, str(img_file),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unify ear landmark datasets")
    parser.add_argument("--skip-dedup", action="store_true", help="Keep hash-duplicate images")
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Write provenance
    with open(RAW_DIR / "sources.json", "w", encoding="utf-8") as f:
        json.dump(SOURCES, f, indent=2)
    print(f"Provenance written to {RAW_DIR / 'sources.json'}")

    c = Collector(dedup=not args.skip_dedup)

    # collectionA: full images, not ear crops -- auto-crop to landmark bbox + 15% buffer
    ingest_pts_dir(c, "collectionA", Path(SOURCES["collectionA"]["path"]), auto_crop=True)
    # collectionB: pre-cropped 256x256 ear images
    ingest_pts_dir(c, "collectionB", Path(SOURCES["collectionB"]["path"]))
    ingest_labelme_zip(c, "audioear2d", Path(SOURCES["audioear2d"]["path"]))
    ingest_labelme_zip(c, "audioear3d", Path(SOURCES["audioear3d"]["path"]))
    # coco_keypoint: DROPPED -- 224x224 crops are valid but point ordering does not
    # match linestrip convention (0-19 helix, 20-34, 35-49, 50-54) used by other sources.
    # Mean step distances 2-4x higher than other sources; no known mapping exists.

    if c.idx == 0:
        print("No samples collected. Check source paths.")
        return

    shape = c.save()

    # Summary
    counts = Counter(r["source"] for r in c.manifest)
    print(f"\n{'=' * 50}")
    print(f"landmarks.npy shape: {shape}")
    print(f"Duplicates removed:  {c.dup_count}")
    print(f"Flipped skipped:     {c.flip_skipped}")
    print(f"{'-' * 50}")
    for src, n in counts.most_common():
        print(f"  {src:20s} {n:>6d}")
    print(f"  {'TOTAL':20s} {c.idx:>6d}")
    print(f"{'=' * 50}")
    print(f"Output: {PREP_DIR}")


if __name__ == "__main__":
    main()
