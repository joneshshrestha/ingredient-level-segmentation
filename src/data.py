"""FoodSeg103 dataset loading, label-map derivation, splits, and sanity checks.

Design goals (driven by the project's accuracy requirements):
  * Nothing about the number of classes is hardcoded. `id2label` and
    `num_labels` are DERIVED from `category_id.txt` and the actual mask pixel
    values, with any mismatch reported loudly rather than hidden.
  * The official FoodSeg103 zip layout is auto-detected, but every path is
    configurable.
  * FoodSeg103 ships train/test only -> a seeded validation split is carved
    from train; the official test split is reserved for final evaluation.
  * Masks are loaded as integer class ids (never RGB), 0 = background.
  * `do_reduce_labels` is OFF by default. When on, existing ignore pixels are
    protected so 255 can never be shifted to 254.

Run the sanity check directly:
    python -m src.data --config configs/segformer_b0.yaml [--full-scan]
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_transforms
from .utils import get_logger, load_yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
MASK_EXTS = (".png", ".PNG")
log = get_logger("foodseg.data")


# ===========================================================================
# Layout detection
# ===========================================================================
@dataclass
class Layout:
    root: Path
    img_dir: Path           # contains <split>/ subfolders of images
    ann_dir: Path           # contains <split>/ subfolders of masks
    imagesets_dir: Optional[Path]
    label_map_file: Optional[Path]
    splits: List[str] = field(default_factory=lambda: ["train", "test"])


def detect_layout(cfg: dict) -> Layout:
    root = Path(cfg["dataset_root"]).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"dataset_root does not exist: {root}\n"
            "Download/place FoodSeg103 first (see README) or fix dataset_root.")

    # Image/mask dirs: honor overrides, else try the official locations.
    def _resolve(sub_override, candidates):
        if sub_override:
            return root / sub_override
        for c in candidates:
            if (root / c).is_dir():
                return root / c
        return root / candidates[0]  # fall back to first (will error later if absent)

    img_dir = _resolve(cfg.get("image_subdir"),
                       ["Images/img_dir", "images/img_dir", "img_dir", "Images"])
    ann_dir = _resolve(cfg.get("mask_subdir"),
                       ["Images/ann_dir", "images/ann_dir", "ann_dir", "Annotations"])

    imagesets = None
    for c in ["ImageSets", "imagesets", "splits"]:
        if (root / c).is_dir():
            imagesets = root / c
            break

    label_map = find_label_map_file(root, cfg.get("label_map_file"))
    return Layout(root=root, img_dir=img_dir, ann_dir=ann_dir,
                  imagesets_dir=imagesets, label_map_file=label_map)


def find_label_map_file(root: Path, override: Optional[str]) -> Optional[Path]:
    if override:
        p = Path(override)
        p = p if p.is_absolute() else root / p
        return p if p.exists() else None
    for name in ["category_id.txt", "categories.txt", "category_id.csv", "labels.txt"]:
        hits = list(root.rglob(name))
        if hits:
            return sorted(hits, key=lambda x: len(str(x)))[0]
    return None


# ===========================================================================
# Label map
# ===========================================================================
def parse_label_map(path: Optional[Path]) -> Tuple[Dict[int, str], List[str]]:
    """Parse `category_id.txt` into {id: name}. Tolerant of tab/space/CSV and
    header lines. Returns (id2label_raw, notes)."""
    notes: List[str] = []
    if path is None or not Path(path).exists():
        notes.append("No label-map file found; names will fall back to class_<id>.")
        return {}, notes

    id2label: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # split on tab first, then comma, then any whitespace
            if "\t" in line:
                parts = line.split("\t", 1)
            elif "," in line and line.split(",", 1)[0].strip().lstrip("-").isdigit():
                parts = line.split(",", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            head, name = parts[0].strip(), parts[1].strip()
            if not head.lstrip("-").isdigit():
                continue  # header row like "id name"
            id2label[int(head)] = name

    if id2label:
        ids = sorted(id2label)
        notes.append(f"Parsed {len(id2label)} label names from {path.name} "
                     f"(id range {ids[0]}..{ids[-1]}).")
        if ids[0] == 1:
            notes.append("WARNING: label-map ids start at 1. If masks contain id 0 "
                         "(background), this file may be 1-indexed -> names could be "
                         "off-by-one vs mask values. VERIFY before training.")
    else:
        notes.append(f"Could not parse any id/name rows from {path}.")
    return id2label, notes


def build_label_space(id2label_raw: Dict[int, str], observed: List[int],
                      cfg: dict) -> Tuple[Dict[int, str], Dict[str, int], int, List[str]]:
    """Reconcile parsed names with observed mask values into a final label space
    aligned to mask pixel ids. Returns (id2label, label2id, num_labels, notes)."""
    notes: List[str] = []
    forced = cfg.get("num_labels")
    max_file = max(id2label_raw) if id2label_raw else -1
    max_obs = max(observed) if observed else -1

    if forced:
        num_labels = int(forced)
        notes.append(f"num_labels FORCED to {num_labels} via config (override).")
    else:
        num_labels = max(max_file, max_obs) + 1
        notes.append(f"num_labels DERIVED = {num_labels} "
                     f"(max label-map id={max_file}, max observed mask id={max_obs}).")

    if max_obs >= num_labels:
        notes.append(f"WARNING: observed mask id {max_obs} >= num_labels {num_labels}; "
                     "predictions for those pixels are impossible. Check the label map.")

    id2label = {i: id2label_raw.get(i, f"class_{i}") for i in range(num_labels)}
    missing_named = [v for v in observed if v not in id2label_raw and v < num_labels]
    if missing_named:
        notes.append(f"{len(missing_named)} observed class id(s) have no name in the "
                     f"label map; using class_<id> for: {missing_named[:10]}"
                     f"{'...' if len(missing_named) > 10 else ''}")

    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id, num_labels, notes


def apply_label_reduction(mask: np.ndarray, background_id: int,
                          ignore_index: int) -> np.ndarray:
    """ADE20K-style reduction (only when do_reduce_labels=True).

    background_id -> ignore_index, every other label shifted down by 1.
    Existing ignore pixels are PROTECTED: they stay ignore_index and are never
    decremented (so 255 never becomes 254). Assumes background_id == 0.
    """
    if background_id != 0:
        raise ValueError("do_reduce_labels assumes background_id == 0 for FoodSeg103.")
    out = mask.copy()
    ignore_pixels = mask == ignore_index
    bg_pixels = mask == background_id
    shift_pixels = ~ignore_pixels & ~bg_pixels
    out[bg_pixels] = ignore_index
    out[shift_pixels] = mask[shift_pixels] - 1
    out[ignore_pixels] = ignore_index  # explicit: protected, unchanged
    return out


def reduce_label_space(id2label: Dict[int, str], background_id: int
                       ) -> Tuple[Dict[int, str], Dict[str, int], int]:
    """Rebuild the label space after reduction: drop background, shift ids down."""
    n = len(id2label)
    new = {i: id2label[i + 1] for i in range(n - 1)}
    return new, {v: k for k, v in new.items()}, n - 1


# ===========================================================================
# Pairing / splits
# ===========================================================================
def _index_by_stem(folder: Path, exts) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            out.setdefault(p.stem, p)
    return out


def _read_imageset(imagesets_dir: Optional[Path], split: str) -> Optional[List[str]]:
    if imagesets_dir is None:
        return None
    for name in (f"{split}.txt", f"{split}.lst"):
        p = imagesets_dir / name
        if p.exists():
            stems = []
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                tok = line.split()[0]
                stems.append(Path(tok).stem)
            return stems
    return None


def list_pairs(layout: Layout, split: str) -> Tuple[List[Tuple[Path, Path]], dict]:
    """Return matched (image, mask) pairs for a split plus a diagnostics dict."""
    img_split = layout.img_dir / split
    ann_split = layout.ann_dir / split
    img_idx = _index_by_stem(img_split, IMAGE_EXTS)
    ann_idx = _index_by_stem(ann_split, MASK_EXTS)

    listed = _read_imageset(layout.imagesets_dir, split)
    if listed is not None:
        stems = listed
        source = f"ImageSets/{split}.txt ({len(stems)} entries)"
    else:
        stems = sorted(set(img_idx) | set(ann_idx))
        source = f"folder glob ({img_split.name}/, {ann_split.name}/)"

    pairs, missing_img, missing_mask = [], [], []
    for s in stems:
        ip, mp = img_idx.get(s), ann_idx.get(s)
        if ip is None:
            missing_img.append(s)
        elif mp is None:
            missing_mask.append(s)
        else:
            pairs.append((ip, mp))

    orphan_masks = sorted(set(ann_idx) - set(img_idx)) if listed is None else []
    diag = {
        "split": split, "source": source,
        "n_images": len(img_idx), "n_masks": len(ann_idx), "n_pairs": len(pairs),
        "missing_images": missing_img, "missing_masks": missing_mask,
        "orphan_masks": orphan_masks,
        "img_split_dir": str(img_split), "ann_split_dir": str(ann_split),
    }
    return pairs, diag


def split_train_val(pairs: List[Tuple[Path, Path]], val_fraction: float, seed: int
                    ) -> Tuple[List, List]:
    """Deterministic train/val split of the official train pairs."""
    pairs = sorted(pairs, key=lambda x: x[0].stem)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs))
    n_val = int(round(len(pairs) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val


# ===========================================================================
# Mask scanning
# ===========================================================================
def load_mask(path) -> np.ndarray:
    """Load a segmentation mask as a 2D int64 array of class ids."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        log.warning("Mask %s is multi-channel (%s); using channel 0.",
                    Path(path).name, arr.shape)
        arr = arr[..., 0]
    return arr.astype(np.int64)


def scan_mask_values(pairs: List[Tuple[Path, Path]], sample: Optional[int],
                     ignore_index: int) -> Tuple[List[int], Dict[int, int], int]:
    """Scan masks for unique class ids and per-class pixel counts.

    sample=None scans all; otherwise scans the first `sample` masks.
    Returns (sorted_unique_excluding_ignore, class_pixel_counts, n_scanned).
    """
    subset = pairs if sample is None else pairs[:sample]
    counts: Dict[int, int] = {}
    uniques = set()
    for _, mp in subset:
        m = load_mask(mp)
        vals, cnts = np.unique(m, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            counts[v] = counts.get(v, 0) + c
            uniques.add(v)
    uniques.discard(ignore_index)
    return sorted(uniques), counts, len(subset)


# ===========================================================================
# Dataset
# ===========================================================================
class FoodSeg103Dataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], cfg: dict, train: bool,
                 num_labels: int):
        self.pairs = pairs
        self.cfg = cfg
        self.train = train
        self.num_labels = num_labels
        self.ignore_index = cfg.get("ignore_index", 255)
        self.background_id = cfg.get("background_id", 0)
        self.do_reduce_labels = bool(cfg.get("do_reduce_labels", False))
        self.transform = build_transforms(cfg, train=train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = load_mask(mask_path)
        orig_h, orig_w = mask.shape[:2]

        if self.do_reduce_labels:
            mask = apply_label_reduction(mask, self.background_id, self.ignore_index)

        pixel_values, labels = self.transform(image, mask)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "orig_size": (orig_h, orig_w),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "image_paths": [b["image_path"] for b in batch],
        "mask_paths": [b["mask_path"] for b in batch],
        "orig_sizes": [b["orig_size"] for b in batch],
    }


# ===========================================================================
# Build datasets (shared label space)
# ===========================================================================
def build_label_metadata(cfg: dict, layout: Layout,
                         train_pairs: List[Tuple[Path, Path]],
                         scan_sample: Optional[int],
                         allow_sample_without_labelmap: bool = False) -> dict:
    """Derive id2label / num_labels from the label map + observed mask values."""
    id2label_raw, notes = parse_label_map(layout.label_map_file)
    # Without a label map we normally MUST scan all masks to find the full id
    # range. In smoke mode we allow sampling (label space may be incomplete).
    if id2label_raw or allow_sample_without_labelmap:
        sample = scan_sample
        if not id2label_raw:
            notes.append("WARNING: no label map AND sampled scan -> num_labels may be "
                         "incomplete. Provide category_id.txt or run --full-scan for real runs.")
    else:
        sample = None
    observed, counts, n_scanned = scan_mask_values(
        train_pairs, sample, cfg.get("ignore_index", 255))
    notes.append(f"Scanned {n_scanned} train mask(s); observed ids: "
                 f"{observed[:12]}{'...' if len(observed) > 12 else ''}")

    id2label, label2id, num_labels, more = build_label_space(id2label_raw, observed, cfg)
    notes += more

    do_reduce = bool(cfg.get("do_reduce_labels", False))
    if do_reduce:
        id2label, label2id, num_labels = reduce_label_space(
            id2label, cfg.get("background_id", 0))
        notes.append(f"do_reduce_labels=True -> reduced to {num_labels} classes "
                     "(background folded into ignore_index).")

    return {
        "id2label": id2label, "label2id": label2id, "num_labels": num_labels,
        "observed_ids": observed, "class_pixel_counts": counts, "notes": notes,
    }


def make_datasets(cfg: dict, scan_sample: Optional[int] = 300,
                  allow_sample_without_labelmap: bool = False) -> dict:
    """Build train/val/test datasets sharing one derived label space."""
    layout = detect_layout(cfg)
    train_all, train_diag = list_pairs(layout, "train")
    test_pairs, test_diag = list_pairs(layout, "test")
    if not train_all:
        raise RuntimeError(f"No train image/mask pairs found.\n  {train_diag}")

    meta = build_label_metadata(cfg, layout, train_all, scan_sample,
                                allow_sample_without_labelmap)
    num_labels = meta["num_labels"]

    train_pairs, val_pairs = split_train_val(
        train_all, cfg.get("val_split_fraction", 0.1), cfg.get("seed", 42))

    datasets = {
        "train": FoodSeg103Dataset(train_pairs, cfg, train=True, num_labels=num_labels),
        "val": FoodSeg103Dataset(val_pairs, cfg, train=False, num_labels=num_labels),
        "test": FoodSeg103Dataset(test_pairs, cfg, train=False, num_labels=num_labels),
    }
    meta.update({"layout": layout, "train_diag": train_diag, "test_diag": test_diag,
                 "n_train": len(train_pairs), "n_val": len(val_pairs),
                 "n_test": len(test_pairs)})
    return {"datasets": datasets, "meta": meta}


# ===========================================================================
# Sanity check
# ===========================================================================
def run_sanity_check(cfg: dict, full_scan: bool = False) -> dict:
    log.info("=" * 72)
    log.info("FoodSeg103 DATA SANITY CHECK")
    log.info("=" * 72)
    layout = detect_layout(cfg)
    log.info("dataset_root : %s", layout.root)
    log.info("img_dir      : %s", layout.img_dir)
    log.info("ann_dir      : %s", layout.ann_dir)
    log.info("ImageSets    : %s", layout.imagesets_dir)
    log.info("label_map    : %s", layout.label_map_file)

    train_pairs, train_diag = list_pairs(layout, "train")
    test_pairs, test_diag = list_pairs(layout, "test")
    for diag in (train_diag, test_diag):
        log.info("-" * 72)
        log.info("[%s] source=%s", diag["split"], diag["source"])
        log.info("  images=%d masks=%d matched_pairs=%d",
                 diag["n_images"], diag["n_masks"], diag["n_pairs"])
        if diag["missing_masks"]:
            log.warning("  %d image(s) with NO mask, e.g. %s",
                        len(diag["missing_masks"]), diag["missing_masks"][:5])
        if diag["missing_images"]:
            log.warning("  %d listed/mask stem(s) with NO image, e.g. %s",
                        len(diag["missing_images"]), diag["missing_images"][:5])
        if diag["orphan_masks"]:
            log.warning("  %d orphan mask(s) with no image, e.g. %s",
                        len(diag["orphan_masks"]), diag["orphan_masks"][:5])

    if not train_pairs:
        log.error("No train pairs found -- cannot derive label space. Check paths above.")
        return {"ok": False, "train_diag": train_diag, "test_diag": test_diag}

    sample = None if full_scan else 300
    id2label_raw, notes = parse_label_map(layout.label_map_file)
    observed, counts, n_scanned = scan_mask_values(
        train_pairs, None if not id2label_raw else sample, cfg.get("ignore_index", 255))
    id2label, label2id, num_labels, more = build_label_space(id2label_raw, observed, cfg)

    log.info("-" * 72)
    for n in notes + more:
        log.info("  %s", n)
    log.info("  DERIVED num_labels = %d", num_labels)
    ignore_index = cfg.get("ignore_index", 255)
    log.info("  ignore_index = %s (present in scanned masks: %s)",
             ignore_index, ignore_index in counts)
    log.info("  do_reduce_labels = %s (default False is correct for FoodSeg103)",
             cfg.get("do_reduce_labels", False))

    # split sizes
    train_pairs_s, val_pairs_s = split_train_val(
        train_pairs, cfg.get("val_split_fraction", 0.1), cfg.get("seed", 42))
    log.info("  splits: train=%d  val=%d (held out from train)  test=%d",
             len(train_pairs_s), len(val_pairs_s), len(test_pairs))

    # class distribution summary (imbalance flag)
    total_px = sum(counts.get(v, 0) for v in observed)
    log.info("-" * 72)
    log.info("  Class distribution over %d scanned mask(s) (top 10 by pixels):",
             n_scanned)
    ranked = sorted(observed, key=lambda v: counts.get(v, 0), reverse=True)
    for v in ranked[:10]:
        share = 100.0 * counts.get(v, 0) / total_px if total_px else 0.0
        log.info("    id %-4d %-24s %8.3f%%  (%d px)",
                 v, id2label.get(v, f"class_{v}"), share, counts.get(v, 0))
    rare = [v for v in observed if total_px and counts.get(v, 0) / total_px < 1e-4]
    if rare:
        log.info("  %d very rare class(es) (<0.01%% of pixels) -> class imbalance: %s",
                 len(rare), rare[:15])

    log.info("=" * 72)
    log.info("SANITY CHECK COMPLETE. Verify the label range/names above before training.")
    log.info("(Pass --full-scan to scan every mask instead of a 300-mask sample.)")
    log.info("=" * 72)
    return {
        "ok": True, "num_labels": num_labels, "id2label": id2label,
        "observed_ids": observed, "n_train": len(train_pairs_s),
        "n_val": len(val_pairs_s), "n_test": len(test_pairs),
        "train_diag": train_diag, "test_diag": test_diag,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FoodSeg103 data sanity check.")
    ap.add_argument("--config", default="configs/segformer_b0.yaml")
    ap.add_argument("--dataset-root", default=None, help="override dataset_root")
    ap.add_argument("--full-scan", action="store_true",
                    help="scan every mask (slower) instead of a 300-mask sample")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.dataset_root:
        cfg["dataset_root"] = args.dataset_root
    run_sanity_check(cfg, full_scan=args.full_scan)


if __name__ == "__main__":
    main()
