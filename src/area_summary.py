"""Approximate *visible pixel-area* composition summaries from a predicted mask.

IMPORTANT SCOPE / DISCLAIMER
----------------------------
These percentages describe the share of *visible image pixels* assigned to each
ingredient class. They are a 2D projection of the visible food surface and are
**NOT**:
  - calorie percentages,
  - macronutrient / nutrition percentages,
  - food weight, mass, volume, or density,
  - real portion sizes.
A thin sauce drizzle and a dense pile of rice can occupy similar pixel areas
while differing enormously in calories/weight. Treat these numbers only as a
coarse "what is visible, and roughly how much of the picture" summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

DISCLAIMER = (
    "Visible pixel-area % is the share of visible image pixels per ingredient. "
    "It is NOT calories, nutrition, weight, volume, or portion size."
)


def compute_area_percentages(
    mask: np.ndarray,
    id2label: Optional[Dict[int, str]] = None,
    background_id: int = 0,
    ignore_index: int = 255,
    exclude_background: bool = True,
) -> pd.DataFrame:
    """Per-class visible pixel-area summary for a single mask.

    Returns a DataFrame sorted by descending pixel count with columns:
      class_id, class_name, pixels, pct_of_image, pct_of_food
    where
      pct_of_image = pixels / (all non-ignore pixels, incl. background)
      pct_of_food  = pixels / (non-background, non-ignore pixels)
    Background and ignore rows are dropped when exclude_background=True.
    """
    mask = np.asarray(mask).astype(np.int64)
    counts = np.bincount(mask[mask != ignore_index].ravel()) if mask.size else np.array([])

    total_image = int((mask != ignore_index).sum())
    food_sel = (mask != ignore_index) & (mask != background_id)
    total_food = int(food_sel.sum())

    rows: List[dict] = []
    for class_id, px in enumerate(counts):
        px = int(px)
        if px == 0:
            continue
        if exclude_background and class_id == background_id:
            continue
        name = id2label.get(class_id, f"class_{class_id}") if id2label else f"class_{class_id}"
        rows.append({
            "class_id": class_id,
            "class_name": name,
            "pixels": px,
            "pct_of_image": 100.0 * px / total_image if total_image else 0.0,
            "pct_of_food": 100.0 * px / total_food if total_food else 0.0,
        })

    df = pd.DataFrame(rows, columns=["class_id", "class_name", "pixels",
                                     "pct_of_image", "pct_of_food"])
    if not df.empty:
        df = df.sort_values("pixels", ascending=False).reset_index(drop=True)
    return df


def top_k(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return df.head(k).reset_index(drop=True)


def format_summary(df: pd.DataFrame, image_name: str = "", k: int = 5) -> str:
    """Human-readable top-k summary string (for console / logs)."""
    lines = [f"Visible ingredient area summary{f' for {image_name}' if image_name else ''}:"]
    if df.empty:
        lines.append("  (no foreground ingredients detected)")
    else:
        for _, r in top_k(df, k).iterrows():
            lines.append(f"  {r['class_name']:<24s} "
                         f"{r['pct_of_food']:6.2f}% of food  "
                         f"({r['pct_of_image']:6.2f}% of image, {int(r['pixels'])} px)")
    lines.append(f"  NOTE: {DISCLAIMER}")
    return "\n".join(lines)


def save_per_image_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a single image's summary, with the disclaimer as a comment header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {DISCLAIMER}\n")
        df.to_csv(f, index=False)


def aggregate_folder(per_image: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-image summaries into a dataset-level mean of pct_of_food.

    Useful for a 'typical composition' table across an image folder.
    """
    frames = []
    for name, df in per_image.items():
        if df.empty:
            continue
        d = df[["class_id", "class_name", "pixels", "pct_of_food"]].copy()
        d["image"] = name
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["class_id", "class_name", "n_images",
                                     "total_pixels", "mean_pct_of_food"])
    big = pd.concat(frames, ignore_index=True)
    agg = (big.groupby(["class_id", "class_name"])
              .agg(n_images=("image", "nunique"),
                   total_pixels=("pixels", "sum"),
                   mean_pct_of_food=("pct_of_food", "mean"))
              .reset_index()
              .sort_values("total_pixels", ascending=False)
              .reset_index(drop=True))
    return agg


def _load_mask(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img)  # 'P'/'L' palette index == class id


def main() -> None:
    ap = argparse.ArgumentParser(description="Visible pixel-area % from a mask PNG.")
    ap.add_argument("--mask", required=True, help="mask PNG with integer class ids")
    ap.add_argument("--background-id", type=int, default=0)
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    mask = _load_mask(args.mask)
    df = compute_area_percentages(mask, background_id=args.background_id,
                                  ignore_index=args.ignore_index)
    print(format_summary(df, Path(args.mask).name, args.topk))
    if args.out_csv:
        save_per_image_csv(df, args.out_csv)
        print(f"Saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
