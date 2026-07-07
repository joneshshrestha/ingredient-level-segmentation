"""Visualization helpers: color palettes, mask colorization, overlays,
legends, and side-by-side paper figures.

Distinction that matters for this project:
  * integer class-id masks are colorized with a fixed lookup table (no
    interpolation, no blending of ids);
  * RGB *images* may be alpha-blended with the colorized mask for overlays.

Importable from scripts and the notebook. Also has a small CLI to rebuild
figures from an image + saved mask PNG(s).
"""
from __future__ import annotations

import argparse
import colorsys
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# NOTE: we do NOT force a matplotlib backend here. matplotlib auto-selects a
# non-interactive backend (Agg) when there is no display, and the shell scripts
# export MPLBACKEND=Agg for headless runs. Leaving it unset lets Jupyter's
# inline backend work normally in the notebook.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GOLDEN_RATIO_CONJUGATE = 0.61803398875


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
def build_palette(num_labels: int, background_id: int = 0,
                  ignore_index: int = 255) -> np.ndarray:
    """Return a deterministic (256, 3) uint8 lookup table.

    - background_id   -> black
    - ignore_index    -> white
    - every other id  -> a distinct, evenly spread HSV hue
    Deterministic: a given (num_labels, background_id) always yields the same
    colors, so figures are reproducible across runs.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(min(num_labels, 256)):
        if c == background_id:
            lut[c] = (0, 0, 0)
            continue
        hue = (c * GOLDEN_RATIO_CONJUGATE) % 1.0
        # vary saturation/value slightly with index for extra separability
        sat = 0.55 + 0.30 * ((c * 7) % 5) / 4.0
        val = 0.75 + 0.20 * ((c * 3) % 4) / 3.0
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        lut[c] = (int(r * 255), int(g * 255), int(b * 255))
    if 0 <= ignore_index < 256:
        lut[ignore_index] = (255, 255, 255)
    return lut


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map an integer class-id mask (H, W) to an RGB image (H, W, 3) uint8."""
    mask = np.asarray(mask)
    safe = np.clip(mask.astype(np.int64), 0, 255)
    return palette[safe]


def mask_to_pil(mask: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Return a compact palette-mode ('P') PIL image with `palette` embedded.

    The raw pixel values stay as the integer class ids, so the PNG is both
    human-viewable and machine-readable (re-loadable as ids).
    """
    pil = Image.fromarray(np.asarray(mask).astype(np.uint8), mode="P")
    pil.putpalette(palette.reshape(-1).tolist())
    return pil


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def overlay_mask(image: np.ndarray, mask: np.ndarray, palette: np.ndarray,
                 alpha: float = 0.5, background_id: Optional[int] = 0,
                 ignore_index: int = 255) -> np.ndarray:
    """Alpha-blend a colorized mask over an RGB image.

    Background and ignore pixels are left untinted so the underlying food /
    context stays visible.
    """
    image = np.asarray(image).astype(np.float32)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    color = colorize_mask(mask, palette).astype(np.float32)

    tint = np.ones(mask.shape, dtype=bool)
    if background_id is not None:
        tint &= mask != background_id
    tint &= mask != ignore_index

    out = image.copy()
    blended = (1.0 - alpha) * image + alpha * color
    out[tint] = blended[tint]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
def present_classes(mask: np.ndarray, ignore_index: int = 255) -> List[int]:
    vals = np.unique(np.asarray(mask))
    return [int(v) for v in vals if v != ignore_index]


def legend_handles(class_ids: Sequence[int], palette: np.ndarray,
                   id2label: Optional[Dict[int, str]] = None,
                   max_items: int = 25) -> List[Patch]:
    handles: List[Patch] = []
    for c in list(class_ids)[:max_items]:
        name = id2label.get(c, f"class_{c}") if id2label else f"class_{c}"
        color = palette[c] / 255.0
        handles.append(Patch(facecolor=color, edgecolor="black", label=f"{c}: {name}"))
    return handles


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def side_by_side(image: np.ndarray,
                 gt: Optional[np.ndarray] = None,
                 pred: Optional[np.ndarray] = None,
                 palette: Optional[np.ndarray] = None,
                 id2label: Optional[Dict[int, str]] = None,
                 alpha: float = 0.5,
                 ignore_index: int = 255,
                 background_id: int = 0,
                 title: Optional[str] = None,
                 max_legend_items: int = 25):
    """Build a (image | [GT] | [pred] | overlay) figure with a legend.

    Returns the matplotlib Figure (caller decides to save or show).
    """
    if palette is None:
        n = 1 + int(max([m.max() for m in [gt, pred] if m is not None], default=0))
        palette = build_palette(n, background_id, ignore_index)

    panels: List[Tuple[str, np.ndarray]] = [("Image", np.asarray(image))]
    if gt is not None:
        panels.append(("Ground truth", colorize_mask(gt, palette)))
    if pred is not None:
        panels.append(("Prediction", colorize_mask(pred, palette)))
    overlay_src = pred if pred is not None else gt
    if overlay_src is not None:
        panels.append(("Overlay", overlay_mask(image, overlay_src, palette, alpha,
                                                background_id, ignore_index)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    # Legend from whatever mask(s) we have.
    ids = set()
    for m in (gt, pred):
        if m is not None:
            ids.update(present_classes(m, ignore_index))
    handles = legend_handles(sorted(ids), palette, id2label, max_legend_items)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def save_figure(fig, path: str | os.PathLike, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI: rebuild a figure from an image + saved mask PNG(s)
# ---------------------------------------------------------------------------
def _load_mask(path: str) -> np.ndarray:
    """Load a mask PNG as raw integer class ids (handles 'P' and 'L' modes)."""
    img = Image.open(path)
    if img.mode == "P":
        img = img  # palette index values ARE the class ids
    return np.array(img.convert("L") if img.mode not in ("P", "L", "I") else img)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild a side-by-side figure.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", default=None, help="ground-truth mask PNG (class ids)")
    ap.add_argument("--pred", default=None, help="predicted mask PNG (class ids)")
    ap.add_argument("--num-labels", type=int, default=104)
    ap.add_argument("--background-id", type=int, default=0)
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    gt = _load_mask(args.gt) if args.gt else None
    pred = _load_mask(args.pred) if args.pred else None
    palette = build_palette(args.num_labels, args.background_id, args.ignore_index)
    fig = side_by_side(image, gt, pred, palette, alpha=args.alpha,
                       ignore_index=args.ignore_index, background_id=args.background_id)
    save_figure(fig, args.out, dpi=args.dpi)
    print(f"Saved figure -> {args.out}")


if __name__ == "__main__":
    main()
