"""Run a fine-tuned SegFormer-B0 checkpoint on a single image or a folder.

For each image it writes:
  * <stem>_mask.png     -- palette PNG whose pixel values ARE the class ids
  * <stem>_overlay.png  -- ingredient mask blended over the original image
  * <stem>_area.csv      -- visible pixel-area % per ingredient (see disclaimer)
and an aggregate area summary across all inputs.

Predictions are restored to each image's ORIGINAL size before saving.

Example:
    python -m src.predict --config configs/segformer_b0.yaml \
        --checkpoint <output_dir>/best --input some_images/ --output-dir results/predictions
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .utils import apply_hf_home, ensure_dir, get_logger, load_yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict ingredient masks with SegFormer-B0.")
    ap.add_argument("--config", default="configs/segformer_b0.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or a folder of images")
    ap.add_argument("--output-dir", default="results/predictions")
    ap.add_argument("--topk", type=int, default=None, help="top-k ingredients to print")
    ap.add_argument("--alpha", type=float, default=None, help="overlay transparency")
    return ap.parse_args()


def gather_images(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = [q for q in sorted(p.iterdir()) if q.suffix in IMAGE_EXTS]
        return files
    raise FileNotFoundError(f"--input not found: {path}")


def main() -> None:
    args = parse_args()
    log = get_logger("foodseg.predict")
    cfg = load_yaml(args.config)
    apply_hf_home(cfg)

    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image

    from .area_summary import (DISCLAIMER, aggregate_folder,
                               compute_area_percentages, format_summary,
                               save_per_image_csv)
    from .model import id2label_from_config, load_model_from_checkpoint
    from .transforms import IMAGENET_MEAN, IMAGENET_STD
    from .utils import get_device, save_csv
    from .visualize import build_palette, mask_to_pil, overlay_mask

    device = get_device()
    out_dir = ensure_dir(args.output_dir)
    vis_cfg = cfg.get("visualization", {})
    alpha = args.alpha if args.alpha is not None else vis_cfg.get("alpha", 0.5)
    topk = args.topk if args.topk is not None else vis_cfg.get("topk", 5)
    image_size = cfg.get("image_size", 512)
    ignore_index = cfg.get("ignore_index", 255)
    background_id = cfg.get("background_id", 0)

    model = load_model_from_checkpoint(args.checkpoint, ignore_index).to(device).eval()
    num_labels = model.config.num_labels
    id2label = id2label_from_config(model)
    palette = build_palette(num_labels, background_id, ignore_index)

    images = gather_images(args.input)
    if not images:
        log.error("No images found at %s", args.input)
        return
    log.info("Predicting on %d image(s) -> %s", len(images), out_dir)
    log.info("NOTE: %s", DISCLAIMER)

    per_image = {}
    with torch.no_grad():
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            resized = image.resize((image_size, image_size), Image.BILINEAR)
            pv = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
            pv = pv.unsqueeze(0).to(device)

            logits = model(pixel_values=pv).logits
            up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear",
                               align_corners=False)
            mask = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

            stem = img_path.stem
            mask_to_pil(mask, palette).save(out_dir / f"{stem}_mask.png")
            overlay = overlay_mask(np.array(image), mask, palette, alpha,
                                   background_id, ignore_index)
            Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")

            df = compute_area_percentages(mask, id2label, background_id, ignore_index)
            save_per_image_csv(df, out_dir / f"{stem}_area.csv")
            per_image[stem] = df
            log.info("%s", format_summary(df, img_path.name, topk))

    # Dataset-level "typical visible composition" aggregate.
    agg = aggregate_folder(per_image)
    if not agg.empty:
        save_csv(agg, out_dir / "area_summary_aggregate.csv")
        log.info("Aggregate visible-area summary -> %s",
                 out_dir / "area_summary_aggregate.csv")
    log.info("Done. Masks/overlays/area CSVs in %s", out_dir)


if __name__ == "__main__":
    main()
