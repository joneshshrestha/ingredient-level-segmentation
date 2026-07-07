"""Evaluate a fine-tuned SegFormer-B0 checkpoint on FoodSeg103.

Metrics are computed at the ORIGINAL mask resolution: each image is run at the
training input size, its logits are upsampled (bilinear, on the continuous
logits) to the original size, then argmax'd and compared to the full-resolution
ground-truth mask. Outputs: metrics.json + per_class_iou.csv. Nothing is
assumed or fabricated -- numbers come only from this run.

Example:
    python -m src.evaluate --config configs/segformer_b0.yaml \
        --checkpoint <output_dir>/best --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .utils import apply_hf_home, apply_overrides, ensure_dir, get_logger, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a SegFormer-B0 checkpoint.")
    ap.add_argument("--config", default="configs/segformer_b0.yaml")
    ap.add_argument("--checkpoint", required=True, help="path to a saved checkpoint dir (e.g. .../best)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--output-dir", default=None, help="where to write metrics (default: <checkpoint>/eval_<split>)")
    ap.add_argument("--num-vis", type=int, default=0,
                    help="also save N qualitative GT-vs-pred figures for the paper")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("foodseg.eval")
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, dataset_root=args.dataset_root)
    apply_hf_home(cfg)

    import torch
    import torch.nn.functional as F
    from .data import apply_label_reduction, load_mask, make_datasets
    from .metrics import ConfusionMatrixMeter
    from .model import id2label_from_config, load_model_from_checkpoint
    from .utils import get_device

    device = get_device()
    out_dir = ensure_dir(args.output_dir or (Path(args.checkpoint) / f"eval_{args.split}"))
    ignore_index = cfg.get("ignore_index", 255)
    background_id = cfg.get("background_id", 0)
    do_reduce = bool(cfg.get("do_reduce_labels", False))

    model = load_model_from_checkpoint(args.checkpoint, ignore_index).to(device).eval()
    num_labels = model.config.num_labels
    id2label = id2label_from_config(model)

    bundle = make_datasets(cfg, scan_sample=300)
    dataset = bundle["datasets"][args.split]
    if bundle["meta"]["num_labels"] != num_labels:
        log.warning("Dataset-derived num_labels=%d != checkpoint num_labels=%d. "
                    "Using checkpoint's label space.",
                    bundle["meta"]["num_labels"], num_labels)
    log.info("Evaluating split='%s' (%d images) at ORIGINAL resolution.",
             args.split, len(dataset))

    from torch.utils.data import DataLoader
    from .data import collate_fn
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], collate_fn=collate_fn)

    meter = ConfusionMatrixMeter(num_labels, ignore_index, background_id)
    vis_saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval[{args.split}]"):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            logits = model(pixel_values=pv).logits  # (B, C, H/4, W/4)
            for i in range(pv.size(0)):
                gt = load_mask(batch["mask_paths"][i])  # full-res integer ids
                if do_reduce:
                    gt = apply_label_reduction(gt, background_id, ignore_index)
                h, w = gt.shape[:2]
                up = F.interpolate(logits[i:i + 1], size=(h, w), mode="bilinear",
                                   align_corners=False)
                pred = up.argmax(dim=1)[0].cpu().numpy()
                meter.update(pred, gt)

                if vis_saved < args.num_vis:
                    _save_vis(batch["image_paths"][i], gt, pred, num_labels, cfg,
                              id2label, out_dir, vis_saved)
                    vis_saved += 1

    metrics = meter.compute()
    metrics_out = {
        "checkpoint": str(args.checkpoint), "split": args.split,
        "num_images": len(dataset), "num_labels": num_labels,
        "eval_resolution": "original",
        "metric_definitions": {
            "mean_iou": "mean of per-class TP/(TP+FP+FN) over classes present in GT or pred",
            "pixel_accuracy": "correct pixels / evaluated pixels (ignore_index excluded)",
            "note": "Same-split/metric/preprocessing as published FoodSeg103 is NOT "
                    "guaranteed; treat published numbers as rough references only.",
        },
        **metrics,
    }
    save_json(metrics_out, out_dir / "metrics.json")
    meter.export_per_class_csv(out_dir / "per_class_iou.csv", id2label)

    log.info("=" * 60)
    log.info("RESULTS (%s split, %d images)", args.split, len(dataset))
    log.info("  pixel_accuracy        : %.4f", metrics["pixel_accuracy"])
    log.info("  mean_iou (all classes): %.4f", metrics["mean_iou"])
    log.info("  mean_iou (no bg)      : %.4f", metrics["mean_iou_no_background"])
    log.info("  mean_class_accuracy   : %.4f", metrics["mean_class_accuracy"])
    log.info("  classes present       : %d / %d",
             metrics["num_classes_present"], metrics["num_classes_total"])
    log.info("  saved -> %s", out_dir / "metrics.json")
    log.info("          %s", out_dir / "per_class_iou.csv")
    log.info("=" * 60)


def _save_vis(image_path, gt, pred, num_labels, cfg, id2label, out_dir, idx):
    import numpy as np
    from PIL import Image
    from .visualize import build_palette, save_figure, side_by_side
    palette = build_palette(num_labels, cfg.get("background_id", 0),
                            cfg.get("ignore_index", 255))
    image = np.array(Image.open(image_path).convert("RGB"))
    fig = side_by_side(image, gt, pred, palette, id2label,
                       alpha=cfg.get("visualization", {}).get("alpha", 0.5),
                       ignore_index=cfg.get("ignore_index", 255),
                       background_id=cfg.get("background_id", 0),
                       title=Path(image_path).name)
    save_figure(fig, Path(out_dir) / "figures" / f"eval_{idx:03d}_{Path(image_path).stem}.png",
                dpi=cfg.get("visualization", {}).get("dpi", 150))


if __name__ == "__main__":
    main()
