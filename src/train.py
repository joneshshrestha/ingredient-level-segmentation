"""Fine-tune SegFormer-B0 on FoodSeg103.

Examples
--------
Smoke test (tiny subset, verifies the whole pipeline fast):
    python -m src.train --config configs/segformer_b0.yaml --smoke-test

Full training:
    python -m src.train --config configs/segformer_b0.yaml

Resume from a previous run's output_dir:
    python -m src.train --config configs/segformer_b0.yaml --resume <output_dir>

Checkpoints: <output_dir>/best (highest val mIoU) and <output_dir>/last.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .utils import (apply_hf_home, apply_overrides, ensure_dir, get_logger,
                    load_yaml, save_json, save_yaml, seed_everything)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fine-tune SegFormer-B0 on FoodSeg103.")
    ap.add_argument("--config", default="configs/segformer_b0.yaml")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--smoke-test", action="store_true",
                    help="run a tiny end-to-end check (overrides epochs/batch/subset)")
    ap.add_argument("--resume", default=None,
                    help="output_dir of a previous run to resume from (loads last/)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    return ap.parse_args()


def build_dataloaders(cfg, smoke):
    import torch
    from torch.utils.data import DataLoader, Subset
    from .data import collate_fn, make_datasets

    scan_sample = 50 if smoke else 300
    bundle = make_datasets(cfg, scan_sample=scan_sample,
                           allow_sample_without_labelmap=smoke)
    ds, meta = bundle["datasets"], bundle["meta"]

    if smoke:
        st = cfg.get("smoke_test", {})
        n_tr = min(st.get("num_train", 8), len(ds["train"]))
        n_va = min(st.get("num_val", 8), len(ds["val"]))
        ds["train"] = Subset(ds["train"], list(range(n_tr)))
        ds["val"] = Subset(ds["val"], list(range(n_va)))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ds["train"], batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=pin,
                              collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(ds["val"], batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=pin,
                            collate_fn=collate_fn)
    return train_loader, val_loader, meta


def make_scheduler(cfg, optimizer, total_steps):
    import torch
    if cfg.get("lr_scheduler", "poly") != "poly" or total_steps <= 0:
        return None
    power = float(cfg.get("poly_power", 1.0))
    warmup = int(cfg.get("warmup_steps", 0))

    def lr_lambda(step):
        if warmup and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, (1.0 - progress)) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def amp_settings(cfg, device):
    """Resolve (enabled, dtype, use_scaler) for autocast."""
    import torch
    mode = str(cfg.get("mixed_precision", "no")).lower()
    if device.type != "cuda" or mode in ("no", "none", "false"):
        return False, None, False
    if mode == "bf16":
        return True, torch.bfloat16, False     # bf16 needs no GradScaler
    return True, torch.float16, True           # fp16 needs a GradScaler


def validate(model, loader, device, num_labels, cfg):
    import torch
    import torch.nn.functional as F
    from .metrics import ConfusionMatrixMeter

    model.eval()
    meter = ConfusionMatrixMeter(num_labels, cfg.get("ignore_index", 255),
                                 cfg.get("background_id", 0))
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = model(pixel_values=pv, labels=labels)
            total_loss += out.loss.item() * pv.size(0)
            n += pv.size(0)
            # logits are H/4 x W/4 -> upsample (bilinear on continuous logits) to
            # label size, then argmax to get integer class predictions.
            up = F.interpolate(out.logits, size=labels.shape[-2:], mode="bilinear",
                               align_corners=False)
            pred = up.argmax(dim=1)
            meter.update(pred, labels)
    metrics = meter.compute()
    metrics["val_loss"] = total_loss / max(1, n)
    return metrics, meter


def main() -> None:
    args = parse_args()
    log = get_logger("foodseg.train")

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, dataset_root=args.dataset_root,
                          output_dir=args.output_dir, epochs=args.epochs,
                          batch_size=args.batch_size)

    smoke = args.smoke_test or cfg.get("smoke_test", {}).get("enabled", False)
    if smoke:
        st = cfg.get("smoke_test", {})
        cfg["epochs"] = st.get("epochs", 1)
        cfg["batch_size"] = st.get("batch_size", 2)
        log.info("SMOKE TEST: epochs=%d batch=%d (tiny subset)",
                 cfg["epochs"], cfg["batch_size"])

    apply_hf_home(cfg)  # set HF cache before importing transformers-heavy code
    import torch
    from .model import build_model, load_model_from_checkpoint
    from .utils import get_device

    seed_everything(cfg.get("seed", 42))
    device = get_device()
    output_dir = ensure_dir(cfg["output_dir"])
    log.info("Device: %s | output_dir: %s", device, output_dir)

    train_loader, val_loader, meta = build_dataloaders(cfg, smoke)
    num_labels = meta["num_labels"]
    log.info("Labels: num_labels=%d | train=%d val=%d (test held separate=%d)",
             num_labels, len(train_loader.dataset), len(val_loader.dataset),
             meta.get("n_test", -1))
    for note in meta["notes"]:
        log.info("  [labels] %s", note)

    # Persist label space + config alongside checkpoints (reproducibility).
    save_json({str(k): v for k, v in meta["id2label"].items()},
              output_dir / "id2label.json")
    save_yaml(copy.deepcopy(cfg), output_dir / "config_used.yaml")

    # Model (resume from last/ if requested, else build from pretrained).
    start_epoch, best_miou = 0, -1.0
    if args.resume:
        last_dir = Path(args.resume) / "last"
        model = load_model_from_checkpoint(str(last_dir), cfg.get("ignore_index", 255))
    else:
        model = build_model(cfg, meta["id2label"], meta["label2id"], num_labels)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler = make_scheduler(cfg, optimizer, total_steps)
    amp_enabled, amp_dtype, use_scaler = amp_settings(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    log.info("AMP: enabled=%s dtype=%s scaler=%s", amp_enabled, amp_dtype, use_scaler)

    if args.resume and (Path(args.resume) / "training_state.pt").exists():
        state = torch.load(Path(args.resume) / "training_state.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state.get("epoch", 0) + 1
        best_miou = state.get("best_miou", -1.0)
        log.info("Resumed at epoch %d (best_miou=%.4f)", start_epoch, best_miou)

    history_path = output_dir / "training_history.csv"
    history = pd.read_csv(history_path).to_dict("records") \
        if (args.resume and history_path.exists()) else []
    max_norm = cfg.get("max_grad_norm") or 0

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['epochs']}", leave=False)
        for batch in pbar:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                out = model(pixel_values=pv, labels=labels)
                loss = out.loss
            if use_scaler:
                scaler.scale(loss).backward()
                if max_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            if scheduler:
                scheduler.step()
            loss_val = loss.item()  # .item() detaches; avoids grad-tensor->scalar warning
            running += loss_val * pv.size(0)
            seen += pv.size(0)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        train_loss = running / max(1, seen)
        metrics, _ = validate(model, val_loader, device, num_labels, cfg)
        lr_now = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0
        log.info("epoch %d | train_loss=%.4f val_loss=%.4f val_mIoU=%.4f "
                 "val_mIoU_nobg=%.4f val_pixAcc=%.4f lr=%.2e (%.1fs)",
                 epoch + 1, train_loss, metrics["val_loss"], metrics["mean_iou"],
                 metrics["mean_iou_no_background"], metrics["pixel_accuracy"],
                 lr_now, secs)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": metrics["val_loss"], "val_mIoU": metrics["mean_iou"],
            "val_mIoU_no_bg": metrics["mean_iou_no_background"],
            "val_pixel_accuracy": metrics["pixel_accuracy"],
            "lr": lr_now, "seconds": secs,
        })
        pd.DataFrame(history).to_csv(history_path, index=False)

        # Always save 'last'; save 'best' on val-mIoU improvement.
        model.save_pretrained(output_dir / "last")
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch, "best_miou": best_miou},
                   output_dir / "training_state.pt")
        if metrics["mean_iou"] > best_miou:
            best_miou = metrics["mean_iou"]
            model.save_pretrained(output_dir / "best")
            save_json({**metrics, "epoch": epoch + 1}, output_dir / "best_val_metrics.json")
            log.info("  -> new best val mIoU=%.4f (saved to %s)",
                     best_miou, output_dir / "best")

    log.info("Training done. Best val mIoU=%.4f. Best checkpoint: %s",
             best_miou, output_dir / "best")
    log.info("Next: evaluate on the official TEST split with src.evaluate.")


if __name__ == "__main__":
    main()
