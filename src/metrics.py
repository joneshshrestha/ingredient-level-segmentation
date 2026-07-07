"""Segmentation metrics via an accumulating confusion matrix.

Metric definitions used throughout the project (stated explicitly so results
are interpretable and not silently compared to published numbers):
  * pixel_accuracy   = sum(correct pixels) / sum(evaluated pixels)
  * IoU per class    = TP / (TP + FP + FN)
  * mean_iou         = mean of IoU over classes PRESENT in GT or prediction
                       (classes absent from both are excluded, not counted as 0)
  * mean_iou_no_bg   = same, but excluding the background class
Pixels equal to `ignore_index` are excluded from everything.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class ConfusionMatrixMeter:
    def __init__(self, num_labels: int, ignore_index: int = 255,
                 background_id: int = 0):
        self.num_labels = int(num_labels)
        self.ignore_index = ignore_index
        self.background_id = background_id
        self.reset()

    def reset(self) -> None:
        self.cm = np.zeros((self.num_labels, self.num_labels), dtype=np.int64)

    def update(self, pred, gt) -> None:
        """Accumulate one (prediction, ground-truth) pair or batch.

        Rows = ground-truth class, columns = predicted class.
        Accepts numpy arrays or torch tensors of integer class ids.
        """
        pred = _to_numpy(pred).astype(np.int64).ravel()
        gt = _to_numpy(gt).astype(np.int64).ravel()
        if pred.shape != gt.shape:
            raise ValueError(f"pred {pred.shape} and gt {gt.shape} shape mismatch")

        valid = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_labels)
        gt_v = gt[valid]
        pred_v = np.clip(pred[valid], 0, self.num_labels - 1)  # head can't exceed range
        inds = self.num_labels * gt_v + pred_v
        self.cm += np.bincount(inds, minlength=self.num_labels ** 2).reshape(
            self.num_labels, self.num_labels)

    # -- derived quantities --------------------------------------------------
    def per_class_iou(self) -> np.ndarray:
        cm = self.cm
        tp = np.diag(cm).astype(np.float64)
        union = cm.sum(axis=1) + cm.sum(axis=0) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)
        return iou

    def per_class_accuracy(self) -> np.ndarray:
        cm = self.cm
        tp = np.diag(cm).astype(np.float64)
        gt_sum = cm.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(gt_sum > 0, tp / gt_sum, np.nan)
        return acc

    def support(self) -> np.ndarray:
        """Ground-truth pixel count per class."""
        return self.cm.sum(axis=1)

    def compute(self) -> Dict[str, float]:
        iou = self.per_class_iou()
        acc = self.per_class_accuracy()
        total = self.cm.sum()
        pixel_acc = float(np.diag(self.cm).sum() / total) if total > 0 else float("nan")

        mean_iou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan")
        mean_acc = float(np.nanmean(acc)) if np.any(~np.isnan(acc)) else float("nan")

        iou_no_bg = iou.copy()
        if 0 <= self.background_id < self.num_labels:
            iou_no_bg[self.background_id] = np.nan
        mean_iou_no_bg = (float(np.nanmean(iou_no_bg))
                          if np.any(~np.isnan(iou_no_bg)) else float("nan"))

        n_present = int(np.sum(~np.isnan(iou)))
        return {
            "pixel_accuracy": pixel_acc,
            "mean_iou": mean_iou,
            "mean_iou_no_background": mean_iou_no_bg,
            "mean_class_accuracy": mean_acc,
            "num_classes_present": n_present,
            "num_classes_total": self.num_labels,
        }

    def to_dataframe(self, id2label: Optional[Dict[int, str]] = None) -> pd.DataFrame:
        iou = self.per_class_iou()
        acc = self.per_class_accuracy()
        support = self.support()
        rows = []
        for c in range(self.num_labels):
            name = id2label.get(c, f"class_{c}") if id2label else f"class_{c}"
            rows.append({
                "class_id": c,
                "class_name": name,
                "iou": iou[c],
                "accuracy": acc[c],
                "gt_pixels": int(support[c]),
                "present": bool(support[c] > 0 or self.cm[:, c].sum() > 0),
            })
        return pd.DataFrame(rows)

    def export_per_class_csv(self, path, id2label: Optional[Dict[int, str]] = None) -> None:
        from .utils import save_csv
        save_csv(self.to_dataframe(id2label), path)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)
