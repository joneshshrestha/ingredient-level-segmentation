"""Joint image + mask transforms for semantic segmentation.

Interpolation discipline (a hard requirement of this project):
  * RGB images       -> BILINEAR resize, then ImageNet normalization.
  * integer masks    -> NEAREST resize ONLY, values stay as class ids.
  * padding          -> images padded with 0, masks padded with `ignore_index`
                        (so padded regions never count toward loss / metrics).

Normalization uses ImageNet mean/std, which are the defaults of Hugging Face's
SegformerImageProcessor, so preprocessing matches the pretrained model.
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter

# SegformerImageProcessor defaults (ImageNet statistics).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationTransform:
    """Callable applied to a (PIL RGB image, int mask) pair.

    Returns (pixel_values: FloatTensor[3,H,W], labels: LongTensor[H,W]).
    """

    def __init__(self, image_size: int = 512, train: bool = True,
                 ignore_index: int = 255,
                 scale_min: float = 0.5, scale_max: float = 2.0,
                 hflip_prob: float = 0.5,
                 mean: Tuple[float, float, float] = IMAGENET_MEAN,
                 std: Tuple[float, float, float] = IMAGENET_STD):
        self.image_size = int(image_size)
        self.train = train
        self.ignore_index = int(ignore_index)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std
        # Mild photometric jitter on the image only (never the mask).
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) \
            if train else None

    # -- helpers -------------------------------------------------------------
    def _to_mask_pil(self, mask) -> Image.Image:
        if isinstance(mask, Image.Image):
            return mask.convert("L")
        arr = np.asarray(mask)
        if arr.max() > 255:
            raise ValueError("Mask has values > 255; cannot store as 8-bit PIL.")
        return Image.fromarray(arr.astype(np.uint8), mode="L")

    def _resize_short(self, image: Image.Image, mask: Image.Image,
                      short: int) -> Tuple[Image.Image, Image.Image]:
        w, h = image.size
        scale = short / min(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        image = image.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)  # NEAREST: ids preserved
        return image, mask

    def _pad_to(self, image: Image.Image, mask: Image.Image,
                size: int) -> Tuple[Image.Image, Image.Image]:
        w, h = image.size
        pad_w, pad_h = max(0, size - w), max(0, size - h)
        if pad_w or pad_h:
            # pad right/bottom; image fill=0, mask fill=ignore_index
            new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
            new_img.paste(image, (0, 0))
            new_msk = Image.new("L", (w + pad_w, h + pad_h), self.ignore_index)
            new_msk.paste(mask, (0, 0))
            image, mask = new_img, new_msk
        return image, mask

    def _random_crop(self, image: Image.Image, mask: Image.Image,
                     size: int) -> Tuple[Image.Image, Image.Image]:
        w, h = image.size
        left = random.randint(0, w - size)
        top = random.randint(0, h - size)
        box = (left, top, left + size, top + size)
        return image.crop(box), mask.crop(box)

    # -- main ----------------------------------------------------------------
    def __call__(self, image: Image.Image, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        mask = self._to_mask_pil(mask)
        size = self.image_size

        if self.train:
            short = max(1, int(round(size * random.uniform(self.scale_min, self.scale_max))))
            image, mask = self._resize_short(image, mask, short)
            image, mask = self._pad_to(image, mask, size)
            image, mask = self._random_crop(image, mask, size)
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if self.color_jitter is not None:
                image = self.color_jitter(image)
        else:
            # Deterministic: rescale the whole image to a square (no content lost).
            image = image.resize((size, size), Image.BILINEAR)
            mask = mask.resize((size, size), Image.NEAREST)

        pixel_values = TF.to_tensor(image)
        pixel_values = TF.normalize(pixel_values, self.mean, self.std)
        labels = torch.from_numpy(np.array(mask, dtype=np.int64))
        return pixel_values, labels


def build_transforms(cfg: dict, train: bool) -> SegmentationTransform:
    return SegmentationTransform(
        image_size=cfg.get("image_size", 512),
        train=train,
        ignore_index=cfg.get("ignore_index", 255),
    )


def denormalize(pixel_values: torch.Tensor,
                mean: Tuple[float, float, float] = IMAGENET_MEAN,
                std: Tuple[float, float, float] = IMAGENET_STD) -> np.ndarray:
    """Invert normalization for visualization. Returns uint8 (H, W, 3)."""
    t = pixel_values.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr
