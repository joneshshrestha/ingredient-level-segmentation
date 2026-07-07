"""SegFormer-B0 loading for FoodSeg103 ingredient segmentation.

Default checkpoint: `nvidia/mit-b0` -- the ImageNet-pretrained MiT-B0 backbone
with NO segmentation head. Loading it into SegformerForSemanticSegmentation
creates a FRESH decode head sized to our derived `num_labels`, so there is no
ADE20K label baggage and no label reduction inherited from a 150-class head.
(The "decode_head weights newly initialized" message at load time is expected
and correct -- that head is what we fine-tune.)

Alternative: an ADE20K-finetuned SegFormer-B0 checkpoint can be used by setting
`use_ade20k_checkpoint: true`. Its 150-class head is then replaced with our
num_labels head via `ignore_mismatched_sizes=True`. The MiT backbone weights are
loaded either way; only do_reduce_labels controls label semantics (kept False).

Environment note: transformers>=5 + torch<2.6 refuse to load legacy `.bin`
checkpoints (CVE-2025-32434) and require `.safetensors`. `nvidia/mit-b0`
publishes only `.bin`. With torch>=2.6 (the `foodseg` env) this module
transparently converts it to a local safetensors checkpoint on first use; with
torch<2.6 it raises a clear error rather than calling torch.load on the .bin.
(The ADE20K checkpoint already ships safetensors.) Our own fine-tuned
checkpoints are saved as safetensors.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from .utils import get_logger

log = get_logger("foodseg.model")

ADE20K_B0 = "nvidia/segformer-b0-finetuned-ade-512-512"


def _resolve_loadable_checkpoint(model_name: str, hf_home: Optional[str]) -> str:
    """Return a path/repo that has safetensors weights.

    If `model_name` is a local dir or already has safetensors, it is returned
    unchanged. If it only has `.bin`, it is downloaded and converted to a local
    safetensors checkpoint (cached, so conversion happens once).
    """
    # Local directory already on disk -> trust it.
    if Path(model_name).is_dir():
        return model_name

    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"],
    )
    files = os.listdir(snap)
    if any(f.endswith(".safetensors") for f in files):
        return snap  # loadable as-is

    bin_files = [f for f in files if f.endswith(".bin")]
    if not bin_files:
        return model_name  # nothing to convert; let from_pretrained decide
    if any(f.endswith(".bin.index.json") for f in files):
        raise RuntimeError(
            f"{model_name} uses sharded .bin weights with no safetensors; "
            "please upgrade torch to >=2.6 or pick a safetensors checkpoint.")

    import torch
    from packaging.version import Version

    # torch < 2.6 cannot safely load legacy .bin (CVE-2025-32434). Do NOT call
    # torch.load on it -- fail loudly with guidance instead.
    if Version(torch.__version__.split("+")[0]) < Version("2.6"):
        raise RuntimeError(
            f"'{model_name}' ships only legacy .bin weights, and the active torch "
            f"({torch.__version__}) < 2.6 cannot safely load them (CVE-2025-32434).\n"
            "Fix: use the 'foodseg' env which has torch>=2.6:  conda activate foodseg\n"
            "  (or upgrade torch to >=2.6, or pick a safetensors checkpoint, e.g. set "
            "use_ade20k_checkpoint: true).")

    from safetensors.torch import save_file

    cache_root = Path(hf_home) if hf_home else (Path.home() / ".cache")
    out_dir = cache_root / "foodseg_converted" / model_name.replace("/", "__")
    if (out_dir / "model.safetensors").exists() and (out_dir / "config.json").exists():
        log.info("Using cached safetensors conversion: %s", out_dir)
        return str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    for f in files:  # copy config / aux files alongside the converted weights
        if f == "config.json":
            # Drop the backbone's ImageNet classification metadata so our own
            # num_labels / id2label aren't fought over at load time. We override
            # them in from_pretrained anyway.
            with open(os.path.join(snap, f)) as fh:
                conf = json.load(fh)
            for k in ("id2label", "label2id", "num_labels"):
                conf.pop(k, None)
            with open(out_dir / f, "w") as fh:
                json.dump(conf, fh, indent=2)
        elif f.endswith((".json", ".txt")):
            shutil.copy(os.path.join(snap, f), out_dir / f)

    state = torch.load(os.path.join(snap, bin_files[0]), map_location="cpu",
                       weights_only=True)
    # safetensors needs contiguous, non-shared tensors. Keep ALL tensor entries
    # (including integer buffers), not only floating-point ones.
    tensors = {k: v.detach().clone().contiguous()
               for k, v in state.items() if isinstance(v, torch.Tensor)}
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    log.info("Converted %s (.bin) -> safetensors at %s (%d tensors)",
             model_name, out_dir, len(tensors))
    return str(out_dir)


def build_model(cfg: dict, id2label: Dict[int, str], label2id: Dict[str, int],
                num_labels: int):
    """Construct a SegformerForSemanticSegmentation ready for fine-tuning."""
    from transformers import SegformerForSemanticSegmentation

    model_name = cfg.get("model_name", "nvidia/mit-b0")
    use_ade = bool(cfg.get("use_ade20k_checkpoint", False))
    ignore_index = cfg.get("ignore_index", 255)

    # If the user asked for the ADE20K checkpoint but left the backbone name,
    # switch to the real ADE20K checkpoint so the head actually gets replaced.
    if use_ade and ("mit-b0" in model_name):
        log.info("use_ade20k_checkpoint=True -> switching model_name '%s' -> '%s'",
                 model_name, ADE20K_B0)
        model_name = ADE20K_B0

    # ignore_mismatched_sizes is ONLY needed to replace an existing seg head
    # (the ADE20K case). The bare mit-b0 backbone has no head to mismatch.
    ignore_mismatched = use_ade

    load_target = _resolve_loadable_checkpoint(model_name, cfg.get("hf_home"))
    log.info("Loading '%s' with num_labels=%d (ignore_mismatched_sizes=%s)",
             model_name, num_labels, ignore_mismatched)
    model = SegformerForSemanticSegmentation.from_pretrained(
        load_target,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id={k: int(v) for k, v in label2id.items()},
        ignore_mismatched_sizes=ignore_mismatched,
    )

    # Wire the loss ignore index explicitly (HF default is 255; we make it explicit).
    model.config.semantic_loss_ignore_index = int(ignore_index)

    assert model.config.num_labels == num_labels, (
        f"model num_labels {model.config.num_labels} != expected {num_labels}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model ready: %.2fM params (%.2fM trainable). "
             "semantic_loss_ignore_index=%d, decode head classes=%d",
             n_params / 1e6, n_train / 1e6,
             model.config.semantic_loss_ignore_index, model.config.num_labels)
    return model


def load_model_from_checkpoint(checkpoint_dir: str, ignore_index: int = 255):
    """Load a fine-tuned checkpoint saved with `save_pretrained` (safetensors)."""
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir)
    model.config.semantic_loss_ignore_index = int(ignore_index)
    log.info("Loaded checkpoint '%s' (num_labels=%d)",
             checkpoint_dir, model.config.num_labels)
    return model


def id2label_from_config(model) -> Dict[int, str]:
    """Return {int_id: name} from a (possibly str-keyed) model config."""
    raw = model.config.id2label or {}
    return {int(k): v for k, v in raw.items()}
