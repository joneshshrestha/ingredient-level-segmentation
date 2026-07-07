"""Small shared helpers: seeding, device, config IO, logging, file IO.

Kept dependency-light on purpose so every other module can import from here
without pulling in torch unless needed.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # benchmark gives a speed-up when input sizes are fixed (they are here)
            torch.backends.cudnn.benchmark = True
    except ImportError:  # torch not needed for pure-data tooling
        pass


def get_device(prefer_cuda: bool = True) -> "object":
    """Return a torch.device, falling back to CPU when CUDA is unavailable."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config IO
# ---------------------------------------------------------------------------
def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def apply_overrides(cfg: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Override top-level config keys with any non-None CLI values."""
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def apply_hf_home(cfg: Dict[str, Any]) -> None:
    """Point the Hugging Face cache at cfg['hf_home'] if provided.

    Must be called before importing/instantiating transformers models so the
    download cache lands on the configured (large) disk.
    """
    hf_home = cfg.get("hf_home")
    if hf_home:
        ensure_dir(hf_home)
        # Force-set so the configured project cache is actually used even if
        # HF_HOME was already exported elsewhere in the shell.
        os.environ["HF_HOME"] = str(hf_home)


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_csv(df, path: str | os.PathLike) -> None:
    """Save a pandas DataFrame to CSV (kept here so callers need one import)."""
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def _json_default(o: Any):
    """Make numpy scalars / arrays JSON-serialisable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "foodseg", logfile: Optional[str | os.PathLike] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")

    # Avoid duplicate handlers if get_logger is called more than once.
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if logfile is not None:
        ensure_dir(Path(logfile).parent)
        if not any(isinstance(h, logging.FileHandler) and
                   getattr(h, "baseFilename", None) == str(Path(logfile).resolve())
                   for h in logger.handlers):
            fh = logging.FileHandler(logfile)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
