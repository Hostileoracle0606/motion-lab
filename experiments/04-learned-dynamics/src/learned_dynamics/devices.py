"""Explicit CPU/MPS selection; unsupported MPS never becomes hidden CPU work."""

import os

import torch


def require_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name != "mps":
        raise ValueError("device must be explicitly cpu or mps")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
        raise ValueError("MPS CPU fallback must be disabled before importing PyTorch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in the current PyTorch/runtime")
    return torch.device("mps")
