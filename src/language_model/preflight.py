"""Environment checks shared by local smoke runs and Kaggle training."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


class PreflightError(RuntimeError):
    """Raised before expensive work when the requested environment is unavailable."""


def cuda_capability_supported(capability: tuple[int, int], architectures: list[str]) -> bool:
    if not architectures:
        return True
    requested = f"sm_{capability[0]}{capability[1]}"
    return requested in architectures


def _validate_cuda_compatibility() -> None:
    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    architectures = torch.cuda.get_arch_list()
    if not cuda_capability_supported(capability, architectures):
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        raise PreflightError(
            f"GPU {name} has capability sm_{capability[0]}{capability[1]}, but this PyTorch build supports "
            f"{', '.join(architectures) or 'an unknown architecture set'}; select a T4 or newer Kaggle GPU"
        )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise PreflightError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "cuda" or requested == "auto" and torch.cuda.is_available():
        _validate_cuda_compatibility()
    if requested == "mps" and not torch.backends.mps.is_available():
        raise PreflightError("MPS was requested but this PyTorch build or machine does not support it")
    return torch.device(requested)


def environment_info(work_dir: str | Path = ".") -> dict[str, Any]:
    path = Path(work_dir).resolve()
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda.update(
            {
                "current_device": index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "supported_architectures": torch.cuda.get_arch_list(),
                "compatible": cuda_capability_supported(
                    (properties.major, properties.minor), torch.cuda.get_arch_list()
                ),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": cuda,
        "disk_free_bytes": usage.free,
        "work_dir": str(path),
    }


def print_preflight(work_dir: str | Path = ".") -> dict[str, Any]:
    info = environment_info(work_dir)
    print(json.dumps({"phase": "preflight", "status": "completed", **info}, ensure_ascii=False), flush=True)
    if info["cuda"]["available"] and not info["cuda"].get("compatible", True):
        _validate_cuda_compatibility()
    return info
