"""Shared helpers for Hugging Face Trainer / SFTTrainer."""
from __future__ import annotations

import inspect


def make_training_args(
    training_args_cls: type,
    *,
    use_cpu: bool,
    use_mps: bool,
    **kwargs,
):
    supported = inspect.signature(training_args_cls.__init__).parameters
    if use_cpu:
        if "use_cpu" in supported:
            kwargs["use_cpu"] = True
        elif "no_cuda" in supported:
            kwargs["no_cuda"] = True
    if use_mps and "use_mps_device" in supported:
        kwargs["use_mps_device"] = True
    return training_args_cls(**kwargs)


def resolve_device_flags(*, cpu: bool = False) -> tuple[bool, bool, bool]:
    """Return (use_cuda, use_mps, use_cpu)."""
    import torch

    use_cuda = torch.cuda.is_available() and not cpu
    use_mps = (
        not use_cuda
        and not cpu
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    return use_cuda, use_mps, cpu
