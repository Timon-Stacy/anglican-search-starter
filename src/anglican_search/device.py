"""Compute-device selection: Nvidia CUDA, Intel Arc (XPU), or CPU.

The embedder and reranker run on a GPU when one is present and fall back to CPU.
Two accelerator backends are supported, picked at runtime:

  * ``cuda`` — Nvidia (the cu128 torch build; default on the original embed box).
  * ``xpu``  — Intel Arc / Data Center GPUs (an XPU-enabled torch build).

FAISS always stays on CPU (``faiss-cpu``); only the transformer models move to the
accelerator, so the index/search path is identical on every backend.

Selection order is ``cuda -> xpu -> cpu`` by what's actually available. Set
``ANGLICAN_DEVICE=cuda|xpu|cpu`` to force a choice (an unavailable forced GPU
falls back to the next best device rather than crashing).

The probe functions import torch lazily and guard everything, so this module is
safe to import even where torch isn't installed; the pure selection logic
(``_choose``) takes the availability booleans as arguments and is unit-tested
without torch.
"""

from __future__ import annotations

import os

VALID_DEVICES = ("cuda", "xpu", "cpu")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - any import/runtime issue => not available
        return False


def _xpu_available() -> bool:
    # torch.xpu exists only in XPU-enabled builds; guard the attribute and call.
    try:
        import torch

        xpu = getattr(torch, "xpu", None)
        return bool(xpu is not None and xpu.is_available())
    except Exception:  # noqa: BLE001
        return False


def _choose(preferred: str, cuda: bool, xpu: bool) -> str:
    """Pure device-selection logic.

    ``preferred`` is "", "cuda", "xpu", or "cpu". A forced GPU that isn't
    available falls through to autodetection (cuda first, then xpu, then cpu).
    """
    want = (preferred or "").strip().lower()
    if want == "cpu":
        return "cpu"
    if want == "cuda" and cuda:
        return "cuda"
    if want == "xpu" and xpu:
        return "xpu"
    if cuda:
        return "cuda"
    if xpu:
        return "xpu"
    return "cpu"


def select_device(preferred: str | None = None) -> str:
    """Return the device string to load models onto: "cuda", "xpu", or "cpu"."""
    pref = preferred if preferred is not None else os.environ.get("ANGLICAN_DEVICE", "")
    return _choose(pref, _cuda_available(), _xpu_available())


def supports_fp16(device: str) -> bool:
    """fp16 inference is a throughput/VRAM win on both GPU backends; skip on CPU
    (fp16 on CPU is slow and can be numerically worse). Escape hatch: set
    ANGLICAN_FP16=0 to force fp32 everywhere (e.g. to rule fp16 out when debugging
    an XPU kernel error)."""
    if os.environ.get("ANGLICAN_FP16", "1") == "0":
        return False
    return device in ("cuda", "xpu")


def model_load_kwargs(device: str) -> dict:
    """Extra `from_pretrained` kwargs per backend (passed through SentenceTransformer
    / CrossEncoder as model_kwargs).

    Intel XPU's fused scaled-dot-product-attention kernel raises a oneAPI
    "UR error" on the Qwen3 / XLM-RoBERTa forward pass, so force the eager
    attention path there. CUDA keeps SDPA (FlashAttention) for speed. Override
    with ANGLICAN_ATTN (e.g. "sdpa", "eager", "flash_attention_2").
    """
    attn = os.environ.get("ANGLICAN_ATTN", "").strip()
    if not attn:
        attn = "eager" if device == "xpu" else ""
    return {"attn_implementation": attn} if attn else {}


def synchronize(device: str) -> None:
    """Block until queued accelerator work finishes. No-op on CPU."""
    try:
        import torch

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "xpu" and getattr(torch, "xpu", None) is not None:
            torch.xpu.synchronize()
    except Exception:  # noqa: BLE001
        pass
