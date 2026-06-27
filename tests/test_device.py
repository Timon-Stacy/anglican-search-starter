"""Gate tests for accelerator selection (Nvidia CUDA / Intel Arc XPU / CPU).

Deterministic and torch-free: they exercise the pure selection logic `_choose`
and `supports_fp16`, so they run in milliseconds with no GPU and no torch build.
"""

from __future__ import annotations

import pytest

from anglican_search.device import _choose, supports_fp16


# --- autodetection (no forced device) -------------------------------------
def test_autodetect_prefers_cuda_when_both_present():
    assert _choose("", cuda=True, xpu=True) == "cuda"


def test_autodetect_uses_xpu_when_no_cuda():
    assert _choose("", cuda=False, xpu=True) == "xpu"


def test_autodetect_falls_back_to_cpu_when_no_gpu():
    assert _choose("", cuda=False, xpu=False) == "cpu"


# --- forced device (ANGLICAN_DEVICE) --------------------------------------
def test_force_xpu_over_available_cuda():
    # An Arc box (or a deliberate override) must honour the request.
    assert _choose("xpu", cuda=True, xpu=True) == "xpu"


def test_force_cuda_over_available_xpu():
    assert _choose("cuda", cuda=True, xpu=True) == "cuda"


def test_force_cpu_even_with_gpu_present():
    assert _choose("cpu", cuda=True, xpu=True) == "cpu"


@pytest.mark.parametrize("pref", ["XPU", " xpu ", "Xpu"])
def test_force_is_case_and_space_insensitive(pref):
    assert _choose(pref, cuda=False, xpu=True) == "xpu"


# --- forced-but-unavailable falls back gracefully -------------------------
def test_force_cuda_unavailable_falls_back_to_xpu():
    assert _choose("cuda", cuda=False, xpu=True) == "xpu"


def test_force_xpu_unavailable_falls_back_to_cuda():
    assert _choose("xpu", cuda=True, xpu=False) == "cuda"


def test_force_xpu_unavailable_no_gpu_falls_back_to_cpu():
    assert _choose("xpu", cuda=False, xpu=False) == "cpu"


# --- fp16 policy ----------------------------------------------------------
def test_fp16_enabled_on_gpu_backends():
    assert supports_fp16("cuda")
    assert supports_fp16("xpu")


def test_fp16_disabled_on_cpu():
    assert not supports_fp16("cpu")
