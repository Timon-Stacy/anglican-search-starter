"""Environment / accelerator sanity check for the Anglican search project.

Detects the compute backend the same way the engine does (Nvidia CUDA, Intel
Arc XPU, or CPU), runs a real matmul on whichever GPU is present (is_available()
alone can lie if the arch is unsupported), and confirms faiss /
sentence-transformers / mcp import cleanly.

    uv run python scripts/check_env.py                 # autodetect
    ANGLICAN_DEVICE=xpu uv run python scripts/check_env.py   # force a backend
"""

import os
import platform
import sys

# Make `anglican_search` importable when run straight from the repo.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _check_cuda(torch) -> bool:
    if not torch.cuda.is_available():
        return False
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    major, minor = torch.cuda.get_device_capability(i)
    print("backend : CUDA (Nvidia)")
    print(f"device : {torch.cuda.get_device_name(i)}")
    print(f"compute capability : sm_{major}{minor}")
    print(f"total VRAM : {props.total_memory / 1024**3:.1f} GiB")
    print(f"arch list supported by this torch build : {torch.cuda.get_arch_list()}")
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"GPU matmul OK (checksum={c:.1f})")
    return True


def _check_xpu(torch) -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        return False
    i = xpu.current_device()
    print("backend : XPU (Intel Arc / Data Center GPU)")
    print(f"device : {xpu.get_device_name(i)}")
    try:
        props = xpu.get_device_properties(i)
        total = getattr(props, "total_memory", None)
        if total:
            print(f"total VRAM : {total / 1024**3:.1f} GiB")
    except Exception as e:  # noqa: BLE001 - props API varies by torch version
        print(f"(device properties unavailable: {e})")
    a = torch.randn(2048, 2048, device="xpu")
    b = torch.randn(2048, 2048, device="xpu")
    c = (a @ b).sum().item()
    xpu.synchronize()
    print(f"GPU matmul OK (checksum={c:.1f})")
    return True


def main() -> int:
    print(f"Python : {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    import torch

    print(f"torch  : {torch.__version__}")
    print(f"CUDA build : {torch.version.cuda}")
    print(f"XPU build  : {getattr(getattr(torch, 'version', None), 'xpu', None)}")

    from anglican_search.device import select_device

    chosen = select_device()
    print(f"engine would select : {chosen}"
          f"  (ANGLICAN_DEVICE={os.environ.get('ANGLICAN_DEVICE') or 'unset'})")
    print("-" * 60)

    gpu_ok = _check_cuda(torch) or _check_xpu(torch)
    if not gpu_ok:
        print("!! No CUDA or XPU GPU available — torch will run on CPU.")
        print("   (That's expected on a CPU-only serving box.)")
    print("-" * 60)

    import faiss
    print(f"faiss  : {faiss.__version__}")

    import sentence_transformers
    print(f"sentence-transformers : {sentence_transformers.__version__}")

    import mcp
    print(f"mcp SDK : {getattr(mcp, '__version__', 'installed')}")

    return 0 if gpu_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
