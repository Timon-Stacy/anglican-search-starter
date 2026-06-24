"""Environment / GPU sanity check for the Anglican search project.

Verifies that the cu128 torch build actually sees and can run a kernel on the
Blackwell (sm_120) card, and that faiss / sentence-transformers import cleanly.
"""

import platform
import sys


def main() -> int:
    print(f"Python : {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    import torch

    print(f"torch  : {torch.__version__}")
    print(f"CUDA build : {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    print(f"cuda.is_available() : {cuda_ok}")

    if cuda_ok:
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"device : {name}")
        print(f"compute capability : sm_{major}{minor}")
        print(f"total VRAM : {total_gb:.1f} GiB")
        print(f"arch list supported by this torch build : {torch.cuda.get_arch_list()}")

        # Actually run a kernel — is_available() alone can lie if the arch is
        # unsupported; a real matmul proves Blackwell kernels are present.
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        print(f"GPU matmul OK (checksum={c:.1f})")
    else:
        print("!! CUDA not available — torch will fall back to CPU.")

    import faiss

    print(f"faiss  : {faiss.__version__}")

    import sentence_transformers

    print(f"sentence-transformers : {sentence_transformers.__version__}")

    import mcp

    print(f"mcp SDK : {getattr(mcp, '__version__', 'installed')}")

    return 0 if cuda_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
