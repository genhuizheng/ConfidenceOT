"""Cross-platform installation diagnostic for ``python -m confidenceot``."""

from __future__ import annotations

import json
import platform
import sys

from confidenceot.cuda import cuda_available, cuda_device_name


def main() -> None:
    try:
        import torch
    except ImportError:
        torch_version = None
        torch_cuda_build = None
    else:
        torch_version = torch.__version__
        torch_cuda_build = torch.version.cuda
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torch_cuda_build": torch_cuda_build,
        "cuda_available": cuda_available(),
        "cuda_device": cuda_device_name(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
