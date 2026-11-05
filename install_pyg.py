"""Installs pyg-lib (needed by Node2Vec) for this machine's torch build. Run after requirements.txt: python install_pyg.py"""
import subprocess
import sys

import torch


def wheel_page(version, cuda):
    """'2.14.1+cpu', None -> .../torch-2.14.0+cpu.html; the wheels are built per minor version of torch."""
    minor = ".".join(version.split("+")[0].split(".")[:2])
    build = "cu" + cuda.replace(".", "") if cuda else "cpu"
    return f"https://data.pyg.org/whl/torch-{minor}.0+{build}.html"


assert wheel_page("2.14.1+cpu", None).endswith("torch-2.14.0+cpu.html")
assert wheel_page("2.8.0+cu126", "12.6").endswith("torch-2.8.0+cu126.html")

if __name__ == "__main__":
    page = wheel_page(torch.__version__, torch.version.cuda)
    if subprocess.call([sys.executable, "-m", "pip", "install", "-q", "pyg_lib", "-f", page]):
        sys.exit(f"No pyg-lib for torch {torch.__version__} at {page}. Only Node2Vec needs it. Fix: install a torch "
                 "version listed at https://data.pyg.org/whl/ (cpu, cu126 or cu130), restart, and run this again.")
    subprocess.check_call([sys.executable, "-c", "import pyg_lib, torch_geometric; print('PyG ready', torch_geometric.__version__)"])
