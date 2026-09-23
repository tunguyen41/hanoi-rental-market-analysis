"""Run both notebooks from clean kernels using the Python interpreter invoking this script."""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent


def main():
    for name in ['01_data_audit.ipynb', '02_rental_eda.ipynb']:
        path = ROOT / 'notebooks' / name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        print(f'Executing {name} with {sys.executable}', flush=True)
        client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                                resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)
        print(f'Saved executed notebook: {path}', flush=True)
    print(f'Report: {ROOT / "reports/eda_report.md"}')


if __name__ == '__main__':
    main()
