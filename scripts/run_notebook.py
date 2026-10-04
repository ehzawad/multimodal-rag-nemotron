"""Execute a notebook top to bottom without a browser and save the executed copy.

Meant to be launched on a Colab GPU with colab-persist, e.g.:

    colab-persist run scripts/run_notebook.py --source . --project mmrag -- notebooks/multimodal_rag.ipynb

The kernel runs in a scratch directory outside the persisted workspace (downloaded embeddings, images and model
caches don't need to be checkpointed); the executed notebook is written to $COLAB_OUTPUT_DIR.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbclient", "nbformat", "ipykernel"], check=True)

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

notebook_path = Path(sys.argv[1]).resolve()
output_dir = Path(os.environ.get("COLAB_OUTPUT_DIR", "outputs")).resolve()
run_dir = Path(os.environ.get("NOTEBOOK_RUN_DIR", "/content/notebook_run"))
output_dir.mkdir(parents=True, exist_ok=True)
run_dir.mkdir(parents=True, exist_ok=True)

# Lets the notebook skip interactive-only cells (e.g. a blocking demo.launch())
os.environ["HEADLESS"] = "1"

nb = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(nb,
                        timeout=int(os.environ.get("CELL_TIMEOUT", 3600)),
                        kernel_name="python3",
                        resources={"metadata": {"path": str(run_dir)}})

executed_path = output_dir / f"executed_{notebook_path.name}"
start_time = time.time()
status = 0
try:
    client.execute()
except CellExecutionError as e:
    print(f"[ERROR] Cell failed:\n{e}")
    status = 1
finally:
    nbformat.write(nb, executed_path)
    print(f"[INFO] Total time: {time.time() - start_time:.1f}s | Executed notebook: {executed_path}")

# Print the text outputs so they show up in the terminal running colab-persist
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for output in cell.get("outputs", []):
        text = output.get("text") or output.get("data", {}).get("text/plain") or ""
        if output.get("output_type") == "error":
            text = f"{output['ename']}: {output['evalue']}"
        if text.strip():
            print(f"--- [cell {i}] ---\n{text[:3000]}")

sys.exit(status)
