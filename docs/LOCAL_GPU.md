# Running notebook 05 on your own computer's GPU

Notebook 05 (zinc parameters and membrane system) can run on your own machine instead of Colab. The code is the same; only where it runs changes.

## What you need

| | Works | Notes |
|---|---|---|
| **Linux + NVIDIA GPU** | ✅ | PySCF (QM) and OpenMM (MD) both use the GPU |
| **Windows + NVIDIA GPU** | ✅ through **WSL2** | AmberTools has no native Windows version. WSL2 (Ubuntu) sees the NVIDIA GPU with the normal Windows driver. |
| **macOS (Apple silicon or Intel)** | ⚠️ CPU QM only | No CUDA: PySCF runs on the CPU, and OpenMM uses OpenCL/CPU |

You also need:
- an NVIDIA driver recent enough for **CUDA 12**: `nvidia-smi` should run and show "CUDA Version: 12.x" or higher;
- about **10 GB** of free disk for the environment.

## One-time setup

**Windows only:** install WSL2 with Ubuntu (in PowerShell as administrator: `wsl --install -d Ubuntu`), then restart. Keep your normal Windows NVIDIA driver; do *not* install a Linux driver inside WSL. Open the *Ubuntu* app and do everything below in it.

In a terminal (Linux, the WSL2 Ubuntu app, or macOS):

```bash
nvidia-smi                                   # should list your GPU (skip on macOS)
git clone https://github.com/victorolatunde23-dev/psma_structural_modelling.git
cd psma_structural_modelling
python3 -m venv ~/psma_venv && source ~/psma_venv/bin/activate
pip install -e . jupyter
```

The notebook's first run then builds the separate MD environment (AmberTools, PySCF, OpenMM) in `~/psma_mdenv`. This takes about 5–15 min and happens **once ever**, not once per session as in Colab. To put it somewhere else, set the `PSMA_MDENV` environment variable to the folder.

## Your earlier results

Notebook 05 needs the outputs of notebooks 01–04, which are on your Google Drive. Use one of these:

- **Google Drive for desktop** (Windows/macOS) syncs the Drive folder to your computer. Point `DATA_DIR` (cell 0a) at the synced folder:
  - WSL2: `"/mnt/g/My Drive/psma_structural_modelling"` (if Drive for desktop uses the letter G:);
  - macOS: `"~/Library/CloudStorage/GoogleDrive-<you>@gmail.com/My Drive/psma_structural_modelling"`.

  Results then go back to Drive automatically, where Colab and the other notebooks can see them. Inside WSL2, writing to `/mnt/g` is slower than the Linux disk, but these stages write little.
- **Copy the folder.** Download `psma_structural_modelling` from Drive, unzip it, and point `DATA_DIR` at it. Upload the `outputs` folder back to Drive afterwards.

`DATA_DIR` must be the folder that *contains* `outputs` and `cache`.

## Running it: choose one

**A. Jupyter on your machine.** This is the simplest.

```bash
source ~/psma_venv/bin/activate
cd psma_structural_modelling
jupyter lab notebooks/05_md_system_setup.ipynb
```

Set `DATA_DIR` in cell 0a, then *Run all*.

**B. The Colab page, computed on your machine.**

```bash
source ~/psma_venv/bin/activate
cd psma_structural_modelling
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
```

Copy the `http://localhost:8888/?token=…` address it prints. In Colab, choose **Connect ▾ → Connect to a local runtime** and paste the address. Set `DATA_DIR` in cell 0a. The Colab Drive mount is not available in this mode, so `DATA_DIR` is required.

## How to tell it is using your GPU

Cell 0b prints:
- `NVIDIA GPU:` your card;
- `PySCF on GPU: [true, "gpu4pyscf …"]`, which is what matters for the QM step. If it says false, it gives the reason; send that line;
- `openmm platforms:` should include `CUDA`.

While the QM runs, `nvidia-smi` in another terminal should show a `python` process using the GPU.

## If it is interrupted

The geometry optimisation saves its progress after every step, and every finished QM step is cached in `outputs/md_setup/zinc_mcpb/`. Re-running continues from where it stopped.
