import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": s.strip("\n").splitlines(True)})

md("""
# 07 · Trajectory analysis — part 1: stability by region

**Milestone 7.** Reads the production trajectory that notebook 06 saves on Drive (`outputs/md_run/`). It answers: **which parts of the protein are moving?**

The model is a hybrid:

| Region (UniProt, each chain) | Source | Expected behaviour |
|---|---|---|
| 1–19 cytoplasmic tail | AlphaFold (low confidence) | very flexible |
| 20–43 transmembrane helix | AlphaFold | moves as a body in the membrane |
| 44–61 stalk | AlphaFold | flexible linker |
| **62–750 ectodomain** | **9S7U crystal structure** | **should stay close to the start** |
| 541–548 entrance lid | crystal | can open and close |

Every frame is superposed on the crystal ectodomain of both chains. The RMSD of each region then shows how it moves **relative to** the well-determined part. "All Cα, global fit" is the number notebook 06 logs.

**Runtime:** a **CPU** runtime is enough: only numpy is used, and no GPU units are needed. It can run while production runs in another session; it only reads finished segments.
""")
code("""
#@title 0 · the pipeline
import os, sys, subprocess
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
DATA_DIR = None
REPO = "https://github.com/victorolatunde23-dev/psma_structural_modelling.git"
if IN_COLAB and not Path("psma_structural_modelling").exists() and not Path("src/psma_pipeline").exists():
    subprocess.run(["git", "clone", "-q", REPO], check=True)
if Path("psma_structural_modelling").exists():
    os.chdir("psma_structural_modelling")
if IN_COLAB:
    subprocess.run(["git", "pull", "-q"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
sys.path.insert(0, str(Path("src").resolve()))
for _m in [m for m in list(sys.modules) if m == "psma_pipeline" or m.startswith("psma_pipeline.")]:
    del sys.modules[_m]
import psma_pipeline
print("psma_pipeline", psma_pipeline.__version__)
""")
code("""
#@title 1 · data
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from psma_pipeline import load_config, setup_logging
from psma_pipeline.colab import local_overrides, storage_overrides
from psma_pipeline import md_analysis as MA
setup_logging()
cfg = load_config(overrides=(storage_overrides(True) if IN_COLAB else local_overrides(DATA_DIR)) or None)
MD = cfg.outputs_dir / "md_run"
OUT7 = cfg.outputs_dir / "md_analysis"
OUT7.mkdir(parents=True, exist_ok=True)
files = MA.segment_files(MD, "prod")
print(f"finished production segment files: {len(files)}; last starts at {files[-1][0] / 1000:.2f} ns" if files else "no production yet")
STRIDE = 1   #@param {type:"integer"}
# every frame is 10 ps; STRIDE = 5 uses one frame per 50 ps (faster for long runs)
""")
md("## A · RMSD by region")
code("""
ca = MA.ca_table(MD / "solute.pdb")
df, rf = MA.stability(MD, ca, frame_ps=float(cfg.get("md.production.frame_ps", 10)), stride=STRIDE)
print(f"{len(df)} frames, {df['time (ns)'].iloc[0]:.2f}–{df['time (ns)'].iloc[-1]:.2f} ns of production")
S = MA.summary(df)
display(S)
df.to_csv(OUT7 / "rmsd_by_region.csv", index=False); rf.to_csv(OUT7 / "rmsf_ca.csv", index=False); S.to_csv(OUT7 / "rmsd_summary.csv", index=False)
""")
code("""
t = df["time (ns)"]
fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
for c, lab in (("all Cα, global fit", "all Cα (as logged)"), ("ectodomain, crystal (62–750), both chains", "ectodomain, both chains"),
               ("ectodomain, chain A (own fit)", "ectodomain A (own fit)"), ("ectodomain, chain B (own fit)", "ectodomain B (own fit)")):
    ax[0].plot(t, df[c], label=lab)
ax[0].set_title("Whole protein vs crystal ectodomain"); ax[0].legend(fontsize=8)
for ch, ls in (("A", "-"), ("B", "--")):
    for reg in ("cytoplasmic tail (1–19)", "transmembrane helix (20–43)", "stalk (44–61)"):
        ax[1].plot(t, df[f"{reg}, chain {ch}"], ls, label=f"{reg.split(' (')[0]} {ch}")
ax[1].set_title("AlphaFold anchor, relative to the ectodomain"); ax[1].legend(fontsize=7, ncol=2)
for ch in ("A", "B"):
    ax[2].plot(t, df[f"entrance lid (541–548), chain {ch}"], label=f"lid {ch}")
ax[2].set_title("Entrance lid (541–548), relative to the ectodomain"); ax[2].legend(fontsize=8)
for a in ax: a.set_xlabel("production time (ns)"); a.set_ylabel("RMSD (Å)")
plt.tight_layout(); plt.savefig(OUT7 / "rmsd_by_region.png", dpi=150); plt.show()
""")
md("## B · Flexibility per residue (RMSF)")
code("""
fig, ax = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
for i, ch in enumerate("AB"):
    r = rf[rf["chain"] == ch]
    ax[i].plot(r["uniprot"], r["RMSF (Å)"], lw=0.8)
    ax[i].axvspan(1, 61, color="orange", alpha=0.15, label="AlphaFold anchor (1–61)")
    ax[i].axvspan(541, 548, color="red", alpha=0.2, label="entrance lid")
    ax[i].set_ylabel(f"chain {ch}\\nRMSF (Å)")
ax[0].legend(fontsize=8); ax[1].set_xlabel("UniProt residue")
plt.tight_layout(); plt.savefig(OUT7 / "rmsf_ca.png", dpi=150); plt.show()
top = rf[rf["uniprot"] >= 62].sort_values("RMSF (Å)", ascending=False).head(15)
print("Most flexible residues of the crystal ectodomain:"); display(top)
""")
md("""
## How to read this

- **The whole-protein RMSD is high, but the ectodomain is low (about 1–2.5 Å) and flat:** the drift comes from the AlphaFold anchor (tails, transmembrane helices, stalk). That is expected, and the crystal part is stable.
- **The ectodomain RMSD climbs steadily:** the extracellular part itself is changing. Look at the two chains (own fit) and at the RMSF peaks to see where.
- **The transmembrane helix RMSD is large and steady:** the helices have tilted or moved as a body in the membrane relative to the ectodomain. That says something about how the AlphaFold anchor sits; it is not a sign of instability.
- **The lid RMSD changes:** the entrance lid is moving, which is relevant for ligand access (analysed in later parts of M7).

Outputs: `outputs/md_analysis/` (`rmsd_by_region.csv`, `rmsd_summary.csv`, `rmsf_ca.csv` and the two figures).
""")

nb = {"nbformat": 4, "nbformat_minor": 0, "cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}}
json.dump(nb, open("/home/claude/psma_structural_modelling/notebooks/07_md_analysis.ipynb", "w"), indent=1)
print(len(cells), "cells")
