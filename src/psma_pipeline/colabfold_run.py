"""Run AlphaFold2 inside Colab with ColabFold (Milestone 4 model generation).

Everything stays in the Colab session: ColabFold is installed into the
runtime, MSAs come from the ColabFold MMseqs2 server, and AlphaFold2 /
AlphaFold2-multimer runs on the Colab GPU. Results are written under the
cache (Google Drive in Colab), so a disconnected run resumes: finished jobs
are skipped, and ColabFold itself skips finished queries.

Jobs are configured in ``alphafold.colabfold.jobs``. The default for PSMA is
the full-length homodimer (two copies, ``SEQ:SEQ``), which AlphaFold2-multimer
v3 predicts with the TM helices, so the membrane stage can use the C2 axis.
AlphaFold2 does not model metal ions or glycans: they are listed as blockers
in the MD readiness table.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .status import IssueLog, Status

log = logging.getLogger("psma_pipeline.colabfold_run")

INSTALL_SPEC = "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"


def colabfold_available() -> bool:
    return shutil.which("colabfold_batch") is not None


def gpu_name() -> str | None:
    """GPU model reported by nvidia-smi, or None without a GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def install_colabfold(quiet: bool = True) -> bool:
    """Install ColabFold into the running Colab session (uses Colab's own GPU JAX, as ColabFold's notebook does)."""
    if colabfold_available():
        return True
    cmd = [sys.executable, "-m", "pip", "install", "--no-warn-conflicts", INSTALL_SPEC]
    if quiet:
        cmd.insert(4, "-q")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("ColabFold install failed: %s", r.stderr[-2000:])
    return colabfold_available()


def job_sequence(useq: str, copies: int = 1, start: int = 1, end: int | None = None) -> str:
    """ColabFold query sequence: one chain, or ``copies`` identical chains joined by ':' (complex)."""
    seq = useq[start - 1:end] if end else useq[start - 1:]
    return ":".join([seq] * int(copies))


def write_query(job: str, sequence: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    fasta = folder / f"{job}.fasta"
    text = f">{job}\n{sequence}\n"
    if not fasta.exists() or fasta.read_text() != text:
        fasta.write_text(text)
    return fasta


def colabfold_command(fasta: Path, out_dir: Path, opts: dict) -> list[str]:
    """``colabfold_batch`` command line from ``alphafold.colabfold`` options."""
    cmd = ["colabfold_batch", "--msa-mode", str(opts.get("msa_mode", "mmseqs2_uniref_env")),
           "--model-type", str(opts.get("model_type", "auto")),
           "--num-models", str(int(opts.get("num_models", 5))),
           "--num-seeds", str(int(opts.get("num_seeds", 1)))]
    if opts.get("num_recycle") is not None:
        cmd += ["--num-recycle", str(int(opts["num_recycle"]))]
    if opts.get("pair_mode"):
        cmd += ["--pair-mode", str(opts["pair_mode"])]
    if opts.get("templates"):
        cmd.append("--templates")
    if opts.get("amber"):
        cmd += ["--amber", "--num-relax", "1"]
    cmd += [str(fasta), str(out_dir)]
    return cmd


def job_done(out_dir: Path, job: str, num_models: int) -> bool:
    ranks = list(out_dir.glob(f"{job}_*rank_*.pdb"))
    scores = list(out_dir.glob(f"{job}_scores_rank_*.json"))
    return (out_dir / f"{job}.done.txt").exists() or (len(scores) >= num_models and len(ranks) >= num_models)


def run_colabfold_jobs(cfg: Any, useq: str, issues: IssueLog, jobs: dict | None = None,
                       dry_run: bool = False) -> list[dict]:
    """Run the configured AlphaFold2 jobs; return uploads for :func:`alphafold.run_alphafold`.

    Each finished job becomes ``{"kind": "colabfold", "path": <result folder>, "label": <job>, "stage": False}``.
    Failures are recorded in ``issues`` and the remaining jobs still run.
    """
    opts = dict(cfg.get("alphafold.colabfold", {}) or {})
    jobs = jobs if jobs is not None else (opts.get("jobs") or {})
    root = cfg.cache_dir / "alphafold" / "colabfold"
    out: list[dict] = []
    for job, spec in jobs.items():
        spec = spec or {}
        if not spec.get("enabled", True):
            continue
        seq = job_sequence(useq, int(spec.get("copies", 1)), int(spec.get("start", 1)), spec.get("end"))
        jdir = root / job
        fasta = write_query(job, seq, jdir / "input")
        n_models = int(spec.get("num_models", opts.get("num_models", 5)))
        jopts = {**opts, **{k: v for k, v in spec.items() if k not in ("copies", "start", "end", "enabled")}}
        cmd = colabfold_command(fasta, jdir, jopts)
        if job_done(jdir, job, n_models):
            log.info("AlphaFold2 job %s already finished: %s", job, jdir)
        elif dry_run:
            log.info("dry run: %s", " ".join(cmd))
            continue
        else:
            if not colabfold_available():
                issues.add("colabfold", Status.NOT_RUN, f"job {job}: colabfold_batch not installed "
                           "(run install_colabfold() in a GPU runtime)")
                continue
            if gpu_name() is None:
                issues.add("colabfold", Status.WARNING, f"job {job}: no GPU detected - AlphaFold2 will be very slow")
            n_res = len(seq.replace(":", ""))
            log.info("AlphaFold2 job %s: %d residues in %d chain(s), %d model(s) -> %s",
                     job, n_res, seq.count(":") + 1, n_models, jdir)
            t0 = time.time()
            with open(jdir / "colabfold_stdout.log", "a") as fh:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                   env={**os.environ, "TF_FORCE_UNIFIED_MEMORY": "1",
                                        "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0"})
            if r.returncode != 0 or not list(jdir.glob(f"{job}_*rank_001*.pdb")):
                issues.add("colabfold", Status.FAIL, f"job {job}: colabfold_batch exited {r.returncode}; "
                           f"see {jdir / 'colabfold_stdout.log'}")
                continue
            log.info("AlphaFold2 job %s finished in %.1f min", job, (time.time() - t0) / 60)
        if list(jdir.glob(f"{job}_*rank_001*.pdb")):
            out.append({"kind": "colabfold", "path": str(jdir), "label": job, "stage": False,
                        "command": " ".join(cmd)})
    return out
