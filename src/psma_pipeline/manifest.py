"""Reproducibility manifest and optional-software detection.

The manifest records everything needed to reproduce or audit a run: date,
accession, package and tool versions, the full merged configuration, config
file hashes, endpoints, cache paths, the structures used and their hashes,
random seeds and method versions. It is written as JSON and shown as the
``Reproducibility`` sheet.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .cache import sha256_file
from .config import Config

# Bump a method's version whenever its scientific behaviour changes.
METHOD_VERSIONS = {
    "survey.process_entry": "1.1 (M1: identifiers, status columns, assembly reason)",
    "assemblies.choose_assembly": "1.1 (author → software → first, reason recorded)",
    "pockets.binding_site_analysis": "1.1 (unique-water count; UniProt mismatch flag)",
    "glycans.find_glycosylation_sites": "1.0",
    "active_site.active_site_analysis": "1.1 (mutated residues excluded from summaries)",
    "conformations.lid_conformation": "1.0",
    "conformations.structural_variability": "1.0",
    "sanity.run_sanity_checks": "1.1 (Zn long contacts; offset-based numbering check)",
    "interfaces.analyse_interface": "1.0",
    "interfaces.lattice_interfaces": "1.0",
    "interfaces.compare_interfaces": "1.0 (consensus canonical set)",
    "sasa.buried_surface": "1.0",
    "local_fit.categorise": "1.0 (X-ray RSCC/RSRZ; EM Q vs Pintilie 2020 expectation)",
    "ranking.rank_templates": "1.1 (exclude = any condition; empty filter)",
}

PYTHON_PACKAGES = {
    "gemmi": "gemmi", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "scikit-learn": "sklearn", "networkx": "networkx", "requests": "requests", "pyyaml": "yaml",
    "openpyxl": "openpyxl", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "biopython": "Bio", "freesasa": "freesasa", "rdkit": "rdkit", "MDAnalysis": "MDAnalysis",
    "mdtraj": "mdtraj", "openmm": "openmm", "pdbfixer": "pdbfixer", "propka": "propka",
    "pdb2pqr": "pdb2pqr", "tmtools": "tmtools", "pyarrow": "pyarrow", "plip": "plip",
    "modeller": "modeller", "openbabel": "openbabel",
}
EXECUTABLES = ["mkdssp", "dssp", "fpocket", "obabel", "antechamber", "tleap", "parmchk2",
               "gmx", "chimerax", "apbs", "propka3", "pdb2pqr30", "phenix.molprobity", "reduce"]


def detect_software() -> pd.DataFrame:
    """Availability and version of optional Python packages and executables.

    Returns a table with columns Software, Kind, Available, Version, Location.
    Nothing is installed or executed beyond ``import`` / ``which``.
    """
    rows = []
    for dist, module in PYTHON_PACKAGES.items():
        try:
            importlib.import_module(module)
            avail = True
        except Exception:  # noqa: BLE001 - any import problem means "not usable"
            avail = False
        try:
            ver = md.version(dist)
        except md.PackageNotFoundError:
            ver = None
        rows.append({"Software": dist, "Kind": "python package", "Available": avail,
                     "Version": ver if avail else None, "Location": module})
    for exe in EXECUTABLES:
        path = shutil.which(exe)
        rows.append({"Software": exe, "Kind": "executable", "Available": path is not None,
                     "Version": None, "Location": path})
    return pd.DataFrame(rows)


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (out.stdout.strip() + (" (uncommitted changes)" if dirty else "")) or None
    except Exception:  # noqa: BLE001
        return None


def build_manifest(cfg: Config, *, structure_ids: list[str], input_files: list[Path] | None = None,
                   software: pd.DataFrame | None = None, extra: dict[str, Any] | None = None) -> dict:
    """Assemble the reproducibility manifest as a JSON-serialisable dict."""
    software = software if software is not None else detect_software()
    files = []
    for p in input_files or []:
        p = Path(p)
        if p.exists():
            files.append({"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    cfg_files = []
    for src in cfg.sources:
        p = Path(src)
        if p.exists():
            cfg_files.append({"path": str(p), "sha256": sha256_file(p)})
        else:
            cfg_files.append({"path": src, "sha256": None})
    return {
        "run_date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "uniprot_accession": cfg.accession,
        "pipeline_version": __version__,
        "git_commit": _git_commit(cfg.project_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "random_seed": cfg.get("run.random_seed"),
        "method_versions": METHOD_VERSIONS,
        "endpoints": cfg.get("endpoints"),
        "cache_path": str(cfg.cache_dir),
        "outputs_path": str(cfg.outputs_dir),
        "config_files": cfg_files,
        "configuration": cfg.raw,
        "structure_ids": list(structure_ids),
        "input_file_hashes": files,
        "software": software.to_dict(orient="records"),
        **(extra or {}),
    }


def write_manifest(manifest: dict, path: Path) -> Path:
    """Write the manifest as indented JSON and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def manifest_table(manifest: dict) -> pd.DataFrame:
    """Flatten the manifest into an Item / Value table for the Reproducibility sheet."""
    rows: list[tuple[str, Any]] = []

    def walk(prefix: str, v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x)
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            for i, x in enumerate(v):
                walk(f"{prefix}[{i}]", x)
        else:
            rows.append((prefix, json.dumps(v, default=str) if isinstance(v, (list, tuple)) else v))

    walk("", manifest)
    return pd.DataFrame(rows, columns=["Item", "Value"])
