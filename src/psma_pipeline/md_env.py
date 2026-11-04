"""The separate MD environment used by notebook 05 (AmberTools, PySCF, OpenMM).

Colab's Python moves on (3.13 in late 2026) faster than AmberTools' conda builds, so installing
AmberTools into Colab's own Python (condacolab) can fail to solve. Instead, notebook 05 builds an
independent micromamba environment with its own Python 3.11 and runs the Milestone 6 stages there
through ``scripts/run_md_setup.py``. The notebook's own Python only displays the results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ENV_PREFIX = "/content/mdenv"
MAMBA = "/content/micromamba/bin/micromamba"
PACKAGES = ["python=3.11", "ambertools>=23", "packmol", "pyscf", "geometric", "openmm>=8.1",
            "numpy", "pandas>=2.2", "scipy", "openpyxl", "pyyaml", "requests", "gemmi", "matplotlib", "seaborn",
            "scikit-learn", "networkx", "pytest"]


def env_python(prefix: str = ENV_PREFIX) -> Path:
    return Path(prefix) / "bin" / "python"


def env_ready(prefix: str = ENV_PREFIX) -> bool:
    b = Path(prefix) / "bin"
    return all((b / t).exists() for t in ("python", "tleap", "MCPB.py", "packmol-memgen"))


def _sh(cmd: str, log: Path | None = None) -> None:
    """Run a shell command; on failure print the last 60 lines of output and raise."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if log:
        with open(log, "a") as fh:
            fh.write(f"$ {cmd}\n{out}\n")
    if r.returncode != 0:
        print("\n".join(out.splitlines()[-60:]))
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


def install_env(repo: str | Path, prefix: str = ENV_PREFIX, gpu: bool = True, log: str | Path = "/content/mdenv_install.log") -> dict:
    """Create the MD environment once (about 5 min) and install this repository into it."""
    log = Path(log)
    if not Path(MAMBA).exists():
        Path(MAMBA).parent.mkdir(parents=True, exist_ok=True)
        _sh(f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C {Path(MAMBA).parents[1]} bin/micromamba", log)
    if not env_ready(prefix):
        pk = " ".join(f"'{p}'" for p in PACKAGES)
        _sh(f"{MAMBA} create -y -q -p {prefix} -c conda-forge {pk}", log)
    py = env_python(prefix)
    _sh(f"{py} -m pip install -q --no-deps -e {repo}", log)
    gpu_ok = False
    if gpu:
        try:
            _sh(f"{py} -m pip install -q gpu4pyscf-cuda12x cutensor-cu12", log)
            gpu_ok = True
        except RuntimeError:
            print("gpu4pyscf not installed: PySCF will use the CPU")
    return {"prefix": prefix, "python": str(py), "gpu4pyscf": gpu_ok, **versions(prefix)}


def versions(prefix: str = ENV_PREFIX) -> dict:
    code = ("import json, importlib\nv={}\nfor m in ('pyscf','openmm','pandas','numpy'):\n"
            "    try: v[m]=importlib.import_module(m).__version__\n    except Exception as e: v[m]=None\n"
            "try:\n    import openmm as o; v['openmm platforms']=[o.Platform.getPlatform(i).getName() for i in range(o.Platform.getNumPlatforms())]\n"
            "except Exception: pass\nprint(json.dumps(v))")
    r = subprocess.run([str(env_python(prefix)), "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": r.stderr[-500:]}


def run_stage(stage: str, outputs: str | Path, cache: str | Path | None = None, repo: str | Path = ".",
              prefix: str = ENV_PREFIX, extra: list[str] | None = None) -> dict:
    """Run one stage of ``scripts/run_md_setup.py`` in the MD environment, streaming its output;
    return the loaded results (``returncode``, ``extra``, ``tables``)."""
    cmd = [str(env_python(prefix)), "-u", str(Path(repo) / "scripts" / "run_md_setup.py"), stage, "--outputs", str(outputs)]
    if cache:
        cmd += ["--cache", str(cache)]
    cmd += extra or []
    env = dict(os.environ)
    env["PATH"] = f"{Path(prefix) / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["AMBERHOME"] = str(prefix)
    env.pop("PYTHONPATH", None)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:                     # live progress in the notebook
        sys.stdout.write(line)
    rc = p.wait()
    f = Path(outputs) / "md_setup" / f"{stage}_results.json"
    res = load_results(f) if f.exists() else {"stage": stage, "extra": {}, "tables": {}}
    res["returncode"] = rc
    return res


def dump_results(path: Path, stage: str, tables: dict, issues: pd.DataFrame | None, extra: dict) -> Path:
    out = {"stage": stage, "extra": extra, "tables": {}}
    for k, v in dict(tables or {}, **({"Issues": issues} if issues is not None else {})).items():
        if isinstance(v, pd.DataFrame):
            out["tables"][k] = v.astype(object).where(v.notna(), None).to_dict(orient="split")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, default=str))
    return path


def load_results(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    d["tables"] = {k: pd.DataFrame(v["data"], columns=v["columns"]) for k, v in d["tables"].items()}
    return d
