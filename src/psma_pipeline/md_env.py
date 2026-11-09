"""The separate MD environment used by notebook 05 (AmberTools, PySCF, OpenMM).

Colab's Python moves on (3.13 in late 2026) faster than AmberTools' conda builds, so installing
AmberTools into Colab's own Python (condacolab) can fail to solve. Instead, notebook 05 builds an
independent micromamba environment with its own Python 3.11 and runs the Milestone 6 stages there
through ``scripts/run_md_setup.py``. The notebook's own Python only displays the results.

The same code runs on your own machine (Linux, Windows via WSL2, macOS): see ``docs/LOCAL_GPU.md``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PACKAGES = ["python=3.11", "ambertools>=23", "packmol", "pyscf", "geometric", "openmm>=8.1",
            "numpy", "pandas>=2.2", "scipy", "openpyxl", "pyyaml", "requests", "gemmi", "matplotlib", "seaborn",
            "scikit-learn", "networkx", "pytest"]


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def default_prefix() -> str:
    """Where the MD environment lives: Colab's scratch disk, or ``PSMA_MDENV`` / ``~/psma_mdenv`` on your own machine."""
    if in_colab():
        return "/content/mdenv"
    return os.environ.get("PSMA_MDENV", str(Path.home() / "psma_mdenv"))


ENV_PREFIX = default_prefix()


def mamba_platform() -> str:
    """micromamba platform for this machine. Windows needs WSL2 (AmberTools has no native Windows build)."""
    sysname, mach = platform.system(), platform.machine().lower()
    if sysname == "Linux":
        return "linux-aarch64" if mach in ("aarch64", "arm64") else "linux-64"
    if sysname == "Darwin":
        return "osx-arm64" if mach == "arm64" else "osx-64"
    raise RuntimeError("AmberTools does not run natively on Windows: install WSL2 (Ubuntu) and run this notebook "
                       "inside it; see docs/LOCAL_GPU.md")


def nvidia_gpu() -> str | None:
    """Name of the NVIDIA GPU (from nvidia-smi), or None."""
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None


def _mamba(prefix: str) -> str:
    """An existing micromamba / mamba / conda on PATH, else a private micromamba next to the environment."""
    for t in ("micromamba", "mamba", "conda"):
        if shutil.which(t):
            return shutil.which(t)
    return str(Path(prefix).parent / "micromamba" / "bin" / "micromamba")


def env_python(prefix: str | None = None) -> Path:
    return Path(prefix or ENV_PREFIX) / "bin" / "python"


def env_ready(prefix: str | None = None) -> bool:
    b = Path(prefix or ENV_PREFIX) / "bin"
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


def install_env(repo: str | Path, prefix: str | None = None, gpu: bool = True, log: str | Path | None = None) -> dict:
    """Create the MD environment once (5–15 min) and install this repository into it.
    Works in Colab and on your own Linux / WSL2 / macOS machine; the GPU packages are added only
    when an NVIDIA GPU is visible (nvidia-smi)."""
    prefix = prefix or ENV_PREFIX
    log = Path(log) if log else Path(prefix).parent / "mdenv_install.log"
    plat = mamba_platform()
    mamba = _mamba(prefix)
    if not Path(mamba).exists():
        root = Path(mamba).parents[1]
        Path(mamba).parent.mkdir(parents=True, exist_ok=True)
        _sh(f"curl -Ls https://micro.mamba.pm/api/micromamba/{plat}/latest | tar -xj -C '{root}' bin/micromamba", log)
    if not env_ready(prefix):
        pk = " ".join(f"'{p}'" for p in PACKAGES)
        _sh(f"'{mamba}' create -y -q -p '{prefix}' -c conda-forge {pk}", log)
    py = env_python(prefix)
    _sh(f"'{py}' -m pip install -q --no-deps -e '{repo}'", log)
    card = nvidia_gpu()
    gpu_ok = False
    if gpu and card and plat.startswith("linux"):
        gpu_ok = _ensure_pyscf_gpu(py, log)
    elif gpu:
        print("No NVIDIA GPU found (nvidia-smi): PySCF will use the CPU" if not card else
              "PySCF GPU support needs Linux or WSL2: PySCF will use the CPU")
    return {"machine": "Colab" if in_colab() else f"local ({plat})", "NVIDIA GPU": card, "prefix": prefix,
            "python": str(py), "gpu4pyscf installed": gpu_ok, "install log": str(log), **versions(prefix)}


def pyscf_gpu_status(py: str | Path) -> tuple[bool, str]:
    """Ask the MD environment itself whether PySCF can use the GPU (and why not)."""
    r = subprocess.run([str(py), "-c", "import json; from psma_pipeline.qm_bridge import gpu_status; "
                        "print(json.dumps(gpu_status()))"], capture_output=True, text=True)
    try:
        ok, why = json.loads(r.stdout.strip().splitlines()[-1])
        return bool(ok), why
    except Exception:  # noqa: BLE001
        return False, (r.stderr or r.stdout)[-400:]


def _ensure_pyscf_gpu(py: Path, log: Path) -> bool:
    """Install gpu4pyscf and check it really works; if the CUDA libraries are missing, add them as pip
    wheels (cupy's ``ctk`` extra) and check again. Prints the reason if PySCF still cannot use the GPU."""
    attempts = [("gpu4pyscf", "gpu4pyscf-cuda12x cutensor-cu12"),
                ("CUDA libraries for cupy", "'cupy-cuda12x[ctk]' nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 "
                                            "nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-curand-cu12 "
                                            "nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12")]
    why = "not tried"
    for label, pkgs in attempts:
        try:
            _sh(f"'{py}' -m pip install -q {pkgs}", log)
        except RuntimeError:
            print(f"pip could not install {label}")
        ok, why = pyscf_gpu_status(py)
        if ok:
            print("PySCF will use the GPU:", why)
            return True
    print("PySCF will use the CPU. Reason:", why)
    return False


def versions(prefix: str | None = None) -> dict:
    code = ("import json, importlib\nv={}\nfor m in ('pyscf','openmm','pandas','numpy'):\n"
            "    try: v[m]=importlib.import_module(m).__version__\n    except Exception as e: v[m]=None\n"
            "try:\n    import openmm as o; v['openmm platforms']=[o.Platform.getPlatform(i).getName() for i in range(o.Platform.getNumPlatforms())]\n"
            "except Exception: pass\n"
            "try:\n    from psma_pipeline.qm_bridge import gpu_status; v['PySCF on GPU']=gpu_status()\n"
            "except Exception as e: v['PySCF on GPU']=str(e)\nprint(json.dumps(v))")
    r = subprocess.run([str(env_python(prefix)), "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": r.stderr[-500:]}


def run_stage(stage: str, outputs: str | Path, cache: str | Path | None = None, repo: str | Path = ".",
              prefix: str | None = None, extra: list[str] | None = None) -> dict:
    """Run one stage of ``scripts/run_md_setup.py`` in the MD environment, streaming its output;
    return the loaded results (``returncode``, ``extra``, ``tables``)."""
    prefix = prefix or ENV_PREFIX
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
