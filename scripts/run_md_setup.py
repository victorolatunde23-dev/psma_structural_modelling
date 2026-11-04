#!/usr/bin/env python3
"""Milestone 6 stages (notebook 05) run inside the separate MD environment.

AmberTools, PySCF and OpenMM live in their own conda environment (Python 3.11, built by notebook 05,
section 0), so they never have to match Colab's own Python. The notebook calls this script with that
environment's Python; each stage saves its workbook as usual and writes ``<stage>_results.json``
(tables + issues) for the notebook to display.

    python scripts/run_md_setup.py selftest --outputs /content/drive/MyDrive/psma_structural_modelling/outputs
    python scripts/run_md_setup.py zinc     --outputs ... --cache ... [--no-gpu]
    python scripts/run_md_setup.py system   --outputs ... --cache ... [--no-md]

Exit status 0 = finished with no critical failure; 2 = a critical check failed; 1 = error.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from psma_pipeline import load_config, setup_logging  # noqa: E402
from psma_pipeline.md_env import dump_results  # noqa: E402
from psma_pipeline.sanity import critical_failures  # noqa: E402


def dump(path: Path, stage: str, tables: dict, issues, extra: dict) -> None:
    print(f"results: {dump_results(path, stage, tables, issues, extra)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["selftest", "zinc", "system"])
    ap.add_argument("--outputs")
    ap.add_argument("--cache")
    ap.add_argument("--prepared", help="prepared model (default: notebook 04 default microstate)")
    ap.add_argument("--no-gpu", action="store_true", help="PySCF on CPU")
    ap.add_argument("--no-md", action="store_true", help="system: build only, skip OpenMM")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    setup_logging()
    paths = {k: v for k, v in (("outputs", a.outputs), ("cache", a.cache)) if v}
    cfg = load_config(overrides={"paths": paths} if paths else None)
    res_file = cfg.outputs_dir / "md_setup" / f"{a.stage}_results.json"
    t0 = time.time()
    try:
        if a.stage == "selftest":
            from psma_pipeline.qm_bridge import gpu_available, self_test
            st = self_test(cfg.outputs_dir / "md_setup" / "qm_selftest", use_gpu=(not a.no_gpu) and gpu_available())
            dump(res_file, a.stage, {"SelfTest": pd.DataFrame([{"Item": k, "Value": v} for k, v in st.items()])}, None,
                 {"passed": bool(st.get("passed")), "minutes": round((time.time() - t0) / 60, 1)})
            return 0 if st.get("passed") else 2
        prep = Path(a.prepared) if a.prepared else (cfg.outputs_dir / "model_preparation" / cfg.get("preparation.name")
                                                    / "microstates" / "microstate_00_default.pdb")
        if not prep.exists():
            print(f"{prep} not found: run notebook 04 first", file=sys.stderr)
            return 1
        if a.stage == "zinc":
            from psma_pipeline.qm_bridge import gpu_available
            from psma_pipeline.zinc_params import run_zinc_parameters, save_zinc_parameters
            res = run_zinc_parameters(cfg, prep, use_gpu=(not a.no_gpu) and gpu_available())
            checks = "ZincChecks"
            saved = save_zinc_parameters(cfg, res, tag=a.tag) if checks in res.tables else {}
        else:
            from psma_pipeline.system_build import run_system_build, save_system
            res = run_system_build(cfg, prep, run_md=not a.no_md)
            checks = "SystemChecks"
            saved = save_system(cfg, res, tag=a.tag) if checks in res.tables else {}
        crit = critical_failures(res.tables[checks]) if checks in res.tables else None
        ok = crit is not None and crit.empty
        dump(res_file, a.stage, res.tables, res.issues.to_frame(),
             {"workdir": str(res.workdir), "saved": {k: str(v) for k, v in saved.items()},
              "critical_ok": ok, "minutes": round((time.time() - t0) / 60, 1)})
        return 0 if ok else 2
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        dump(res_file, a.stage, {}, None, {"error": traceback.format_exc(), "minutes": round((time.time() - t0) / 60, 1)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
