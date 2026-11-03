#!/usr/bin/env python3
"""Command-line equivalent of notebooks/04_model_preparation.ipynb (Milestone 5).

    python scripts/run_preparation.py
    python scripts/run_preparation.py --zn-treatment nonbonded_12_6_4 --tag test
    python scripts/run_preparation.py --template 9S7U --no-propka
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psma_pipeline import load_config, setup_logging  # noqa: E402
from psma_pipeline.assessment import load_survey_tables  # noqa: E402
from psma_pipeline.model_preparation import run_preparation, save_preparation  # noqa: E402
from psma_pipeline.sanity import critical_failures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--overrides")
    ap.add_argument("--survey", help="survey workbook from notebook 01")
    ap.add_argument("--alphafold", help="AlphaFold/membrane workbook from notebook 03 (membrane topology)")
    ap.add_argument("--template", help="experimental template PDB ID (default: preparation.template)")
    ap.add_argument("--zn-treatment", choices=["nonbonded_12_6_4", "bonded_mcpb", "cationic_dummy"])
    ap.add_argument("--no-propka", action="store_true", help="skip PROPKA (default pKa values)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.overrides)
    over = {}
    if args.template:
        over["template"] = args.template.upper()
    if args.zn_treatment:
        over["zn_md_treatment"] = args.zn_treatment
    if over:
        cfg = cfg.with_overrides({"preparation": over})
    survey = load_survey_tables(args.survey or cfg.output_file("survey_workbook"))
    a_path = Path(args.alphafold) if args.alphafold else cfg.output_file("alphafold_workbook")
    alphafold = load_survey_tables(a_path) if a_path.exists() else None
    res = run_preparation(cfg, survey, alphafold, run_pka=not args.no_propka)
    paths = save_preparation(cfg, res, tag=args.tag)
    checks = res.tables["PrepChecks"]
    print(checks[["Check", "Critical", "Status", "Chain", "Detail"]].to_string(index=False))
    print(res.tables["PrepReadiness"].to_string(index=False))
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 2 if not critical_failures(checks).empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
