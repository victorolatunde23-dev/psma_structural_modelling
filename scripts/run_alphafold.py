#!/usr/bin/env python3
"""Command-line equivalent of notebooks/03_alphafold_comparison.ipynb (Milestone 4 + membrane context).

    python scripts/run_alphafold.py                                   # AlphaFold DB model only
    python scripts/run_alphafold.py --af3 fold_psma_dimer.zip --colabfold cf_results.zip
    python scripts/run_alphafold.py --model my_model.pdb --plddt-in-b --no-deeptmhmm --tag test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psma_pipeline import load_config, setup_logging  # noqa: E402
from psma_pipeline.alphafold import run_alphafold, save_alphafold  # noqa: E402
from psma_pipeline.assessment import load_survey_tables  # noqa: E402
from psma_pipeline.sanity import critical_failures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--overrides")
    ap.add_argument("--survey", help="survey workbook from notebook 01")
    ap.add_argument("--assessment", help="assessment workbook from notebook 02 (canonical interface)")
    ap.add_argument("--af3", nargs="*", default=[], help="AlphaFold 3 / AlphaFold Server result zips or folders")
    ap.add_argument("--colabfold", nargs="*", default=[], help="ColabFold result zips or folders")
    ap.add_argument("--model", nargs="*", default=[], help="other PDB/mmCIF models")
    ap.add_argument("--plddt-in-b", action="store_true", help="--model files hold pLDDT in the B-factor column")
    ap.add_argument("--no-afdb", action="store_true")
    ap.add_argument("--no-deeptmhmm", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.overrides)
    if args.no_deeptmhmm:
        cfg = cfg.with_overrides({"membrane": {"deeptmhmm": {"enabled": False}}})
    survey = load_survey_tables(args.survey or cfg.output_file("survey_workbook"))
    a_path = Path(args.assessment) if args.assessment else cfg.output_file("assessment_workbook")
    assessment = load_survey_tables(a_path) if a_path.exists() else None
    uploads = ([{"kind": "af3", "path": p} for p in args.af3] + [{"kind": "colabfold", "path": p} for p in args.colabfold]
               + [{"kind": "model", "path": p, "plddt_in_b": args.plddt_in_b} for p in args.model])
    res = run_alphafold(cfg, survey, assessment, uploads=uploads, use_afdb=not args.no_afdb)
    paths = save_alphafold(cfg, res, tag=args.tag)
    checks = res.tables["AlphaFoldChecks"]
    print(checks.groupby(["Check", "Status"]).size().to_string())
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 2 if not critical_failures(checks).empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
