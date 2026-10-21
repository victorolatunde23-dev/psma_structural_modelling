#!/usr/bin/env python3
"""Command-line equivalent of notebooks/02_interface_pocket_validation.ipynb (Milestone 2).

    python scripts/run_assessment.py                      # uses outputs/<ACC>_structures.xlsx
    python scripts/run_assessment.py --survey my_survey.xlsx --ids 3BHX 4P44 --tag test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psma_pipeline import load_config, setup_logging  # noqa: E402
from psma_pipeline.assessment import load_survey_tables, run_assessment, save_assessment  # noqa: E402
from psma_pipeline.sanity import critical_failures, run_assessment_checks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--overrides")
    ap.add_argument("--survey", help="survey workbook from notebook 01")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.overrides)
    tables = load_survey_tables(args.survey or cfg.output_file("survey_workbook"))
    res = run_assessment(cfg, tables, ids=args.ids)
    checks = run_assessment_checks(res.tables, tables["Summary"])
    res.tables["AssessmentChecks"] = checks
    paths = save_assessment(cfg, res, tag=args.tag)
    print(checks.groupby(["Check", "Status"]).size().to_string())
    print("workbook:", paths["workbook"])
    return 2 if not critical_failures(checks).empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
