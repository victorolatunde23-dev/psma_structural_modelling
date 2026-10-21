#!/usr/bin/env python3
"""Command-line equivalent of notebooks/01_structure_survey.ipynb.

Examples
--------
    python scripts/run_survey.py                          # validation set, then full survey
    python scripts/run_survey.py --validation-only
    python scripts/run_survey.py --ids 3BHX 2XEG --tag test
    python scripts/run_survey.py --config my_config.yaml --previous old_workbook.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psma_pipeline import load_config, setup_logging  # noqa: E402
from psma_pipeline.excel_output import compare_workbooks  # noqa: E402
from psma_pipeline.sanity import (critical_failures, run_sanity_checks, select_validation_set,  # noqa: E402
                                  validation_ids)
from psma_pipeline.survey import fetch_entry_metadata, run_survey, save_survey  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="main YAML (default config/default_config.yaml)")
    ap.add_argument("--overrides", help="YAML merged on top of the main config")
    ap.add_argument("--ids", nargs="*", help="explicit PDB IDs (skips the validation set)")
    ap.add_argument("--tag", default="", help="suffix for output names")
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--previous", help="earlier workbook to compare against")
    ap.add_argument("--force", action="store_true", help="continue despite critical sanity failures")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config, args.overrides)

    def run(ids, tag):
        res = run_survey(cfg, ids=ids)
        sanity = run_sanity_checks(res.tables, cfg)
        paths = save_survey(cfg, res, extra_sheets={"SanityChecks": sanity}, tag=tag)
        crit = critical_failures(sanity)
        print(f"[{tag or 'full'}] {len(res.ids)} entries -> {paths['workbook']}")
        print(sanity.groupby(["Check", "Status"]).size().to_string())
        return res, crit, paths

    if args.ids:
        _, crit, paths = run(args.ids, args.tag)
    else:
        _, meta = fetch_entry_metadata(cfg)
        sel = select_validation_set(meta, cfg, cfg.get("validation_set.previous_workbook"))
        print(sel.to_string(index=False))
        _, crit, paths = run(validation_ids(sel, meta), "validation")
        if not crit.empty and not args.force:
            print("Critical sanity checks failed on the validation set; full run stopped.\n"
                  + crit.to_string(index=False))
            return 2
        if not args.validation_only:
            _, crit, paths = run(None, args.tag)
    if args.previous:
        cmp = compare_workbooks(args.previous, paths["workbook"])
        out = paths["workbook"].with_name(paths["workbook"].stem + "_comparison.csv")
        cmp.to_csv(out, index=False)
        print(f"comparison with {args.previous}: {len(cmp)} rows -> {out}")
    return 0 if crit.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
