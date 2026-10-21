#!/usr/bin/env python3
"""Command-line equivalent of notebook 02 part 2 (Milestone 3): pockets, conserved waters, variants.

    python scripts/run_pockets.py                                   # all survey entries
    python scripts/run_pockets.py --ids 3BHX 2XEG --tag test
    python scripts/run_pockets.py --ligand <CCD ID> --subpockets "S1' glutamate pocket" --lid closed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psma_pipeline import load_config, setup_logging  # noqa: E402
from psma_pipeline.assessment import load_survey_tables  # noqa: E402
from psma_pipeline.pocket_analysis import rank_for_ligand, run_pocket_analysis, save_pocket_analysis  # noqa: E402
from psma_pipeline.sanity import critical_failures, run_pocket_checks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--overrides")
    ap.add_argument("--survey", help="survey workbook from notebook 01")
    ap.add_argument("--assessment", help="assessment workbook from notebook 02 part 1 (optional)")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ligand", help="reference ligand component ID for the ligand-conditioned ranking")
    ap.add_argument("--smiles", help="reference SMILES (ligand not in the PDB)")
    ap.add_argument("--subpockets", nargs="*", default=None)
    ap.add_argument("--lid", choices=["open", "closed"])
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.overrides)
    tables = load_survey_tables(args.survey or cfg.output_file("survey_workbook"))
    apath = Path(args.assessment or cfg.output_file("assessment_workbook"))
    assessment = load_survey_tables(apath) if apath.exists() else None
    res = run_pocket_analysis(cfg, tables, assessment, ids=args.ids)
    if args.ligand or args.smiles:
        res.tables["LigandConditionedRanking"] = rank_for_ligand(res, cfg, comp=args.ligand, smiles=args.smiles,
                                                                 required_subpockets=args.subpockets,
                                                                 lid_state=args.lid)
    checks = run_pocket_checks(res.tables, tables, ids=args.ids)
    res.tables["PocketChecks"] = checks
    paths = save_pocket_analysis(cfg, res, tag=args.tag)
    print(checks[["Check", "Status", "Detail"]].to_string(index=False))
    print("workbook:", paths["workbook"])
    return 2 if not critical_failures(checks).empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
