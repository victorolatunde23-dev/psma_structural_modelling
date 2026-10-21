"""Workbook writing, data dictionary coverage, backwards compatibility and comparison."""

from __future__ import annotations

import pandas as pd

from psma_pipeline.excel_output import compare_workbooks, data_dictionary, write_workbook
from psma_pipeline.survey import ORIGINAL_SHEETS, save_survey

# Columns of the original notebook's workbook that must survive the refactor
# (checked on sheets that are populated without coordinates).
ORIGINAL_COLUMNS = {
    "Summary": ["PDB ID", "Title", "Release date", "Method", "Resolution (Å)", "R-free", "Space group",
                "Target chains (auth)", "Construct (UniProt range, extensions, mutations)",
                "Missing residues (UniProt numbering)", "Missing internal residues", "Zn ASU", "Zn assembly",
                "Ca ASU", "Ca assembly", "Cl ASU", "Cl assembly", "Ligands of interest", "Glycans",
                "Glycan chains ASU", "Glycan chains assembly", "Glycan residues assembly (NAG+MAN+...)",
                "Additives / buffer", "Other polymer entities", "Assembly used", "RCSB oligomeric state",
                "Target copies in ASU", "Target copies in assembly", "Monomer / dimer", "Assembly note",
                "Warnings", "Clashscore", "Ramachandran outliers (%)", "Rotamer outliers (%)",
                "RSRZ outliers (%)", "Clashscore percentile (similar resolution)"],
    "Chains": ["PDB ID", "Entity", "Chain (auth)", "Chain (label)", "Description", "Entity length",
               "Observed residues", "UniProt range (SIFTS)", "UniProt coverage",
               "N-terminal extension (non-UniProt)", "C-terminal extension (non-UniProt)",
               "Substitutions vs UniProt", "Mutation (author annotation)", "Internal gaps / insertions",
               "Missing residues (UniProt numbering)", "N missing (UniProt)", "Missing internal (loop) residues",
               "N missing internal", "Missing non-UniProt (tag) residues", "Missing residues (entity label_seq)",
               "Entity sequence"],
    "Ligands": ["PDB ID", "Component", "Name", "Class", "Count in ASU", "Count in assembly", "Assembly",
                "Chains (auth)"],
    "Assemblies": ["PDB ID", "Assembly", "Used in summary", "Defined by", "Method details", "Oligomeric details",
                   "Total polymer chains", "Target copies (gemmi)", "Target state", "Symmetry", "Stoichiometry",
                   "RCSB oligomeric state"],
    "CrossCheck": ["PDB ID", "In RCSB", "In PDBe", "In UniProt", "RCSB resolution", "PDBe resolution",
                   "UniProt resolution", "RCSB method", "PDBe method", "UniProt method",
                   "PDBe chains (UniProt range)", "UniProt chains", "Flags"],
    "TemplateRanking": ["PDB ID", "Rank", "Method", "Resolution (Å)", "Clashscore", "Ramachandran outliers (%)",
                        "Best UniProt coverage", "Score (0-1, higher = better)"],
}


def test_original_sheets_and_columns_preserved(offline_survey, cfg, tmp_path):
    paths = save_survey(cfg, offline_survey, workbook=tmp_path / "wb.xlsx")
    wb = pd.read_excel(paths["workbook"], sheet_name=None)
    assert list(wb)[:14] == ORIGINAL_SHEETS
    for sheet, cols in ORIGINAL_COLUMNS.items():
        missing = [c for c in cols if c not in wb[sheet].columns]
        assert not missing, f"{sheet} lost columns {missing}"
    for extra in ("ActiveSiteDefinition", "RunIssues", "Software", "Reproducibility", "DataDictionary"):
        assert extra in wb
    assert paths["manifest"].exists() and (paths["tables_dir"] / "Summary.csv").exists()


def test_every_written_column_is_documented(offline_survey, cfg, tmp_path):
    paths = save_survey(cfg, offline_survey, workbook=tmp_path / "wb.xlsx")
    dd = pd.read_excel(paths["workbook"], sheet_name="DataDictionary")
    undocumented = dd[dd["Description"] == "UNDOCUMENTED"]
    assert undocumented.empty, undocumented[["Sheet", "Column"]].values.tolist()


def test_missing_values_are_empty_cells(tmp_path):
    p = write_workbook(tmp_path / "x.xlsx", {"S": pd.DataFrame({"PDB ID": ["A", "B"], "v": [1.0, None],
                                                                 "Calculation status": ["OK", "FAIL"]})})
    back = pd.read_excel(p, sheet_name="S")
    assert pd.isna(back.loc[1, "v"])
    assert "DataDictionary" in pd.read_excel(p, sheet_name=None)


def test_compare_workbooks_reports_changes(tmp_path):
    a = pd.DataFrame({"PDB ID": ["A", "B"], "Resolution (Å)": [1.5, 2.0], "old": [1, 2]})
    b = pd.DataFrame({"PDB ID": ["A", "B", "C"], "Resolution (Å)": [1.5, 2.5, 3.0], "new": [1, 2, 3]})
    write_workbook(tmp_path / "a.xlsx", {"Summary": a}, add_dictionary=False)
    write_workbook(tmp_path / "b.xlsx", {"Summary": b}, add_dictionary=False)
    cmp = compare_workbooks(tmp_path / "a.xlsx", tmp_path / "b.xlsx")
    kinds = set(cmp["Kind"])
    assert {"column removed", "column added", "row count", "rows only in new", "values differ", "cell"} <= kinds
    cell = cmp[cmp["Kind"] == "cell"].iloc[0]
    assert cell["Key"] == "B" and cell["Old"] == 2.0 and cell["New"] == 2.5


def test_dictionary_covers_identifier_columns():
    dd = data_dictionary({"BindingSites": pd.DataFrame(columns=["PDB ID", "Model", "Chain (label)", "Entity",
                                                                 "Residue no.", "Calculation status"])})
    assert (dd["Description"] != "UNDOCUMENTED").all()
    assert dd.set_index("Column").loc["Residue no.", "Numbering"] == "author"
