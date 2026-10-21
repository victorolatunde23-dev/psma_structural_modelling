"""Synthetic, network-free inputs that mimic RCSB / PDBe / UniProt responses.

They exercise the metadata-only code paths (no coordinates): construct
differences, SIFTS numbering, assembly choice, ligand classification,
validation merging, cross-check and ranking.
"""

from __future__ import annotations

ACC = "Q04609"


def uniprot_sequence() -> str:
    """750-residue sequence with the configured active-site residues in place."""
    seq = ["A"] * 750
    for code in ["R210", "N257", "H377", "D387", "E424", "E425", "D453", "R463", "R511", "R534",
                 "R536", "W541", "Y552", "H553", "K699", "Y700"]:
        seq[int(code[1:]) - 1] = code[0]
    return "".join(seq)


USEQ = uniprot_sequence()


def uniprot_entry() -> dict:
    return {
        "sequence": {"value": USEQ},
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Glutamate carboxypeptidase 2"}}},
        "uniProtKBCrossReferences": [
            {"database": "PDB", "id": "1AAA", "properties": [
                {"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "1.50 A"},
                {"key": "Chains", "value": "A/B=44-750"}]},
            {"database": "PDB", "id": "2BBB", "properties": [
                {"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "2.10 A"},
                {"key": "Chains", "value": "A=44-750"}]},
            {"database": "PDB", "id": "9ZZZ", "properties": [
                {"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "2.00 A"},
                {"key": "Chains", "value": "A=44-750"}]},
        ],
        "features": [
            {"type": "Binding site", "location": {"start": {"value": 377}, "end": {"value": 377}},
             "ligand": {"name": "Zn(2+)"}},
            {"type": "Active site", "location": {"start": {"value": 424}, "end": {"value": 424}},
             "description": "Nucleophile"},
            {"type": "Topological domain", "location": {"start": {"value": 44}, "end": {"value": 750}},
             "description": "Extracellular"},
        ],
    }


def _poly(pid: str, ent: str, seq: str, asyms: list[str], auths: list[str], acc: str | None = ACC,
          desc: str = "Glutamate carboxypeptidase 2", mutation: str | None = None) -> dict:
    refs = [{"database_name": "UniProt", "database_accession": acc}] if acc else []
    return {"rcsb_id": f"{pid}_{ent}",
            "entity_poly": {"type": "polypeptide(L)", "pdbx_seq_one_letter_code_can": seq},
            "rcsb_polymer_entity": {"pdbx_description": desc, "pdbx_mutation": mutation},
            "rcsb_polymer_entity_container_identifiers": {
                "entity_id": ent, "asym_ids": asyms, "auth_asym_ids": auths,
                "reference_sequence_identifiers": refs}}


def _np(comp: str, name: str, asyms: list[str], auths: list[str]) -> dict:
    return {"pdbx_entity_nonpoly": {"comp_id": comp, "name": name},
            "rcsb_nonpolymer_entity_container_identifiers": {"asym_ids": asyms, "auth_asym_ids": auths}}


def _asm(pid: str, aid: str, details: str, count: int, state: str = "Homo 2-mer") -> dict:
    return {"rcsb_id": f"{pid}-{aid}",
            "pdbx_struct_assembly": {"id": aid, "details": details, "method_details": "PISA",
                                     "oligomeric_details": "dimeric", "oligomeric_count": count},
            "rcsb_struct_symmetry": [{"kind": "Global Symmetry", "type": "Cyclic",
                                      "oligomeric_state": state, "stoichiometry": ["A2"], "symbol": "C2"}]}


def _entry(pid: str, method: str, res: float | None, polys: list, nonpoly=(), branched=(), assemblies=(),
           date: str = "2010-05-05T00:00:00Z") -> dict:
    return {"rcsb_id": pid, "struct": {"title": f"Synthetic {pid}"}, "exptl": [{"method": method}],
            "rcsb_accession_info": {"initial_release_date": date},
            "rcsb_entry_info": {"resolution_combined": [res] if res else None},
            "refine": [{"ls_R_factor_R_free": 0.21}] if "X-RAY" in method else [],
            "symmetry": {"space_group_name_H_M": "I 2 2 2"} if "X-RAY" in method else None,
            "assemblies": list(assemblies), "polymer_entities": polys,
            "nonpolymer_entities": list(nonpoly), "branched_entities": list(branched)}


ECD = USEQ[43:750]                     # UniProt 44-750
TAG = "GSHHHHHH"


def metadata() -> dict:
    return {
        # 2 chains in ASU, author dimer, ligand + Zn + glycan
        "1AAA": _entry("1AAA", "X-RAY DIFFRACTION", 1.5,
                       [_poly("1AAA", "1", ECD, ["A", "B"], ["A", "B"])],
                       [_np("ZN", "ZINC ION", ["C", "D", "E", "F"], ["A", "A", "B", "B"]),
                        _np("CA", "CALCIUM ION", ["G", "H"], ["A", "B"]),
                        _np("CL", "CHLORIDE ION", ["I", "J"], ["A", "B"]),
                        _np("XYZ", "INHIBITOR", ["K", "L"], ["A", "B"]),
                        _np("GOL", "GLYCEROL", ["M"], ["A"]),
                        _np("NAG", "N-ACETYL-D-GLUCOSAMINE", ["N"], ["A"])],
                       [{"rcsb_branched_entity": {"pdbx_description": "NAG-NAG"},
                         "rcsb_branched_entity_container_identifiers": {"asym_ids": ["O", "P"],
                                                                        "auth_asym_ids": ["A", "B"]}}],
                       [_asm("1AAA", "1", "author_and_software_defined_assembly", 2)]),
        # 1 chain in ASU, dimer by symmetry; N-terminal tag; E424A mutant
        "2BBB": _entry("2BBB", "X-RAY DIFFRACTION", 2.1,
                       [_poly("2BBB", "1", TAG + ECD[:380] + "A" + ECD[381:], ["A"], ["A"],
                              mutation="E424A")],
                       [_np("ZN", "ZINC ION", ["B", "C"], ["A", "A"])],
                       assemblies=[_asm("2BBB", "1", "author_defined_assembly", 2)]),
        # software-only assembly; no SIFTS mapping in PDBe; apo
        "3CCC": _entry("3CCC", "X-RAY DIFFRACTION", 3.0,
                       [_poly("3CCC", "1", ECD, ["A"], ["A"])],
                       assemblies=[_asm("3CCC", "2", "software_defined_assembly", 1, "Monomer"),
                                   _asm("3CCC", "1", "software_defined_assembly", 2)]),
        # cryo-EM with Fab chains (heteromer)
        "4DDD": _entry("4DDD", "ELECTRON MICROSCOPY", 3.3,
                       [_poly("4DDD", "1", ECD, ["A", "B"], ["A", "B"]),
                        _poly("4DDD", "2", "QVQLVESG" * 15, ["C", "D"], ["H", "L"], acc=None,
                              desc="Fab heavy chain")],
                       [_np("ZN", "ZINC ION", ["E", "F", "G", "H"], ["A", "A", "B", "B"])],
                       assemblies=[_asm("4DDD", "1", "author_defined_assembly", 4, "Hetero 4-mer")],
                       date="2023-02-01T00:00:00Z"),
    }


def sifts(pid: str) -> dict:
    segs = {
        "1AAA": [("A", "A", 1, 707, 44, 750), ("B", "B", 1, 707, 44, 750)],
        "2BBB": [("A", "A", 9, 715, 44, 750)],
        "4DDD": [("A", "A", 1, 707, 44, 750), ("B", "B", 1, 707, 44, 750)],
    }.get(pid)
    if not segs:
        return {}
    return {pid.lower(): {"UniProt": {ACC: {"mappings": [
        {"entity_id": 1, "chain_id": auth, "struct_asym_id": asym,
         "start": {"residue_number": s}, "end": {"residue_number": e}, "unp_start": us, "unp_end": ue}
        for asym, auth, s, e, us, ue in segs]}}}}


def best_structures() -> list[dict]:
    return [
        {"pdb_id": "1aaa", "chain_id": "A", "unp_start": 44, "unp_end": 750, "resolution": 1.5,
         "experimental_method": "X-ray diffraction"},
        {"pdb_id": "1aaa", "chain_id": "B", "unp_start": 44, "unp_end": 750, "resolution": 1.5,
         "experimental_method": "X-ray diffraction"},
        {"pdb_id": "2bbb", "chain_id": "A", "unp_start": 44, "unp_end": 750, "resolution": 2.2,
         "experimental_method": "X-ray diffraction"},
        {"pdb_id": "4ddd", "chain_id": "A", "unp_start": 44, "unp_end": 750, "resolution": 3.3,
         "experimental_method": "Electron Microscopy"},
    ]


def rcsb_validation() -> dict:
    return {"1AAA": {"clashscore": 2.0, "rama_outliers_pct": 0.1, "rotamer_outliers_pct": 1.0,
                     "rsrz_outliers_pct": 1.5, "dcc_rfree": 0.2},
            "2BBB": {"clashscore": None, "rama_outliers_pct": 0.5, "rotamer_outliers_pct": None,
                     "rsrz_outliers_pct": 2.0, "dcc_rfree": None},
            "3CCC": {"clashscore": 9.0, "rama_outliers_pct": 1.2, "rotamer_outliers_pct": 3.0,
                     "rsrz_outliers_pct": 4.0, "dcc_rfree": 0.3},
            "4DDD": {"clashscore": 6.0, "rama_outliers_pct": 0.3, "rotamer_outliers_pct": 2.0,
                     "rsrz_outliers_pct": None, "dcc_rfree": None}}


def pdbe_validation(pid: str) -> tuple[dict, dict]:
    pct = {"2BBB": {"PDBe clashscore": 4.5, "PDBe clashscore percentile (vs similar resolution)": 80,
                    "PDBe clashscore percentile (whole PDB)": 70}}.get(pid, {})
    return pct, {}
