"""Ligand chemistry: chemical-component definitions and (optional) RDKit descriptors.

* Definitions (name, formula, weight, formal charge, SMILES, InChI, InChIKey)
  come from the RCSB Chemical Component Dictionary via GraphQL (cached).
* Descriptors (heavy atoms, MW, H-bond donors/acceptors, rotatable bonds,
  aromatic rings) and chemical similarity (Morgan radius-2, 2048-bit
  Tanimoto) need RDKit. Without RDKit, or without a SMILES, these values are
  left missing with status ``NOT RUN`` – nothing is estimated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .cache import cached_json
from .status import Status

log = logging.getLogger("psma_pipeline.ligand_chem")

CHEM_QUERY = """
query($ids: [String!]!) {
  chem_comps(comp_ids: $ids) {
    chem_comp { id name formula formula_weight pdbx_formal_charge type }
    rcsb_chem_comp_descriptor { SMILES SMILES_stereo InChI InChIKey }
  }
}"""


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_chem_comps(sess: requests.Session, comp_ids: list[str], cache: Path, url: str,
                     batch: int = 50, **http: Any) -> dict[str, dict]:
    """``{comp_id: {name, formula, weight, formal_charge, smiles, smiles_stereo, inchi, inchikey}}``."""
    out: dict[str, dict] = {}
    ids = sorted({c.upper() for c in comp_ids if c})
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        cf = cache / "chem_comp" / f"{chunk[0]}_{chunk[-1]}_{len(chunk)}.json"
        data = cached_json(sess, url, cf, method="POST", payload={"query": CHEM_QUERY, "variables": {"ids": chunk}},
                           **http)
        if data.get("errors"):
            cf.unlink(missing_ok=True)
            raise RuntimeError(f"chem_comp GraphQL error: {data['errors'][:1]}")
        for e in (data.get("data") or {}).get("chem_comps") or []:
            if not e:
                continue
            cc = e.get("chem_comp") or {}
            d = e.get("rcsb_chem_comp_descriptor") or {}
            out[str(cc.get("id", "")).upper()] = {
                "name": cc.get("name"), "formula": cc.get("formula"), "weight": cc.get("formula_weight"),
                "formal_charge": cc.get("pdbx_formal_charge"), "type": cc.get("type"),
                "smiles": d.get("SMILES"), "smiles_stereo": d.get("SMILES_stereo"),
                "inchi": d.get("InChI"), "inchikey": d.get("InChIKey")}
    return out


def descriptors(smiles: str | None) -> dict:
    """RDKit descriptors for a SMILES; values missing with a status when unavailable."""
    keys = ["Heavy atoms", "Molecular weight", "H-bond donors", "H-bond acceptors", "Rotatable bonds",
            "Aromatic rings", "Formal charge (RDKit)"]
    if not smiles:
        return {**dict.fromkeys(keys), "Descriptor status": Status.NOT_RUN.value, "Descriptor note": "no SMILES"}
    if not rdkit_available():
        return {**dict.fromkeys(keys), "Descriptor status": Status.NOT_RUN.value,
                "Descriptor note": "RDKit not installed"}
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {**dict.fromkeys(keys), "Descriptor status": Status.FAIL.value,
                "Descriptor note": "RDKit could not parse the SMILES"}
    return {"Heavy atoms": mol.GetNumHeavyAtoms(), "Molecular weight": round(Descriptors.MolWt(mol), 2),
            "H-bond donors": Lipinski.NumHDonors(mol), "H-bond acceptors": Lipinski.NumHAcceptors(mol),
            "Rotatable bonds": Lipinski.NumRotatableBonds(mol),
            "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "Formal charge (RDKit)": Chem.GetFormalCharge(mol),
            "Descriptor status": Status.OK.value, "Descriptor note": ""}


def chemical_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
    """Morgan (radius 2, 2048 bits) Tanimoto similarity; None without RDKit or SMILES."""
    if not (smiles_a and smiles_b and rdkit_available()):
        return None
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(ma), gen.GetFingerprint(mb)))


def ligand_table(defs: dict[str, dict], comp_ids: list[str]) -> pd.DataFrame:
    """One row per component: identifiers + descriptors (missing where unavailable)."""
    rows = []
    for c in sorted(set(comp_ids)):
        d = defs.get(c.upper(), {})
        smi = d.get("smiles_stereo") or d.get("smiles")
        rows.append({"Component": c, "Ligand name": d.get("name"), "Formula": d.get("formula"),
                     "Formula weight": d.get("weight"), "Formal charge (CCD)": d.get("formal_charge"),
                     "Canonical SMILES": d.get("smiles"), "Isomeric SMILES": d.get("smiles_stereo"),
                     "InChI": d.get("inchi"), "InChIKey": d.get("inchikey"),
                     "Definition status": Status.OK.value if d else Status.NOT_RUN.value,
                     **descriptors(smi)})
    return pd.DataFrame(rows)
