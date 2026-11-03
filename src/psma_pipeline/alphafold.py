"""AlphaFold DB and AlphaFold2 (ColabFold) import, confidence and comparison (Milestone 4).

Design: ``docs/M4_ALPHAFOLD_MEMBRANE.md``.

Every model becomes an :class:`AFModel` holding

* an atom table with the same columns as :mod:`structure_atoms` (author chain,
  residue number, UniProt number from sequence alignment, coordinates,
  B = pLDDT), plus ``copy`` / ``protomer`` so the Milestone 2 interface code
  works on it unchanged;
* a residue-level PAE matrix whose rows/columns are listed in ``pae_index``
  as ``(chain, UniProt number)``; the matrix size and chain blocks are checked
  against the model (critical check ``pae_indexing``).

Sources: the AlphaFold Protein Structure Database (current release, fetched by
accession), AlphaFold2 / AlphaFold2-multimer run inside Colab by
:mod:`colabfold_run` (or any other ColabFold result), and any PDB/mmCIF model.
"""

from __future__ import annotations

import glob
import itertools
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .cache import FetchError, cached_download, cached_json, sha256_file
from .numbering import align_to_uniprot, to_ranges
from .status import IssueLog, Status
from .structure_atoms import COLUMNS

log = logging.getLogger("psma_pipeline.alphafold")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


@dataclass
class AFModel:
    """One predicted model with confidence data, in UniProt numbering."""

    model_id: str
    source: str                       # AlphaFold DB | ColabFold (AlphaFold2) | user model
    kind: str                         # monomer | multimer
    atoms: pd.DataFrame
    pae: np.ndarray | None = None
    pae_index: list[tuple[str, int]] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    chain_info: list[dict] = field(default_factory=list)
    plddt_in_b: bool = True
    pae_raw_shape: tuple | None = None

    @property
    def target_chains(self) -> list[str]:
        return [c["Chain"] for c in self.chain_info if c.get("Target")]

    def ca(self, chain: str) -> dict[int, np.ndarray]:
        """``{UniProt number: CA xyz}`` for one target chain."""
        a = self.atoms
        s = a[(a["chain"] == chain) & (a["atom"] == "CA") & a["unp"].notna()]
        return {int(u): np.array([x, y, z]) for u, x, y, z in zip(s["unp"], s["x"], s["y"], s["z"])}

    def plddt(self, chain: str) -> dict[int, float]:
        if not self.plddt_in_b:
            return {}
        a = self.atoms
        s = a[(a["chain"] == chain) & (a["atom"] == "CA") & a["unp"].notna()]
        return {int(u): float(b) for u, b in zip(s["unp"], s["b"])}

    def pae_lookup(self) -> dict[tuple[str, int], int]:
        return {k: i for i, k in enumerate(self.pae_index)}


# ---------------------------------------------------------------------------
# Reading coordinates into an atom table
# ---------------------------------------------------------------------------
def _atoms_from_pdb_text(text: str) -> pd.DataFrame:
    """Minimal PDB reader (first model) used when gemmi is unavailable (offline tests)."""
    rows = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        alt = line[16].strip()
        if alt not in ("", "A"):
            continue
        resname = line[17:20].strip()
        el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if el in ("H", "D"):
            continue
        chain = line[21].strip() or "A"
        polymer = line.startswith("ATOM") and resname in THREE_TO_ONE
        rows.append((chain, chain, None, resname, int(line[22:26]), line[26].strip(), None, None,
                     line[12:16].strip(), el, alt, float(line[30:38]), float(line[38:46]), float(line[46:54]),
                     float(line[60:66] or 0), float(line[54:60] or 1), polymer, resname == "HOH", False))
    return pd.DataFrame(rows, columns=COLUMNS)


def _atoms_from_gemmi_model(path: Path) -> pd.DataFrame:
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    ent_of_sub = {sub: ent.name for ent in st.entities for sub in ent.subchains}
    rows = []
    for ch in st[0]:
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            polymer = bool(info and info.is_amino_acid()) and res.het_flag != "H" or res.name in THREE_TO_ONE
            seen: set[str] = set()
            for a in res:
                if a.element.name in ("H", "D") or a.name in seen:
                    continue
                seen.add(a.name)
                rows.append((ch.name, res.subchain or ch.name, ent_of_sub.get(res.subchain), res.name, res.seqid.num,
                             res.seqid.icode.strip(), res.label_seq, None, a.name, a.element.name.upper(),
                             "" if a.altloc == "\0" else a.altloc, a.pos.x, a.pos.y, a.pos.z, a.b_iso, a.occ,
                             polymer, res.is_water(), False))
    return pd.DataFrame(rows, columns=COLUMNS)


def read_model_atoms(path: str | Path) -> pd.DataFrame:
    """Heavy-atom table of the first model of a PDB or mmCIF file (gemmi, or a PDB fallback)."""
    path = Path(path)
    try:
        df = _atoms_from_gemmi_model(path)
    except ImportError:
        if path.suffix.lower() not in (".pdb", ".ent"):
            raise ImportError(f"reading {path.name} needs gemmi (pip install gemmi)")
        df = _atoms_from_pdb_text(path.read_text())
    df["unp"] = pd.to_numeric(df["unp"], errors="coerce")
    return df


def residue_order(atoms: pd.DataFrame) -> list[tuple[str, int, str, str]]:
    """Polymer residues in file order as ``(chain, resnum, icode, resname)``."""
    p = atoms[atoms["polymer"]]
    keys = list(dict.fromkeys(zip(p["chain"], p["resnum"], p["icode"], p["resname"])))
    return keys


def number_model(atoms: pd.DataFrame, useq: str, min_identity: float = 0.95,
                 min_identity_target: float = 0.5, min_length: int = 20) -> tuple[pd.DataFrame, list[dict]]:
    """Assign UniProt numbers by sequence alignment and flag target chains.

    Returns the updated atom table (``unp``, ``target``, ``entity``, ``copy``,
    ``protomer`` filled) and one info row per polymer chain.
    """
    atoms = atoms.copy()
    atoms["unp"] = np.nan
    atoms["target"] = False
    info = []
    order = residue_order(atoms)
    for chain in dict.fromkeys(k[0] for k in order):
        res = [k for k in order if k[0] == chain]
        seq = "".join(THREE_TO_ONE.get(r[3], "X") for r in res)
        mapping = align_to_uniprot(seq, useq)
        ident = sum(seq[i] == useq[u - 1] for i, u in mapping.items()) / max(len(mapping), 1)
        target = len(mapping) >= min_length and ident >= min_identity_target
        unp_of = {(r[1], r[2]): mapping.get(i) for i, r in enumerate(res)}
        sel = atoms["chain"] == chain
        atoms.loc[sel, "unp"] = [unp_of.get((n, ic)) for n, ic in zip(atoms.loc[sel, "resnum"], atoms.loc[sel, "icode"])]
        atoms.loc[sel, "target"] = target & atoms.loc[sel, "polymer"]
        mapped = sorted(u for u in mapping.values())
        status, warn = Status.OK, ""
        if not target:
            status, warn = Status.NOT_APPLICABLE, f"identity {ident:.2f} / {len(mapping)} aligned: not the target"
        elif ident < min_identity:
            status, warn = Status.WARNING, f"identity to UniProt {ident:.3f} < {min_identity}"
        info.append({"Chain": chain, "Residues": len(res), "Mapped residues": len(mapping),
                     "UniProt range": to_ranges(mapped) if mapped else "", "Identity to UniProt": round(ident, 4),
                     "Unmapped residues": len(res) - len(mapping), "Target": bool(target),
                     "Calculation status": status.value, "Warning": warn})
    atoms["unp"] = pd.to_numeric(atoms["unp"], errors="coerce")
    atoms["entity"] = atoms["entity"].fillna("model")
    atoms["label_asym"] = atoms["chain"]
    atoms["copy"] = "1"
    atoms["protomer"] = atoms["chain"] + "|1"
    return atoms, info


# ---------------------------------------------------------------------------
# PAE indexing
# ---------------------------------------------------------------------------
def _pae_array(obj: Any) -> np.ndarray | None:
    if isinstance(obj, list):
        obj = obj[0] if obj and isinstance(obj[0], dict) else {"pae": obj}
    if not isinstance(obj, dict):
        return None
    for key in ("predicted_aligned_error", "pae"):
        if key in obj and obj[key] is not None:
            return np.asarray(obj[key], float)
    return None


def pae_from_residue_matrix(pae: np.ndarray, atoms: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Residue-indexed PAE (AlphaFold DB, ColabFold): rows follow polymer residues in file order.

    Raises ValueError if the matrix size differs from the number of polymer residues.
    """
    order = residue_order(atoms)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE matrix is not square: {pae.shape}")
    if pae.shape[0] != len(order):
        raise ValueError(f"PAE size {pae.shape[0]} != polymer residues in model {len(order)}")
    unp = _unp_of_residues(atoms)
    keep = [(i, (k[0], unp[(k[0], k[1], k[2])])) for i, k in enumerate(order) if unp.get((k[0], k[1], k[2])) is not None]
    idx = [i for i, _ in keep]
    return pae[np.ix_(idx, idx)], [k for _, k in keep]


def _unp_of_residues(atoms: pd.DataFrame) -> dict[tuple, int | None]:
    t = atoms[atoms["polymer"] & atoms["target"]].drop_duplicates(["chain", "resnum", "icode"])
    return {(c, n, i): (int(u) if pd.notna(u) else None) for c, n, i, u in zip(t["chain"], t["resnum"], t["icode"], t["unp"])}


def pae_index_check(m: AFModel, diagonal_max: float = 1.5) -> dict:
    """Critical check: square, sized to the index, contiguous chain blocks, small diagonal."""
    row = {"Check": "pae_indexing", "Model ID": m.model_id}
    if m.pae is None:
        return {**row, "Status": Status.NOT_APPLICABLE.value, "Detail": "no PAE for this model"}
    problems = []
    if m.pae.shape[0] != m.pae.shape[1]:
        problems.append(f"not square {m.pae.shape}")
    if m.pae.shape[0] != len(m.pae_index):
        problems.append(f"size {m.pae.shape[0]} != index {len(m.pae_index)}")
    chains = [c for c, _ in m.pae_index]
    blocks = [k for k, _ in itertools.groupby(chains)]
    if len(blocks) != len(set(blocks)):
        problems.append(f"chain blocks not contiguous: {blocks}")
    for c in set(chains):
        nums = [u for cc, u in m.pae_index if cc == c]
        if nums != sorted(nums):
            problems.append(f"chain {c} rows not in UniProt order")
    if not problems and len(m.pae):
        diag = np.diag(m.pae)
        if np.median(diag) > diagonal_max:
            problems.append(f"median diagonal {np.median(diag):.2f} Å > {diagonal_max} (misaligned matrix?)")
    detail = "; ".join(problems) or f"{len(m.pae_index)} residues, chains {', '.join(dict.fromkeys(chains))}"
    return {**row, "Status": (Status.FAIL if problems else Status.PASS).value, "Detail": detail}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _finish(m: AFModel, useq: str, acfg: dict, issues: IssueLog) -> AFModel:
    m.atoms, m.chain_info = number_model(m.atoms, useq, float(acfg.get("min_identity", 0.95)),
                                         float(acfg.get("min_identity_target", 0.5)))
    n_target = len(m.target_chains)
    m.kind = "multimer" if n_target >= 2 else "monomer"
    for ci in m.chain_info:
        if ci["Calculation status"] == Status.WARNING.value:
            issues.add("alphafold", Status.WARNING, f"chain {ci['Chain']}: {ci['Warning']}", pdb_id=m.model_id)
    if not n_target:
        issues.add("alphafold", Status.FAIL, "no chain matches the target sequence", pdb_id=m.model_id)
    return m


def _set_pae(m: AFModel, pae: np.ndarray | None, issues: IssueLog) -> None:
    if pae is None:
        return
    m.pae_raw_shape = tuple(pae.shape)
    try:
        m.pae, m.pae_index = pae_from_residue_matrix(pae, m.atoms)
    except ValueError as ex:
        m.pae, m.pae_index = None, []
        m.meta["PAE problem"] = str(ex)
        issues.add("alphafold", Status.FAIL, f"PAE not usable: {ex}", pdb_id=m.model_id)


def afdb_entries(sess: requests.Session, acc: str, cache: Path, url_template: str, **http: Any) -> list[dict]:
    data = cached_json(sess, url_template.format(acc=acc), cache / "alphafold" / f"afdb_prediction_{acc}.json",
                       allow_404=True, **http)
    if isinstance(data, dict):
        return [data] if data else []
    return list(data or [])


def _afdb_field(e: dict, new: str, old: str, default: Any = None) -> Any:
    """Current AFDB field name, falling back to the name retired in June 2026."""
    v = e.get(new)
    return v if v is not None else e.get(old, default)


def load_afdb(sess: requests.Session, acc: str, useq: str, cfg: Any, issues: IssueLog,
              json_kw: dict | None = None, dl_kw: dict | None = None) -> list[AFModel]:
    """AlphaFold DB model(s) for ``acc``: the canonical-sequence entry, with pLDDT and PAE."""
    json_kw, dl_kw = json_kw or {}, dl_kw or {}
    cache = cfg.cache_dir
    try:
        entries = afdb_entries(sess, acc, cache, cfg.endpoint("afdb_prediction"), **json_kw)
    except FetchError as ex:
        issues.add("alphafold", Status.FAIL, f"AlphaFold DB request failed: {ex}")
        return []
    if not entries:
        issues.add("alphafold", Status.WARNING, f"AlphaFold DB has no prediction for {acc}")
        return []
    canonical = [e for e in entries if _afdb_field(e, "sequence", "uniprotSequence") == useq]
    chosen = canonical or entries[:1]
    if not canonical:
        issues.add("alphafold", Status.WARNING, "no AlphaFold DB entry has the current UniProt canonical sequence; "
                   "using the first entry (sequence version may differ)")
    models = []
    for e in chosen:
        mid = _afdb_field(e, "modelEntityId", "entryId", f"AF-{acc}-F1")
        url = e.get("cifUrl") or e.get("pdbUrl")
        if not url:
            issues.add("alphafold", Status.FAIL, "AlphaFold DB entry has no coordinate URL", pdb_id=mid)
            continue
        try:
            path = cached_download(sess, url, cache / "alphafold" / Path(url).name, **dl_kw)
            atoms = read_model_atoms(path)
        except Exception as ex:  # noqa: BLE001
            issues.add("alphafold", Status.FAIL, f"model download/read failed: {ex}", pdb_id=mid)
            continue
        version = e.get("latestVersion")
        m = AFModel(f"AFDB:{mid}" + (f"-v{version}" if version else ""), "AlphaFold DB", "monomer", atoms,
                    meta={"Version": f"v{version}" if version else "", "Model entity ID": mid,
                          "Model created": e.get("modelCreatedDate"), "File": str(path), "SHA-256": sha256_file(path),
                          "Sequence start": _afdb_field(e, "sequenceStart", "uniprotStart"),
                          "Sequence end": _afdb_field(e, "sequenceEnd", "uniprotEnd"),
                          "Canonical sequence": _afdb_field(e, "sequence", "uniprotSequence") == useq,
                          "Global pLDDT (AFDB)": e.get("globalMetricValue")})
        _finish(m, useq, cfg.get("alphafold", {}), issues)
        if e.get("paeDocUrl"):
            try:
                pdata = cached_json(sess, e["paeDocUrl"], cache / "alphafold" / f"{mid}_pae.json", **json_kw)
                _set_pae(m, _pae_array(pdata), issues)
            except FetchError as ex:
                issues.add("alphafold", Status.WARNING, f"PAE unavailable: {ex}", pdb_id=m.model_id)
        models.append(m)
    return models


def _stage(path: str | Path, dest: Path) -> Path:
    """Copy (or unzip) an uploaded result into the cache; originals are never modified."""
    path = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        out = dest / path.name
        if not out.exists():
            shutil.copytree(path, out)
        return out
    if path.suffix == ".zip":
        out = dest / path.stem
        if not out.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
        return out
    out = dest / path.name
    if not out.exists():
        shutil.copy2(path, out)
    return out


def _find(root: Path, *patterns: str) -> list[Path]:
    for p in patterns:
        hits = sorted(Path(x) for x in glob.glob(str(root / "**" / p), recursive=True))
        if hits:
            return hits
    return []


def load_colabfold(path: str | Path, useq: str, cfg: Any, issues: IssueLog, label: str | None = None,
                   stage: bool = True) -> list[AFModel]:
    """Every ranked model of a ColabFold (AlphaFold2 / AlphaFold2-multimer) result.

    ``stage=False`` reads a result folder in place (runs made in this session by
    :mod:`colabfold_run`, so a resumed run with more models is never read from a stale copy).
    """
    root = Path(path) if not stage and Path(path).is_dir() else _stage(path, cfg.cache_dir / "alphafold" / "imports")
    label = label or root.name
    models = []
    pdbs = _find(root, "*_relaxed_rank_*.pdb") or _find(root, "*_unrelaxed_rank_*.pdb")
    for pdb in pdbs:
        m_rank = re.search(r"rank_(\d+)", pdb.name)
        rank = int(m_rank.group(1)) if m_rank else None
        mm = re.search(r"(model_\d+\S*?)_seed_(\d+)", pdb.name) or re.search(r"(model_\d+)", pdb.name)
        tag = mm.group(1) if mm else pdb.stem
        scores = _find(root, f"*_scores_rank_{rank:03d}_*.json") if rank is not None else []
        data = json.loads(scores[0].read_text()) if scores else {}
        mtype = re.search(r"(alphafold2(?:_multimer_v\d|_ptm)?)", pdb.name)
        m = AFModel(f"CF:{label}:rank_{rank:03d}" if rank is not None else f"CF:{label}:{pdb.stem}",
                    "ColabFold (AlphaFold2)", "monomer", read_model_atoms(pdb),
                    scores={k: data.get(k) for k in ("ptm", "iptm", "max_pae") if data.get(k) is not None},
                    meta={"Rank": rank, "Model": (f"{mtype.group(1)} " if mtype else "") + tag,
                          "File": str(pdb), "SHA-256": sha256_file(pdb),
                          "Relaxed": "_relaxed_" in pdb.name and "_unrelaxed_" not in pdb.name,
                          "Scores file": str(scores[0]) if scores else ""})
        _finish(m, useq, cfg.get("alphafold", {}), issues)
        _set_pae(m, _pae_array(data) if data else None, issues)
        if not scores:
            issues.add("alphafold", Status.WARNING, "no scores JSON for this rank: PAE / pTM unavailable",
                       pdb_id=m.model_id)
        models.append(m)
    if not models:
        issues.add("alphafold", Status.FAIL, f"no ColabFold rank_* PDB files in {path}")
    return models


def load_user_model(path: str | Path, useq: str, cfg: Any, issues: IssueLog, label: str | None = None,
                    plddt_in_b: bool = False, pae_json: str | Path | None = None) -> list[AFModel]:
    """Any PDB/mmCIF model; declare ``plddt_in_b`` if its B-factor column holds pLDDT."""
    staged = _stage(path, cfg.cache_dir / "alphafold" / "imports")
    m = AFModel(f"MODEL:{label or staged.stem}", "user model", "monomer", read_model_atoms(staged),
                plddt_in_b=plddt_in_b, meta={"File": str(staged), "SHA-256": sha256_file(staged)})
    _finish(m, useq, cfg.get("alphafold", {}), issues)
    if pae_json:
        _set_pae(m, _pae_array(json.loads(Path(pae_json).read_text())), issues)
    return [m]


def load_upload(item: dict, useq: str, cfg: Any, issues: IssueLog) -> list[AFModel]:
    """``{"kind": "colabfold" | "model", "path": ..., "label": ..., "plddt_in_b": bool}``."""
    kind = item.get("kind")
    if kind == "colabfold":
        return load_colabfold(item["path"], useq, cfg, issues, item.get("label"), bool(item.get("stage", True)))
    if kind == "model":
        return load_user_model(item["path"], useq, cfg, issues, item.get("label"),
                               bool(item.get("plddt_in_b", False)), item.get("pae_json"))
    issues.add("alphafold", Status.FAIL, f"unknown upload kind {kind!r} for {item.get('path')}")
    return []


# ---------------------------------------------------------------------------
# Regions and confidence tables
# ---------------------------------------------------------------------------
def plddt_band(v: float | None, bands: list[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    hi, mid, lo = bands
    return "very high" if v >= hi else "confident" if v >= mid else "low" if v >= lo else "very low"


def region_sets(cfg: Any, useq: str, sites: list[int], tm_residues: set[int] | None = None,
                canonical: set[int] | None = None) -> dict[str, set[int]]:
    """Named UniProt residue sets used by the confidence, PAE and RMSD tables."""
    out: dict[str, set[int]] = {}
    for name, (a, b) in (cfg.regions.get("domains") or {}).items():
        out[name] = set(range(int(a), int(b) + 1))
    for name, (a, b) in cfg.flexible_regions.items():
        out[name] = set(range(int(a), int(b) + 1))
    start = int(cfg.get("membrane.ectodomain_core_start", 57))
    out["ectodomain (core)"] = set(range(start, len(useq) + 1))
    if sites:
        out["active site"] = set(sites)
    zn = set()
    for ligs in ((cfg.active_site.get("metal_site") or {}).get("zn_ligands") or {}).values():
        zn |= {int(re.sub(r"\D", "", c)) for c in ligs}
    if zn:
        out["Zn ligands"] = zn
    seq_n = {i + 1 for i in range(len(useq) - 2) if useq[i] == "N" and useq[i + 1] != "P" and useq[i + 2] in "ST"}
    if seq_n:
        out["N-glycosylation sequons"] = seq_n
    if tm_residues:
        out["transmembrane (topology)"] = set(tm_residues)
    if canonical:
        out["canonical dimer interface"] = set(canonical)
    return out


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.mean(vals)), 2) if vals else None


def residue_confidence(models: list[AFModel], regions: dict[str, set[int]], bands: list[float],
                       coverage: dict[int, float] | None = None) -> pd.DataFrame:
    """One row per model, target chain and UniProt residue."""
    rows = []
    for m in models:
        for c in m.target_chains:
            pl = m.plddt(c)
            a = m.atoms
            ca = a[(a["chain"] == c) & (a["atom"] == "CA") & a["unp"].notna()]
            for u, rn in zip(ca["unp"].astype(int), ca["resname"]):
                v = pl.get(u)
                rows.append({"Model ID": m.model_id, "Chain": c, "UniProt number": u,
                             "Residue": THREE_TO_ONE.get(rn, "X"), "pLDDT": round(v, 2) if v is not None else None,
                             "Confidence band": plddt_band(v, bands),
                             "Regions": "; ".join(n for n, s in regions.items() if u in s),
                             "Fraction of experimental chains covering residue": (coverage or {}).get(u)})
    return pd.DataFrame(rows)


def region_confidence(models: list[AFModel], regions: dict[str, set[int]], confident: float) -> pd.DataFrame:
    rows = []
    for m in models:
        look = m.pae_lookup()
        for c in m.target_chains:
            pl = m.plddt(c)
            for name, res in regions.items():
                vals = [pl[u] for u in sorted(res) if u in pl]
                idx = [look[(c, u)] for u in sorted(res) if (c, u) in look]
                ipae = round(float(m.pae[np.ix_(idx, idx)].mean()), 2) if m.pae is not None and len(idx) > 1 else None
                rows.append({"Model ID": m.model_id, "Chain": c, "Region": name, "Residues in region": len(res),
                             "Residues modelled": len(vals) if pl else int(sum(u in m.ca(c) for u in res)),
                             "Mean pLDDT": _mean(vals), "Min pLDDT": round(min(vals), 2) if vals else None,
                             "Fraction pLDDT < confident": round(sum(v < confident for v in vals) / len(vals), 3) if vals else None,
                             "Intra-region PAE (Å)": ipae,
                             "Calculation status": (Status.OK if vals or not m.plddt_in_b else Status.NOT_APPLICABLE).value})
    return pd.DataFrame(rows)


def domain_pae(models: list[AFModel], domains: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Mean PAE between every pair of domains (rows = aligned on, columns = scored), per chain pair."""
    rows = []
    for m in models:
        if m.pae is None:
            continue
        look = m.pae_lookup()
        chains = m.target_chains
        for c1, c2 in itertools.product(chains, chains):
            for (n1, (a1, b1)), (n2, (a2, b2)) in itertools.product(domains.items(), domains.items()):
                i = [look[(c1, u)] for u in range(int(a1), int(b1) + 1) if (c1, u) in look]
                j = [look[(c2, u)] for u in range(int(a2), int(b2) + 1) if (c2, u) in look]
                if not i or not j:
                    continue
                rows.append({"Model ID": m.model_id, "Chain (aligned)": c1, "Chain (scored)": c2,
                             "Domain (aligned)": n1, "Domain (scored)": n2,
                             "Mean PAE (Å)": round(float(m.pae[np.ix_(i, j)].mean()), 2),
                             "Median PAE (Å)": round(float(np.median(m.pae[np.ix_(i, j)])), 2)})
    return pd.DataFrame(rows)


def experimental_coverage(chains: pd.DataFrame) -> dict[int, float]:
    """Fraction of target chains whose SIFTS UniProt range covers each residue."""
    from .conformations import construct_ranges
    ranges = list(construct_ranges(chains).values()) if len(chains) else []
    if not ranges:
        return {}
    n = len(ranges)
    top = max(b for _, b in ranges)
    return {u: round(sum(a <= u <= b for a, b in ranges) / n, 3) for u in range(1, top + 1)}


# ---------------------------------------------------------------------------
# Comparison with experimental structures
# ---------------------------------------------------------------------------
def reference_ids(tables: dict[str, pd.DataFrame], cfg: Any) -> list[str]:
    top_n = int(cfg.get("alphafold.compare.top_n", 5))
    rank = tables.get("TemplateRanking", pd.DataFrame())
    ids = rank["PDB ID"].astype(str).str.upper().tolist()[:top_n] if len(rank) else []
    if cfg.get("alphafold.compare.include_lid_calibration", True):
        ids += [p for p in (cfg.lid.get("known_states") or {}) if p not in ids]
    return ids


def experimental_references(cfg: Any, tables: dict[str, pd.DataFrame], sess: requests.Session,
                            issues: IssueLog, ids: list[str] | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    """Atom tables (target polymer, UniProt-numbered) of reference experimental chains."""
    from .pdbe import fetch_sifts, sifts_segments
    from .rcsb import load_structure
    from .structure_atoms import atoms_from_gemmi
    from .survey import http_kwargs

    json_kw, dl_kw = http_kwargs(cfg)
    chains = tables.get("Chains", pd.DataFrame())
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for pdb in ids if ids is not None else reference_ids(tables, cfg):
        try:
            st = load_structure(sess, pdb, cfg.cache_dir / "mmcif", cfg.endpoint("rcsb_cif"), **dl_kw)
            segs = sifts_segments(fetch_sifts(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_sifts"), **json_kw),
                                  pdb, cfg.accession)
            ents = {str(e) for e in chains[chains["PDB ID"] == pdb]["Entity"].dropna()} if len(chains) else set()
            atoms = atoms_from_gemmi(st, 0, segs, ents)
        except Exception as ex:  # noqa: BLE001
            issues.add("alphafold", Status.WARNING, f"reference structure unavailable: {ex}", pdb_id=pdb)
            continue
        t = atoms[atoms["target"] & atoms["polymer"] & atoms["unp"].notna()]
        for ch, g in t.groupby("chain"):
            out[(pdb, ch)] = g
    return out


def _ca_of(atoms: pd.DataFrame) -> dict[int, np.ndarray]:
    s = atoms[(atoms["atom"] == "CA") & atoms["unp"].notna()]
    return {int(u): np.array([x, y, z]) for u, x, y, z in zip(s["unp"], s["x"], s["y"], s["z"])}


def _sidechains(atoms: pd.DataFrame, residues: set[int]) -> dict[tuple[int, str], np.ndarray]:
    s = atoms[atoms["unp"].isin(residues) & ~atoms["atom"].isin(BACKBONE)]
    return {(int(u), a): np.array([x, y, z]) for u, a, x, y, z in zip(s["unp"], s["atom"], s["x"], s["y"], s["z"])}


def compare_to_reference(m: AFModel, chain: str, ref_key: tuple[str, str], ref_atoms: pd.DataFrame, cfg: Any,
                         regions: dict[str, set[int]]) -> tuple[dict, list[dict], dict[int, float]]:
    """Global, core, region and active-site comparison of one model chain with one experimental chain."""
    from scipy.stats import spearmanr

    from .conformations import kabsch, rmsd

    ccfg = cfg.get("alphafold.compare", {})
    conf = float(cfg.get("alphafold.confident_plddt", 70))
    mca, rca = m.ca(chain), _ca_of(ref_atoms)
    pl = m.plddt(chain)
    common = sorted(set(mca) & set(rca))
    row = {"Model ID": m.model_id, "Chain": chain, "Reference PDB ID": ref_key[0], "Reference chain": ref_key[1],
           "Common CA": len(common)}
    if len(common) < int(ccfg.get("min_common_ca", 50)):
        return {**row, "Calculation status": Status.NOT_RUN.value,
                "Warning": f"only {len(common)} common CA atoms"}, [], {}
    core = [u for u in common if pl.get(u, 100.0) >= conf] if pl else common
    warn = []
    if len(core) < int(ccfg.get("min_common_ca", 50)):
        warn.append(f"confident core has {len(core)} residues; fitted on all common residues")
        core = common
    P = np.array([mca[u] for u in core])
    Q = np.array([rca[u] for u in core])
    R, pm, qm = kabsch(P, Q)
    fit = lambda X: (X - pm) @ R.T + qm  # noqa: E731
    A = fit(np.array([mca[u] for u in common]))
    B = np.array([rca[u] for u in common])
    dev = np.linalg.norm(A - B, axis=1)
    devd = {u: float(d) for u, d in zip(common, dev)}
    row.update({"Confident-core residues": len(core),
                "Core CA RMSD (Å)": round(rmsd(fit(P), Q), 3),
                "All-common CA RMSD after core fit (Å)": round(float(np.sqrt(np.mean(dev ** 2))), 3)})
    for c in ccfg.get("gdt_cutoffs", [1, 2, 4, 8]):
        row[f"Fraction within {c:g} Å"] = round(float(np.mean(dev <= c)), 3)
    row["GDT-TS-like score"] = round(float(np.mean([np.mean(dev <= c) for c in (1, 2, 4, 8)])), 3)
    if pl:
        x = [pl[u] for u in common if u in pl]
        y = [devd[u] for u in common if u in pl]
        if len(set(x)) > 1:
            rho = spearmanr(x, y).correlation
            row["pLDDT–deviation Spearman"] = round(float(rho), 3)
    try:
        import tmtools
        seq_m = "".join(m.atoms[(m.atoms["chain"] == chain) & (m.atoms["atom"] == "CA") & m.atoms["unp"].isin(common)]
                        .sort_values("unp")["resname"].map(lambda r: THREE_TO_ONE.get(r, "X")))
        seq_r = "".join(ref_atoms[(ref_atoms["atom"] == "CA") & ref_atoms["unp"].isin(common)]
                        .drop_duplicates("unp").sort_values("unp")["resname"].map(lambda r: THREE_TO_ONE.get(r, "X")))
        res = tmtools.tm_align(np.array([mca[u] for u in common]), B, seq_m, seq_r)
        row["TM-score (reference-normalised)"] = round(float(res.tm_norm_chain2), 4)
    except ImportError:
        row["TM-score (reference-normalised)"] = None
        warn.append("TM-score NOT RUN (tmtools not installed)")
    except Exception as ex:  # noqa: BLE001
        row["TM-score (reference-normalised)"] = None
        warn.append(f"TM-score failed: {ex}")
    reg_rows = []
    for name, res_set in regions.items():
        us = [u for u in common if u in res_set]
        if len(us) < 3:
            continue
        d = np.array([devd[u] for u in us])
        reg_rows.append({"Model ID": m.model_id, "Chain": chain, "Reference PDB ID": ref_key[0],
                         "Reference chain": ref_key[1], "Region": name, "Common residues": len(us),
                         "CA RMSD after core fit (Å)": round(float(np.sqrt(np.mean(d ** 2))), 3),
                         "Max CA deviation (Å)": round(float(d.max()), 3),
                         "Mean pLDDT": _mean([pl.get(u) for u in us]) if pl else None})
    # active site: local superposition on site CA, then side-chain heavy atoms
    for label, key in (("Active-site", "active site"), ("Zn-ligand", "Zn ligands")):
        site = [u for u in common if u in regions.get(key, set())]
        if len(site) < 3:
            row[f"{label} side-chain RMSD (Å)"] = None
            continue
        Rs, pms, qms = kabsch(np.array([mca[u] for u in site]), np.array([rca[u] for u in site]))
        ms, rs = _sidechains(m.atoms[m.atoms["chain"] == chain], set(site)), _sidechains(ref_atoms, set(site))
        keys = sorted(set(ms) & set(rs))
        if keys:
            X = (np.array([ms[k] for k in keys]) - pms) @ Rs.T + qms
            Y = np.array([rs[k] for k in keys])
            row[f"{label} side-chain RMSD (Å)"] = round(rmsd(X, Y), 3)
            row[f"{label} side-chain atoms compared"] = len(keys)
    row["Calculation status"] = (Status.WARNING if warn else Status.OK).value
    row["Warning"] = "; ".join(warn)
    return row, reg_rows, devd


def lid_states(models: list[AFModel], refs: dict[tuple[str, str], pd.DataFrame], cfg: Any,
               reference: tuple[str, str] | None) -> pd.DataFrame:
    """Entrance-lid call for model chains using the Milestone 1 classifier and the reference chains."""
    from .conformations import lid_conformation

    if not cfg.lid:
        return pd.DataFrame()
    ca: dict = {k: {u: tuple(v) for u, v in _ca_of(a).items()} for k, a in refs.items()}
    for m in models:
        for c in m.target_chains:
            ca[(m.model_id, c)] = {u: tuple(v) for u, v in m.ca(c).items()}
    if reference is None or reference not in ca:
        reference = next(iter(refs), None)
    tab, _ = lid_conformation(ca, reference, cfg.lid)
    if not len(tab):
        return tab
    ids = {m.model_id for m in models}
    out = tab[tab["PDB ID"].isin(ids)].rename(columns={"PDB ID": "Model ID"})
    out.insert(2, "Lid reference", f"{reference[0]}/{reference[1]}" if reference else "")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multimers: interfaces and the confident-dimer rule
# ---------------------------------------------------------------------------
def confident_dimer_call(iptm: float | None, ipae: float | None, jac: float | None, dcfg: dict) -> tuple[str, str]:
    """``confident`` only if ipTM, interface PAE and canonical Jaccard all pass (see design §4)."""
    reasons = []
    hi, lo = float(dcfg.get("iptm_confident", 0.8)), float(dcfg.get("iptm_low", 0.6))
    p_hi, p_lo = float(dcfg.get("interface_pae_confident", 5.0)), float(dcfg.get("interface_pae_low", 10.0))
    j_min = float(dcfg.get("canonical_jaccard_min", 0.5))
    if (iptm is not None and iptm < lo) or (ipae is not None and ipae > p_lo):
        if iptm is not None and iptm < lo:
            reasons.append(f"ipTM {iptm:.2f} < {lo}")
        if ipae is not None and ipae > p_lo:
            reasons.append(f"interface PAE {ipae:.1f} > {p_lo} Å")
        return "low confidence", "; ".join(reasons)
    ok = True
    for name, v, good in (("ipTM", iptm, iptm is not None and iptm >= hi),
                          ("interface PAE", ipae, ipae is not None and ipae <= p_hi),
                          ("canonical Jaccard", jac, jac is not None and jac >= j_min)):
        if v is None:
            reasons.append(f"{name} not available (NOT RUN)")
            ok = False
        elif not good:
            reasons.append(f"{name} {v:.2f} outside confident range")
            ok = False
    return ("confident" if ok else "uncertain"), "; ".join(reasons) or "all three criteria met"


def _pair_iptm(m: AFModel, c1: str, c2: str) -> float | None:
    """AlphaFold2-multimer reports one ipTM for the complex (a homodimer has one chain pair)."""
    v = m.scores.get("iptm")
    return float(v) if v is not None else None


def model_interfaces(models: list[AFModel], cfg: Any, canonical: set[int] | None,
                     issues: IssueLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interfaces of multimer models (Milestone 2 interface code) with the confident-dimer call."""
    from .interfaces import analyse_interface, jaccard, residue_set
    from .structure_atoms import target_protomers

    icfg = dict(cfg.require("interfaces"))
    icfg["_glycans"] = cfg.ligand_classes["glycans"]
    dcfg = cfg.get("alphafold.dimer", {})
    rows, tabs = [], []
    for m in models:
        if m.kind != "multimer":
            continue
        asm = m.atoms
        prots = target_protomers(asm)
        look = m.pae_lookup()
        for p1, p2 in itertools.combinations(prots, 2):
            try:
                row, con, _ = analyse_interface(m.model_id, asm, p1, p2, icfg, {}, assembly="predicted model",
                                                source="predicted model")
            except Exception as ex:  # noqa: BLE001
                issues.add("alphafold", Status.FAIL, f"interface {p1}-{p2}: {ex}", pdb_id=m.model_id)
                continue
            row = {"Model ID": m.model_id, **{k: v for k, v in row.items() if k != "PDB ID"}}
            c1, c2 = row.get("Chain 1"), row.get("Chain 2")
            res = residue_set(row)
            jac = jaccard(res, canonical) if canonical else None
            row["Canonical interface Jaccard"] = round(jac, 3) if jac is not None else None
            row["Fraction of canonical interface recovered"] = (round(len(res & canonical) / len(canonical), 3)
                                                                if canonical else None)
            ipae = None
            if m.pae is not None and len(con):
                vals = []
                for a, b in zip(con["unp_a"], con["unp_b"]):
                    if pd.isna(a) or pd.isna(b):
                        continue
                    i, j = look.get((c1, int(a))), look.get((c2, int(b)))
                    if i is not None and j is not None:
                        vals += [m.pae[i, j], m.pae[j, i]]
                ipae = round(float(np.mean(vals)), 2) if vals else None
            row["Interface PAE (Å)"] = ipae
            iptm = _pair_iptm(m, c1, c2)
            row["ipTM"] = round(iptm, 3) if iptm is not None else None
            call, why = confident_dimer_call(row["ipTM"], ipae, row["Canonical interface Jaccard"], dcfg)
            row["Dimer confidence"] = call
            row["Dimer confidence reason"] = why
            rows.append(row)
            if len(con):
                tabs.append(con.assign(**{"Model ID": m.model_id, "Interface": f"{p1}-{p2}"}))
    return pd.DataFrame(rows), (pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame())


# ---------------------------------------------------------------------------
# Model summary, rankings and variant columns
# ---------------------------------------------------------------------------
def model_summary(models: list[AFModel], regions: dict[str, set[int]], cfg: Any, comp: pd.DataFrame,
                  ifaces: pd.DataFrame, lids: pd.DataFrame, checks: list[dict]) -> pd.DataFrame:
    bands = cfg.get("alphafold.plddt_bands", [90, 70, 50])
    rows = []
    for m in models:
        chains = m.target_chains
        pl = [v for c in chains for v in m.plddt(c).values()]
        per = {name: _mean([m.plddt(c).get(u) for c in chains for u in s]) for name, s in regions.items()}
        n = len(pl)
        row = {"Model ID": m.model_id, "Source": m.source, "Model kind": m.kind,
               "Version": m.meta.get("Version", ""), "Model rank": m.meta.get("Rank"), "Model": m.meta.get("Model"),
               "Target chains": ", ".join(chains),
               "Residues (target chains)": sum(ci["Mapped residues"] for ci in m.chain_info if ci["Target"]),
               "UniProt range": "; ".join(ci["UniProt range"] for ci in m.chain_info if ci["Target"]),
               "Identity to UniProt (min)": min((ci["Identity to UniProt"] for ci in m.chain_info if ci["Target"]), default=None),
               "Mean pLDDT": _mean(pl),
               f"pLDDT ≥ {bands[0]} (%)": round(100 * sum(v >= bands[0] for v in pl) / n, 1) if n else None,
               f"pLDDT {bands[1]}–{bands[0]} (%)": round(100 * sum(bands[1] <= v < bands[0] for v in pl) / n, 1) if n else None,
               f"pLDDT {bands[2]}–{bands[1]} (%)": round(100 * sum(bands[2] <= v < bands[1] for v in pl) / n, 1) if n else None,
               f"pLDDT < {bands[2]} (%)": round(100 * sum(v < bands[2] for v in pl) / n, 1) if n else None,
               "Mean pLDDT ectodomain": per.get("ectodomain (core)"),
               "Mean pLDDT TM": per.get("transmembrane (topology)"),
               "Mean pLDDT active site": per.get("active site"),
               "Mean pLDDT entrance lid": next((v for k, v in per.items() if "lid" in k), None),
               "pTM": m.scores.get("ptm"), "ipTM": m.scores.get("iptm"),
               "pLDDT in B-factor": m.plddt_in_b,
               "AFDB canonical sequence": m.meta.get("Canonical sequence") if m.source == "AlphaFold DB" else None,
               "PAE status": next((c["Status"] for c in checks if c["Model ID"] == m.model_id), ""),
               "File": m.meta.get("File"), "SHA-256": m.meta.get("SHA-256")}
        mc = comp[(comp["Model ID"] == m.model_id) & (comp["Calculation status"] != Status.NOT_RUN.value)] \
            if len(comp) else comp
        if len(mc):
            best = mc.iloc[0]
            row.update({"Best template": f"{best['Reference PDB ID']}/{best['Reference chain']}",
                        "Core CA RMSD to best template (Å)": best.get("Core CA RMSD (Å)"),
                        "All-common CA RMSD to best template (Å)": best.get("All-common CA RMSD after core fit (Å)"),
                        "TM-score to best template": best.get("TM-score (reference-normalised)"),
                        "GDT-TS-like score to best template": best.get("GDT-TS-like score"),
                        "Zn-ligand side-chain RMSD (Å)": best.get("Zn-ligand side-chain RMSD (Å)"),
                        "Active-site side-chain RMSD (Å)": best.get("Active-site side-chain RMSD (Å)"),
                        "pLDDT–deviation Spearman": best.get("pLDDT–deviation Spearman")})
        if len(lids):
            lm = lids[lids["Model ID"] == m.model_id]
            row["Entrance lid state"] = ", ".join(sorted(set(lm["Lid state"].astype(str)))) if len(lm) else None
        if len(ifaces):
            im = ifaces[ifaces["Model ID"] == m.model_id]
            if len(im):
                b = im.sort_values("Canonical interface Jaccard", ascending=False, na_position="last").iloc[0]
                row.update({"ipTM": b.get("ipTM") if b.get("ipTM") is not None else row["ipTM"],
                            "Interface PAE (Å)": b.get("Interface PAE (Å)"),
                            "Canonical interface Jaccard": b.get("Canonical interface Jaccard"),
                            "Dimer confidence": b.get("Dimer confidence")})
        problems = [ci["Warning"] for ci in m.chain_info if ci["Calculation status"] == Status.WARNING.value]
        if m.meta.get("PAE problem"):
            problems.append(f"PAE: {m.meta['PAE problem']}")
        row["Calculation status"] = (Status.FAIL if not chains else Status.WARNING if problems else Status.OK).value
        row["Warning"] = "; ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_models(summary: pd.DataFrame, cfg: Any, issues: IssueLog, top_n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AlphaFold rankings with the template-ranking engine (key = Model ID)."""
    from .ranking import rank_templates

    if not len(summary):
        return pd.DataFrame(), pd.DataFrame()
    crit = summary.set_index("Model ID")
    outs, tops = [], []
    for name, mode in (cfg.get("alphafold.ranking.modes") or {}).items():
        refd = set((mode.get("include") or {})) | set((mode.get("exclude") or {}))
        miss_filter = [c for c in refd if c not in crit]
        if miss_filter:
            issues.add("alphafold ranking", Status.NOT_RUN, f"mode {name}: columns not available: {', '.join(miss_filter)}")
            continue
        try:
            r = rank_templates(crit, mode, name)
        except KeyError as ex:
            issues.add("alphafold ranking", Status.FAIL, f"mode {name}: {ex}")
            continue
        r = r.rename(columns={"PDB ID": "Model ID"})
        keep = ["Model ID", "Rank", "Score (0-1, higher = better)", "Criteria used", "Score explanation"]
        r = r[[c for c in keep if c in r] + [c["column"] for c in mode.get("criteria", []) if c["column"] in r]]
        r.insert(0, "Ranking mode", name)
        r["Mode description"] = mode.get("description", "")
        outs.append(r)
        if len(r):
            t = r.head(top_n)[["Ranking mode", "Model ID", "Rank", "Score (0-1, higher = better)"]].copy()
            t["Models eligible"] = len(r)
            tops.append(t)
    return (pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(),
            pd.concat(tops, ignore_index=True) if tops else pd.DataFrame())


def variant_alphafold_columns(mapping: pd.DataFrame, res_conf: pd.DataFrame, model_id: str | None = None) -> pd.DataFrame:
    """Fill the Milestone 3 ``AlphaFold pLDDT`` / ``AlphaFold status`` columns of VariantStructureMapping."""
    out = mapping.copy()
    if not len(out) or not len(res_conf):
        return out
    rc = res_conf
    model_id = model_id or next((m for m in rc["Model ID"].unique() if str(m).startswith("AFDB:")), rc["Model ID"].iloc[0])
    rc = rc[rc["Model ID"] == model_id]
    first_chain = rc["Chain"].iloc[0] if len(rc) else None
    pl = rc[rc["Chain"] == first_chain].set_index("UniProt number")["pLDDT"]
    out["AlphaFold pLDDT"] = out["Position"].map(lambda p: pl.get(int(p)) if pd.notna(p) else None)
    out["AlphaFold status"] = [f"{Status.OK.value} ({model_id})" if pd.notna(v) else
                               f"{Status.NOT_APPLICABLE.value} (residue not in {model_id})" for v in out["AlphaFold pLDDT"]]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class AlphaFoldResults:
    tables: dict[str, pd.DataFrame]
    models: list[AFModel]
    issues: IssueLog
    input_files: list = field(default_factory=list)
    membrane: Any = None            # membrane.MembraneResults when the membrane stage ran


def canonical_from_assessment(assessment: dict[str, pd.DataFrame] | None) -> set[int] | None:
    if not assessment or "CanonicalInterface" not in assessment:
        return None
    can = assessment["CanonicalInterface"]
    v = can.set_index("Item")["Value"].get("canonical residues") if len(can) else None
    return {int(x) for x in re.findall(r"\d+", str(v))} if v and str(v) not in ("nan", "None") else None


def run_alphafold(cfg: Any, survey: dict[str, pd.DataFrame], assessment: dict[str, pd.DataFrame] | None = None,
                  uploads: list[dict] | None = None, sess: requests.Session | None = None,
                  issues: IssueLog | None = None, refs: dict | None = None, use_afdb: bool | None = None,
                  membrane: bool = True, uniprot: dict | None = None) -> AlphaFoldResults:
    """Milestone 4 stage: import models, confidence, comparison, interfaces, membrane context, rankings.

    Parameters
    ----------
    survey, assessment:
        Tables from notebooks 01 and 02 (the assessment workbook supplies the
        canonical interface; without it the dimer rule's third criterion is NOT RUN).
    uploads:
        ``[{"kind": "colabfold" | "model", "path": ..., "label": ...}]``; AlphaFold2 runs made
        in Colab by :func:`colabfold_run.run_colabfold_jobs` return this format.
    refs:
        Pre-loaded reference atom tables (tests); default loads them with gemmi.
    """
    from .cache import session_from_config
    from .sanity import run_alphafold_checks
    from .survey import http_kwargs
    from .uniprot import uniprot_entry, uniprot_info

    issues = issues if issues is not None else IssueLog()
    sess = sess or session_from_config(cfg)
    json_kw, dl_kw = http_kwargs(cfg)
    acc = cfg.accession
    entry = uniprot if uniprot is not None else uniprot_entry(sess, acc, cfg.cache_dir, cfg.endpoint("uniprot_json"), **json_kw)
    useq = uniprot_info(entry)[0]
    acfg = cfg.get("alphafold", {})

    models: list[AFModel] = []
    if acfg.get("use_afdb", True) if use_afdb is None else use_afdb:
        models += load_afdb(sess, acc, useq, cfg, issues, json_kw, dl_kw)
    for item in uploads or []:
        try:
            models += load_upload(item, useq, cfg, issues)
        except Exception as ex:  # noqa: BLE001
            issues.add("alphafold", Status.FAIL, f"import failed for {item.get('path')}: {ex}")
    models = [m for m in models if m.target_chains]

    # topology first: TM residues label the confidence tables and feed the membrane stage
    from .membrane import topology_table
    topo, topo_info = topology_table(cfg, entry, useq, issues, sess=sess)
    tm_res = set(topo.loc[topo["Membrane label"] == "M", "UniProt number"].astype(int)) if len(topo) else set()
    sites = sorted(int(n) for n in survey.get("ActiveSiteDefinition", pd.DataFrame()).query("Status == 'OK'")
                   ["UniProt number"].dropna().unique()) if "ActiveSiteDefinition" in survey else []
    canonical = canonical_from_assessment(assessment)
    if canonical is None:
        issues.add("alphafold", Status.NOT_RUN, "no canonical interface (assessment workbook not given): "
                   "the confident-dimer rule's canonical criterion is NOT RUN")
    regions = region_sets(cfg, useq, sites, tm_res, canonical)
    checks = [pae_index_check(m, float(acfg.get("pae_diagonal_max", 1.5))) for m in models]

    # experimental comparison
    if refs is None:
        refs = experimental_references(cfg, survey, sess, issues) if models else {}
    order = reference_ids(survey, cfg)
    ref_keys = sorted(refs, key=lambda k: (order.index(k[0]) if k[0] in order else 999, k[1]))
    comp_rows, reg_rows, dev_by = [], [], {}
    for m in models:
        for c in m.target_chains:
            for rk in ref_keys:
                row, rr, devd = compare_to_reference(m, c, rk, refs[rk], cfg, regions)
                comp_rows.append(row)
                reg_rows += rr
                if devd and (m.model_id, c) not in dev_by:
                    dev_by[(m.model_id, c)] = (rk, devd)
    comp = pd.DataFrame(comp_rows)
    lids = lid_states(models, refs, cfg, ref_keys[0] if ref_keys else None) if refs else pd.DataFrame()
    ifaces, icontacts = model_interfaces(models, cfg, canonical, issues)

    resconf = residue_confidence(models, regions, acfg.get("plddt_bands", [90, 70, 50]),
                                 experimental_coverage(survey.get("Chains", pd.DataFrame())))
    if len(resconf) and dev_by:
        resconf["Deviation from best template (Å)"] = [
            round(dev_by.get((mid, c), (None, {}))[1].get(int(u), np.nan), 3)
            for mid, c, u in zip(resconf["Model ID"], resconf["Chain"], resconf["UniProt number"])]
        resconf["Deviation reference"] = [f"{dev_by[(mid, c)][0][0]}/{dev_by[(mid, c)][0][1]}" if (mid, c) in dev_by else ""
                                          for mid, c in zip(resconf["Model ID"], resconf["Chain"])]
    summary = model_summary(models, regions, cfg, comp, ifaces, lids, checks)
    tables = {
        "AFModels": summary,
        "AFChains": pd.DataFrame([{"Model ID": m.model_id, **ci} for m in models for ci in m.chain_info]),
        "AFResidueConfidence": resconf,
        "AFRegionConfidence": region_confidence(models, regions, float(acfg.get("confident_plddt", 70))),
        "AFDomainPAE": domain_pae(models, {k: tuple(v) for k, v in (cfg.regions.get("domains") or {}).items()}),
        "AFvsExperimental": comp, "AFRegionRMSD": pd.DataFrame(reg_rows),
        "AFEntranceLid": lids, "AFInterfaces": ifaces, "AFInterfaceContacts": icontacts,
        "MembraneTopology": topo, "TopologyComparison": topo_info,
    }
    res = AlphaFoldResults(tables, models, issues,
                           [Path(m.meta["File"]) for m in models if m.meta.get("File")])
    if membrane:
        from .membrane import run_membrane
        res.membrane = run_membrane(cfg, res, survey, entry, useq, issues)
        tables.update(res.membrane.tables)
        mcols = res.membrane.model_columns
        if len(mcols):
            tables["AFModels"] = tables["AFModels"].merge(mcols, on="Model ID", how="left")
    rankings, top = rank_models(tables["AFModels"], cfg, issues)
    tables["AFRankings"], tables["AFRankingTop"] = rankings, top
    if res.membrane is not None:
        from .membrane import write_md_inputs
        res.membrane.md_files = write_md_inputs(cfg, res, top)
        tables.update({"MembraneMDReadiness": res.membrane.readiness_table(res, top, cfg)})
    tables["AlphaFoldChecks"] = run_alphafold_checks(tables, checks, cfg)
    return res


def save_alphafold(cfg: Any, res: AlphaFoldResults, tag: str = "") -> dict[str, Path]:
    """Write the AlphaFold/membrane workbook, CSV/Parquet tables and manifest."""
    from .excel_output import write_tables, write_workbook
    from .manifest import build_manifest, detect_software, manifest_table, write_manifest

    software = detect_software()
    sheets = {k: v for k, v in res.tables.items() if k != "AFInterfaceContacts" or len(v) < 200000}
    sheets["AlphaFoldIssues"] = res.issues.to_frame()
    sheets["Software"] = software
    manifest = build_manifest(cfg, structure_ids=[m.model_id for m in res.models], input_files=res.input_files,
                              software=software,
                              extra={"stage": "AlphaFold comparison and membrane context (notebook 03)",
                                     "assumptions": {"membrane.builder": cfg.get("membrane.builder"),
                                                     "lipid composition is a placeholder": True},
                                     "issue_counts": {s.value: res.issues.count(s) for s in Status}})
    sheets["Reproducibility"] = manifest_table(manifest)
    suffix = f"_{tag}" if tag else ""
    wb = cfg.output_file("alphafold_workbook")
    wb = wb.with_name(wb.stem + suffix + ".xlsx")
    write_workbook(wb, sheets)
    mpath = write_manifest(manifest, cfg.outputs_dir / f"{cfg.accession}_alphafold_manifest{suffix}.json")
    tdir = cfg.outputs_dir / f"alphafold_tables{suffix}"
    write_tables(tdir, sheets, csv=bool(cfg.get("outputs.write_csv", True)),
                 parquet=bool(cfg.get("outputs.write_parquet", True)))
    out = {"workbook": wb, "manifest": mpath, "tables_dir": tdir}
    if res.membrane is not None and res.membrane.md_files:
        out["md_inputs"] = res.membrane.md_files.get("folder")
    return out


def detect_upload_kind(path: str | Path) -> str:
    """Guess colabfold / model for an uploaded file or folder from its contents."""
    path = Path(path)
    names: list[str] = []
    if path.is_dir():
        names = [p.name for p in path.rglob("*")]
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [Path(n).name for n in z.namelist()]
    else:
        return "model"
    if any("_scores_rank_" in n for n in names):
        return "colabfold"
    return "model"
