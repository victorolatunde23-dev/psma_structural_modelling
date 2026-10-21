"""Sequence-variant mapping onto structure (Milestone 3).

Sources
    * UniProt ``Natural variant`` features (default) and ``Mutagenesis``
      features (experimental, kept separate by ``Source type``);
    * optionally the EBI Proteins API variation endpoint (gnomAD, ClinVar,
      COSMIC, … – ``variants.sources: [uniprot, ebi_proteins]``).

Only single-residue substitutions are mapped; others are listed with status
NOT APPLICABLE. Every variant's reference amino acid is checked against the
canonical sequence; mismatches are FAIL rows and are not mapped.

Structural context (one reference assembly, UniProt numbering)
    relative solvent accessibility (RSA, Tien et al. 2013 maximum ASA), atom
    contacts, normalised B, minimum distances to the active site, ligands,
    Zn and the partner protomer, CA RMSF across all superposed chains,
    sub-pockets, flexible regions, N-glycosylation sequons and disulfides.
    AlphaFold pLDDT / PAE columns are NOT RUN until Milestone 4.

**Interpretation is a hypothesis only.** Categories and the hypothesis text
describe *where* a substitution sits; they are not predictions of
pathogenicity, stability or activity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from .cache import cached_json
from .conformations import kabsch
from .interfaces import THREE_TO_ONE
from .sasa import atom_sasa, radii_for, resolve_method
from .status import Status

MAX_ASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225, "G": 104, "H": 224, "I": 197,
           "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174}
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
       "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V"}
HYDROPHOBIC = set("AVLIMFWC")
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def uniprot_variants(entry: dict) -> list[dict]:
    """Natural-variant and mutagenesis features of a UniProtKB JSON entry."""
    out = []
    for f in entry.get("features", []):
        t = f.get("type")
        if t not in ("Natural variant", "Mutagenesis"):
            continue
        loc = f.get("location", {})
        s, e = loc.get("start", {}).get("value"), loc.get("end", {}).get("value")
        alt = f.get("alternativeSequence") or {}
        ref = alt.get("originalSequence")
        alts = alt.get("alternativeSequences") or []
        desc = f.get("description") or ""
        ids = sorted({x.get("id") for x in (f.get("featureCrossReferences") or []) if x.get("id")})
        for a in alts or [None]:
            out.append({"Source": "UniProt", "Source type": "natural variant" if t == "Natural variant" else "mutagenesis",
                        "Variant ID": f.get("featureId") or "", "Start": s, "End": e,
                        "Reference aa": ref, "Variant aa": a, "Description": desc,
                        "Cross-references": ", ".join(ids + re.findall(r"dbSNP:(rs\d+)", desc)),
                        "Clinical significance": "", "Evidence count": len(f.get("evidences") or [])})
    return out


def ebi_variants(data: dict) -> list[dict]:
    """Missense features from the EBI Proteins API variation response."""
    out = []
    for f in (data or {}).get("features", []):
        if f.get("type") != "VARIANT":
            continue
        s, e = f.get("begin"), f.get("end")
        sig = sorted({c.get("type", "") for c in (f.get("clinicalSignificances") or [])} - {""})
        srcs = sorted({x.get("name", "") for x in (f.get("xrefs") or [])} - {""})
        ids = sorted({x.get("id", "") for x in (f.get("xrefs") or [])} - {""})
        out.append({"Source": "EBI Proteins API", "Source type": f.get("consequenceType") or f.get("sourceType") or "",
                    "Variant ID": ids[0] if ids else "", "Start": _int(s), "End": _int(e),
                    "Reference aa": f.get("wildType"), "Variant aa": f.get("mutatedType"),
                    "Description": "; ".join(srcs), "Cross-references": ", ".join(ids[:5]),
                    "Clinical significance": ", ".join(sig), "Evidence count": len(f.get("evidences") or [])})
    return out


def fetch_ebi_variants(sess: requests.Session, acc: str, cache: Path, url: str, **http: Any) -> list[dict]:
    return ebi_variants(cached_json(sess, url.format(acc=acc), cache / "variants" / f"ebi_{acc}.json",
                                    allow_404=True, **http))


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _one(aa: Any) -> str | None:
    if aa is None or (isinstance(aa, float) and np.isnan(aa)):
        return None
    aa = str(aa)
    if len(aa) == 1:
        return aa.upper()
    return AA3.get(aa.capitalize())


def validate_variants(rows: list[dict], seq: str) -> pd.DataFrame:
    """Variants table with ``Position``, ``Substitution`` and a reference-check status."""
    out = []
    for r in rows:
        r = dict(r)
        s, e = r.get("Start"), r.get("End")
        ref, alt = _one(r.get("Reference aa")), _one(r.get("Variant aa"))
        r["Position"] = s if s == e else None
        if s is None or s != e or ref is None or alt is None or alt == ref:
            r["Status"] = Status.NOT_APPLICABLE.value
            r["Status reason"] = "not a single-residue substitution"
            r["Substitution"] = f"{r.get('Reference aa')}{s}{'-' + str(e) if e != s else ''}{r.get('Variant aa')}"
        elif not 1 <= s <= len(seq):
            r["Status"], r["Status reason"] = Status.FAIL.value, f"position {s} outside sequence"
            r["Substitution"] = f"{ref}{s}{alt}"
        elif seq[s - 1] != ref:
            r["Status"] = Status.FAIL.value
            r["Status reason"] = f"reference {ref}{s} does not match canonical {seq[s - 1]}{s} (isoform/numbering?)"
            r["Substitution"] = f"{ref}{s}{alt}"
        else:
            r["Status"], r["Status reason"] = Status.OK.value, ""
            r["Substitution"] = f"{ref}{s}{alt}"
        out.append(r)
    df = pd.DataFrame(out)
    if len(df):
        df = (df.sort_values(["Position", "Substitution", "Source"], na_position="last")
                .drop_duplicates(["Substitution", "Source", "Source type", "Variant ID"]).reset_index(drop=True))
    return df


# ---------------------------------------------------------------------------
# Sequence features
# ---------------------------------------------------------------------------
def sequons(seq: str) -> list[int]:
    """Positions (1-based) of N in N-X-S/T (X ≠ P) motifs."""
    return [i + 1 for i in range(len(seq) - 2) if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST"]


def sequon_effect(seq: str, pos: int, alt: str) -> str:
    """'' / 'removes sequon at N<k>' / 'creates sequon at N<k>' for a substitution."""
    before = set(sequons(seq))
    mut = seq[:pos - 1] + alt + seq[pos:]
    after = set(sequons(mut))
    lost, gained = sorted(before - after), sorted(after - before)
    parts = [f"removes sequon N{k}" for k in lost] + [f"creates sequon N{k}" for k in gained]
    return "; ".join(parts)


def disulfides(entry: dict) -> list[tuple[int, int]]:
    out = []
    for f in entry.get("features", []):
        if f.get("type") == "Disulfide bond":
            loc = f.get("location", {})
            s, e = loc.get("start", {}).get("value"), loc.get("end", {}).get("value")
            if s and e:
                out.append((int(s), int(e)))
    return out


# ---------------------------------------------------------------------------
# Structural context
# ---------------------------------------------------------------------------
def ca_rmsf(ca: dict[tuple[str, str], dict[int, np.ndarray]], reference: tuple[str, str] | None = None,
            min_common: int = 50) -> tuple[dict[int, float], dict[int, int]]:
    """Per-residue CA RMSF (Å) over chains superposed on a reference (all common CA).

    Returns ``({unp: rmsf}, {unp: chains contributing})``. Residues seen in
    fewer than three chains are omitted.
    """
    if not ca:
        return {}, {}
    ref_key = reference if reference in ca else max(ca, key=lambda k: len(ca[k]))
    ref = ca[ref_key]
    pos: dict[int, list[np.ndarray]] = {}
    for key, d in ca.items():
        common = sorted(set(d) & set(ref))
        if len(common) < min_common:
            continue
        P = np.array([d[n] for n in common])
        Q = np.array([ref[n] for n in common])
        R, pm, qm = kabsch(P, Q)
        for n, v in d.items():
            pos.setdefault(n, []).append((np.asarray(v) - pm) @ R.T + qm)
    rmsf, count = {}, {}
    for n, lst in pos.items():
        if len(lst) >= 3:
            X = np.array(lst)
            rmsf[n] = round(float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())), 2)
            count[n] = len(lst)
    return rmsf, count


def residue_environment(asm: pd.DataFrame, protomer: str, site_numbers: list[int], ligand_classes: dict,
                        classify, sasa_cfg: dict, contact_max: float = 4.5) -> pd.DataFrame:
    """Per-residue context of one target protomer inside its assembly (index = UniProt number)."""
    prot = asm[asm["polymer"] & ~asm["water"]]
    me = prot[(prot["protomer"] == protomer) & prot["unp"].notna()]
    if me.empty:
        return pd.DataFrame()
    method = resolve_method(sasa_cfg.get("method", "auto"))
    allxyz = prot[["x", "y", "z"]].to_numpy()
    idx_me = np.flatnonzero((prot["protomer"] == protomer).to_numpy() & prot["unp"].notna().to_numpy())
    sasa = atom_sasa(allxyz, radii_for(prot["element"]), method, float(sasa_cfg.get("probe", 1.4)),
                     int(sasa_cfg.get("n_points", 100)), subset=idx_me if method != "freesasa" else None)
    if method == "freesasa":
        sasa = sasa[idx_me]
    me = prot.iloc[idx_me].assign(_sasa=sasa)
    b = pd.to_numeric(me["b"], errors="coerce")
    me = me.assign(_bz=(b - b.mean()) / (b.std() or 1))
    het = asm[~asm["polymer"] & ~asm["water"]]
    zn = het[het["resname"].str.upper() == "ZN"][["x", "y", "z"]].to_numpy()
    cls = het["resname"].map(classify)
    lig = het[cls == "ligand of interest"][["x", "y", "z"]].to_numpy()
    site = me[me["unp"].isin(site_numbers)][["x", "y", "z"]].to_numpy()
    partner = prot[(prot["protomer"] != protomer) & prot["target"]][["x", "y", "z"]].to_numpy()
    trees = {k: cKDTree(v) for k, v in (("zn", zn), ("lig", lig), ("site", site), ("partner", partner)) if len(v)}
    all_tree = cKDTree(allxyz)
    rows = {}
    for unp, g in me.groupby("unp"):
        xyz = g[["x", "y", "z"]].to_numpy()
        aa = THREE_TO_ONE.get(g["resname"].iloc[0], "X")
        row = {"Residue": f"{aa}{int(unp)}", "SASA (Å²)": round(float(g["_sasa"].sum()), 1),
               "RSA": round(float(g["_sasa"].sum()) / MAX_ASA[aa], 3) if aa in MAX_ASA else None,
               "Normalised B": round(float(g["_bz"].mean()), 2)}
        own = set(g.index)
        n_cont = 0
        for hits in all_tree.query_ball_point(xyz, contact_max):
            n_cont += sum(1 for h in hits if prot.index[h] not in own)
        row["Atom contacts (≤4.5 Å)"] = n_cont
        for k, col in (("site", "Distance to catalytic pocket (Å)"), ("lig", "Distance to ligand (Å)"),
                       ("zn", "Distance to Zn (Å)"), ("partner", "Distance to partner protomer (Å)")):
            row[col] = round(float(trees[k].query(xyz)[0].min()), 2) if k in trees else None
        if int(unp) in site_numbers:
            row["Distance to catalytic pocket (Å)"] = 0.0
        rows[int(unp)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Mapping, categories, hypotheses
# ---------------------------------------------------------------------------
def categorise(pos: int, ref: str, alt: str, env: dict | None, ctx: dict, cfg: dict) -> tuple[list[str], str]:
    """Structural categories and a hypothesis sentence (never a prediction)."""
    near = float(cfg.get("near_distance", 6.0))
    cats = []
    if pos in ctx["zn_ligands"]:
        cats.append("zinc ligand")
    if pos in ctx["site_numbers"]:
        cats.append("active site")
    sps = [n for n, s in ctx["subpockets"].items() if pos in s]
    if sps:
        cats.append("sub-pocket: " + ", ".join(sorted(sps)))
    if env:
        if env.get("Distance to ligand (Å)") is not None and env["Distance to ligand (Å)"] <= near and "active site" not in cats:
            cats.append("near ligand")
        if env.get("Distance to Zn (Å)") is not None and env["Distance to Zn (Å)"] <= near and "zinc ligand" not in cats:
            cats.append("near Zn")
        if env.get("Distance to partner protomer (Å)") is not None and env["Distance to partner protomer (Å)"] <= near:
            cats.append("dimer interface")
        rsa = env.get("RSA")
        if rsa is not None:
            if rsa <= float(cfg.get("buried_rsa_max", 0.2)):
                cats.append("buried")
            elif rsa >= float(cfg.get("exposed_rsa_min", 0.4)):
                cats.append("exposed surface")
            else:
                cats.append("intermediate exposure")
    else:
        cats.append("not modelled in reference structure")
    if ctx["rmsf"].get(pos) is not None and ctx["rmsf"][pos] >= float(cfg.get("high_variability_rmsf", 1.0)):
        cats.append("structurally variable")
    for name, (a, b) in ctx["regions"].items():
        if a <= pos <= b:
            cats.append(f"region: {name}")
    if ctx["sequon"]:
        cats.append("glycosylation sequon")
    if pos in ctx["disulfide_cys"]:
        cats.append("disulfide")
    hyp = []
    if "zinc ligand" in cats:
        hyp.append("loss of a zinc ligand could disrupt the binuclear site")
    elif "active site" in cats or sps:
        hyp.append("could alter substrate/inhibitor recognition")
    if "buried" in cats:
        if alt == "P" or ref == "G":
            hyp.append(f"{ref}→{alt} in a buried position could perturb local backbone geometry")
        elif (ref in HYDROPHOBIC) != (alt in HYDROPHOBIC) or CHARGE.get(ref, 0) != CHARGE.get(alt, 0):
            hyp.append("a polarity/charge change in the core could destabilise the fold")
        else:
            hyp.append("a buried substitution could affect packing")
    if "dimer interface" in cats:
        hyp.append("could affect dimerisation")
    if "disulfide" in cats:
        hyp.append("removes a disulfide-bonded cysteine")
    if ctx["sequon"]:
        hyp.append(ctx["sequon"])
    if not hyp:
        hyp.append("surface position away from functional sites: no structural mechanism suggested")
    return cats, "Hypothesis (not a prediction): " + "; ".join(hyp) + "."


def map_variants(variants: pd.DataFrame, env: pd.DataFrame, seq: str, ctx: dict, cfg: dict,
                 reference_label: str, modelled_fraction: dict[int, float]) -> pd.DataFrame:
    """VariantStructureMapping rows (validated single substitutions only)."""
    rows = []
    ok = variants[variants["Status"] == Status.OK.value] if len(variants) else variants
    for _, v in ok.iterrows():
        pos = int(v["Position"])
        ref, alt = _one(v["Reference aa"]), _one(v["Variant aa"])
        e = env.loc[pos].to_dict() if len(env) and pos in env.index else None
        c = dict(ctx)
        c["sequon"] = sequon_effect(seq, pos, alt)
        cats, hyp = categorise(pos, ref, alt, e, c, cfg)
        row = {"Substitution": v["Substitution"], "Position": pos, "Source": v["Source"],
               "Source type": v["Source type"], "Variant ID": v["Variant ID"],
               "Reference structure": reference_label,
               "Modelled in reference": e is not None,
               "Fraction of structures modelling residue": modelled_fraction.get(pos),
               **({k: e.get(k) for k in env.columns if k != "Residue"} if e else {}),
               "CA RMSF across structures (Å)": ctx["rmsf"].get(pos),
               "Sequon effect": c["sequon"], "Categories": "; ".join(cats), "Hypothesis": hyp,
               "AlphaFold pLDDT": None, "AlphaFold status": Status.NOT_RUN.value + " (Milestone 4)"}
        rows.append(row)
    return pd.DataFrame(rows)


def variant_summary(mapping: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """Counts per category and per source / status."""
    rows = []
    if len(variants):
        for (src, st), g in variants.groupby(["Source", "Status"]):
            rows.append({"Group": "status", "Item": f"{src}: {st}", "Count": len(g)})
    if len(mapping):
        cats = mapping["Categories"].str.split("; ").explode()
        cats = cats.str.replace(r"^(sub-pocket|region): .*", r"\1", regex=True)
        for cat, n in cats.value_counts().sort_index().items():
            rows.append({"Group": "category", "Item": cat, "Count": int(n),
                         "Examples": ", ".join(mapping.loc[mapping["Categories"].str.contains(
                             re.escape(cat)), "Substitution"].head(8))})
    return pd.DataFrame(rows)
