"""Sanity checks on survey results and selection of the small validation set.

Every check yields rows with ``Status`` in PASS / WARNING / FAIL /
NOT APPLICABLE / NOT RUN. Checks named in ``sanity.critical_checks`` stop the
full run when any row FAILs (see :func:`critical_failures`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .assemblies import choose_assembly
from .config import Config
from .ligands import LIGAND, classify_component
from .pockets import coordination_distances
from .rcsb import entity_is_target
from .status import Status

log = logging.getLogger("psma_pipeline.sanity")


def _row(check: str, status: Status, detail: str, pdb: str | None = None,
         chain: str | None = None) -> dict:
    return {"Check": check, "Status": status.value, "PDB ID": pdb, "Chain": chain, "Detail": detail}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_assembly_copies(summary: pd.DataFrame) -> list[dict]:
    """gemmi assembly copies must equal RCSB's chain count for homomeric entries."""
    out = []
    for _, r in summary.iterrows():
        src, n_ent = r.get("Assembly copies source"), r.get("Polymer entities")
        if src != "gemmi generators":
            out.append(_row("assembly_copies_vs_rcsb", Status.NOT_RUN,
                            "assembly not expanded with gemmi", r["PDB ID"]))
        elif n_ent != 1:
            out.append(_row("assembly_copies_vs_rcsb", Status.NOT_APPLICABLE,
                            f"{n_ent} polymer entities (heteromeric)", r["PDB ID"]))
        elif r.get("Target copies in assembly") == r.get("RCSB assembly chain count"):
            out.append(_row("assembly_copies_vs_rcsb", Status.PASS,
                            f"{r.get('Target copies in assembly')} copies", r["PDB ID"]))
        else:
            out.append(_row("assembly_copies_vs_rcsb", Status.FAIL,
                            f"gemmi {r.get('Target copies in assembly')} vs RCSB "
                            f"{r.get('RCSB assembly chain count')}", r["PDB ID"]))
    return out


def check_sifts_integrity(chains: pd.DataFrame, max_substitution_fraction: float = 0.10) -> list[dict]:
    """Numbering integrity of every target chain.

    * FAIL – more than ``max_substitution_fraction`` of the SIFTS-mapped
      residues differ from UniProt: the mapping is probably offset, so every
      UniProt number for this chain would be wrong.
    * WARNING – no SIFTS mapping (UniProt numbering unavailable, nothing is
      mis-numbered) or an inconsistent segment (excluded from numbering).
    """
    out = []
    for _, r in chains.iterrows():
        rng, gaps = str(r.get("UniProt range (SIFTS)")), str(r.get("Internal gaps / insertions") or "")
        subs = [x for x in str(r.get("Substitutions vs UniProt") or "").split(", ") if x]
        mapped = None
        if rng != "no SIFTS mapping":
            mapped = sum(int(b) - int(a) + 1 for a, b in (p.split("-") for p in rng.split("; ") if "-" in p))
        if rng == "no SIFTS mapping":
            out.append(_row("sifts_segment_integrity", Status.WARNING,
                            "target chain without SIFTS mapping: UniProt numbering unavailable "
                            "(UniProt-based columns left empty)", r["PDB ID"], r["Chain (auth)"]))
        elif mapped and len(subs) / mapped > max_substitution_fraction:
            out.append(_row("sifts_segment_integrity", Status.FAIL,
                            f"{len(subs)} of {mapped} mapped residues differ from UniProt: "
                            "numbering offset suspected", r["PDB ID"], r["Chain (auth)"]))
        elif "length mismatch" in gaps:
            out.append(_row("sifts_segment_integrity", Status.WARNING,
                            "inconsistent SIFTS segment excluded from numbering: " + gaps,
                            r["PDB ID"], r["Chain (auth)"]))
        else:
            out.append(_row("sifts_segment_integrity", Status.PASS, rng, r["PDB ID"], r["Chain (auth)"]))
    return out


def check_active_site_sequence(active: pd.DataFrame, chains: pd.DataFrame) -> list[dict]:
    """Modelled active-site residues must match UniProt, unless the difference
    is a documented construct substitution (then WARNING; excluded from summaries).
    An unexplained mismatch indicates a numbering error (FAIL)."""
    out = []
    if not len(active) or "Matches UniProt" not in active:
        return [_row("active_site_sequence_match", Status.NOT_RUN, "no active-site rows")]
    subs = {}
    if len(chains):
        for _, c in chains.iterrows():
            subs[(c["PDB ID"], c["Chain (auth)"])] = str(c.get("Substitutions vs UniProt") or "")
    mod = active[active["Modelled"] == True]  # noqa: E712
    for (pdb, chain), g in mod.groupby(["PDB ID", "Chain"]):
        bad = g[g["Matches UniProt"] == False]  # noqa: E712
        if not len(bad):
            out.append(_row("active_site_sequence_match", Status.PASS, f"{len(g)} residues match",
                            pdb, chain))
            continue
        documented = subs.get((pdb, chain), "")
        for _, b in bad.iterrows():
            n = b["UniProt residue"][1:]
            if any(s[1:-1] == n for s in documented.split(", ") if s):
                out.append(_row("active_site_sequence_match", Status.WARNING,
                                f"{b['UniProt residue']}: engineered substitution ({documented}); "
                                "excluded from active-site summaries", pdb, chain))
            else:
                out.append(_row("active_site_sequence_match", Status.FAIL,
                                f"{b['UniProt residue']} modelled as {b.get('Residue in model')} without a "
                                "documented substitution: possible numbering error", pdb, chain))
    return out


def check_lid_calibration(lid: pd.DataFrame, known_states: dict) -> list[dict]:
    """Calibration structures must receive their literature lid state.

    FAIL when the geometric call is the opposite state; WARNING when the lid
    is not modelled (the structure cannot calibrate anything, but the
    classifier is not wrong).
    """
    if not known_states:
        return [_row("lid_calibration", Status.NOT_APPLICABLE, "no calibration structures configured")]
    out = []
    for pdb, state in known_states.items():
        g = lid[lid["PDB ID"] == pdb] if len(lid) else lid
        if not len(g):
            out.append(_row("lid_calibration", Status.NOT_RUN, "calibration structure not analysed", pdb))
            continue
        for _, r in g.iterrows():
            call = r.get("Lid state")
            if call == state:
                st, msg = Status.PASS, f"literature {state}, geometric call {call}"
            elif call in ("open", "closed"):
                st, msg = Status.FAIL, f"literature {state}, geometric call {call}: classifier disagrees"
            else:
                st, msg = Status.WARNING, (f"literature {state}, but lid is '{call}': unusable as a "
                                           "calibration structure; replace it in psma_regions.yaml")
            out.append(_row("lid_calibration", st, msg, pdb, r.get("Chain")))
    return out


def check_open_lid_modelled(lid: pd.DataFrame, min_modelled: int) -> list[dict]:
    """No chain may be called 'open' with fewer than ``min_modelled`` lid residues."""
    if not len(lid):
        return [_row("open_lid_requires_modelled_lid", Status.NOT_RUN, "no lid table")]
    bad = []
    for _, r in lid[lid["Lid state"] == "open"].iterrows():
        n = int(str(r.get("Lid residues modelled", "0/0")).split("/")[0])
        if n < min_modelled:
            bad.append(f"{r['PDB ID']}/{r['Chain']} ({n} modelled)")
    if bad:
        return [_row("open_lid_requires_modelled_lid", Status.FAIL, ", ".join(bad))]
    return [_row("open_lid_requires_modelled_lid", Status.PASS, "all open calls have a modelled lid")]


def check_zn_count(summary: pd.DataFrame, expected_per_protomer: int, ion_label: str = "Zn") -> list[dict]:
    """Zn per protomer in the assembly should be 0 (apo/metal-free) or the expected value."""
    out = []
    col = f"{ion_label} assembly"
    if col not in summary:
        return [_row("zn_count", Status.NOT_RUN, f"no '{col}' column")]
    for _, r in summary.iterrows():
        zn, cp = r.get(col), r.get("Target copies in assembly")
        if pd.isna(zn) or pd.isna(cp) or not cp:
            out.append(_row("zn_count", Status.NOT_RUN, "assembly Zn or copies unavailable", r["PDB ID"]))
            continue
        per = zn / cp
        st = Status.PASS if per in (0, expected_per_protomer) else Status.WARNING
        out.append(_row("zn_count", st, f"{zn:g} Zn / {cp:g} protomers = {per:g} per protomer "
                        f"(expected 0 or {expected_per_protomer})", r["PDB ID"]))
    return out


def check_zn_distances(binding: pd.DataFrame, rng: tuple[float, float], comp: str = "ZN") -> list[dict]:
    """Zn–donor distances (N/O/S atoms of protein, ligands and waters) within the configured range.

    Carbon atoms within the metal cut-off (e.g. Asp CG of a bidentate
    carboxylate) are not ligands and are ignored. Donors between the upper
    bound and the metal cut-off (typically the second oxygen of a bidentate
    carboxylate, such as D453 in PSMA) are listed as long contacts (PASS).
    Distances below the lower bound are physically implausible (WARNING).
    """
    if not len(binding):
        return [_row("zn_coordination_distances", Status.NOT_RUN, "no binding-site rows")]
    out = []
    col = "Donor atoms (N/O/S: Å)" if "Donor atoms (N/O/S: Å)" in binding else "Coordination (atom: Å)"
    for _, r in binding[binding["Component"] == comp].iterrows():
        d = coordination_distances(r, col)
        long_ = [f"{a} {x:.2f}" for a, x in d if x > rng[1]]          # up to the metal cut-off
        bad = [f"{a} {x:.2f}" for a, x in d if x < rng[0]]
        where = f"{r['Chain']} {r['Residue no.']}"
        if not d:
            out.append(_row("zn_coordination_distances", Status.WARNING,
                            f"{where}: no N/O/S donor within the cut-off", r["PDB ID"], r["Chain"]))
        elif bad:
            out.append(_row("zn_coordination_distances", Status.WARNING,
                            f"{where}: implausibly short (< {rng[0]} Å): " + "; ".join(bad), r["PDB ID"], r["Chain"]))
        else:
            note = (f"; long contacts {rng[1]}–cut-off (e.g. second O of a bidentate carboxylate): "
                    + "; ".join(long_)) if long_ else ""
            out.append(_row("zn_coordination_distances", Status.PASS,
                            f"{where}: {len(d) - len(long_)} donors in {rng[0]}–{rng[1]} Å{note}",
                            r["PDB ID"], r["Chain"]))
    return out or [_row("zn_coordination_distances", Status.NOT_APPLICABLE, "no Zn in analysed entries")]


def check_positive_values(summary: pd.DataFrame, binding: pd.DataFrame) -> list[dict]:
    """Resolutions and B-factor ratios must be positive when present."""
    out = []
    res = pd.to_numeric(summary.get("Resolution (Å)"), errors="coerce")
    bad = summary.loc[res <= 0, "PDB ID"].tolist() if res is not None else []
    out.append(_row("positive_resolution", Status.FAIL if bad else Status.PASS,
                    ", ".join(bad) or "all positive"))
    if len(binding) and "B ratio ligand/environment" in binding:
        br = pd.to_numeric(binding["B ratio ligand/environment"], errors="coerce")
        badb = binding.loc[br <= 0, "PDB ID"].unique().tolist()
        out.append(_row("positive_b_ratio", Status.FAIL if badb else Status.PASS,
                        ", ".join(badb) or "all positive"))
    return out


def check_target_chains(summary: pd.DataFrame) -> list[dict]:
    """Every processed entry must contain at least one target chain."""
    out = []
    for _, r in summary.iterrows():
        ok = bool(str(r.get("Target chains (auth)") or "").strip())
        out.append(_row("target_chain_present", Status.PASS if ok else Status.FAIL,
                        r.get("Target chains (auth)") or "no target chain", r["PDB ID"]))
    return out


def run_sanity_checks(tables: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Run every check and return one table (``SanityChecks`` sheet).

    Columns: Check, Critical, Status, PDB ID, Chain, Detail.
    """
    s = tables.get("Summary", pd.DataFrame())
    lid_cfg = cfg.lid
    rows: list[dict] = []
    rows += check_assembly_copies(s)
    rows += check_sifts_integrity(tables.get("Chains", pd.DataFrame()))
    rows += check_active_site_sequence(tables.get("ActiveSite", pd.DataFrame()),
                                       tables.get("Chains", pd.DataFrame()))
    rows += check_lid_calibration(tables.get("EntranceLid", pd.DataFrame()),
                                  lid_cfg.get("known_states", {}) if lid_cfg else {})
    rows += check_open_lid_modelled(tables.get("EntranceLid", pd.DataFrame()),
                                    int((lid_cfg or {}).get("classifier", {}).get("min_lid_modelled", 6)))
    rows += check_zn_count(s, int(cfg.active_site.get("metal_site", {}).get("expected_zn_per_protomer", 2)))
    rows += check_zn_distances(tables.get("BindingSites", pd.DataFrame()),
                               tuple(cfg.get("sanity.zn_ligand_distance_range", [1.8, 2.6])))
    rows += check_positive_values(s, tables.get("BindingSites", pd.DataFrame()))
    rows += check_target_chains(s)
    df = pd.DataFrame(rows)
    critical = set(cfg.get("sanity.critical_checks", []))
    df.insert(1, "Critical", df["Check"].isin(critical))
    return df


def critical_failures(sanity: pd.DataFrame) -> pd.DataFrame:
    """Rows of critical checks with Status FAIL (empty = safe to continue)."""
    if not len(sanity):
        return sanity
    return sanity[(sanity["Critical"]) & (sanity["Status"] == Status.FAIL.value)]


def sanity_overview(sanity: pd.DataFrame) -> pd.DataFrame:
    """Counts of each status per check."""
    if not len(sanity):
        return sanity
    return (sanity.pivot_table(index=["Check", "Critical"], columns="Status", values="Detail",
                               aggfunc="count", fill_value=0).reset_index())


# ---------------------------------------------------------------------------
# Validation-set selection
# ---------------------------------------------------------------------------
def entry_features(meta: dict, cfg: Config) -> pd.DataFrame:
    """Metadata-only features used to pick validation entries (no coordinates needed)."""
    acc, classes = cfg.accession, cfg.ligand_classes
    rows = []
    for pid, e in meta.items():
        targets = [p for p in e.get("polymer_entities") or [] if entity_is_target(p, acc)]
        asu = sum(len(p["rcsb_polymer_entity_container_identifiers"].get("asym_ids") or []) for p in targets)
        chosen, _ = choose_assembly(e.get("assemblies") or [],
                                    cfg.get("structure.assembly_preference", ["author", "software", "first"]))
        n_ent = len(e.get("polymer_entities") or [])
        asm = (chosen or {}).get("pdbx_struct_assembly", {}).get("oligomeric_count") if n_ent == 1 else None
        ligs = [n["pdbx_entity_nonpoly"]["comp_id"] for n in e.get("nonpolymer_entities") or []
                if classify_component(n["pdbx_entity_nonpoly"]["comp_id"], classes) == LIGAND]
        rows.append({
            "PDB ID": pid, "Method": "; ".join(x["method"] for x in e.get("exptl") or []),
            "Resolution (Å)": ((e.get("rcsb_entry_info") or {}).get("resolution_combined") or [None])[0],
            "Target chains in ASU": asu, "Assembly chains (homomer)": asm, "Polymer entities": n_ent,
            "Ligands of interest": ", ".join(ligs),
        })
    return pd.DataFrame(rows)


def select_validation_set(meta: dict, cfg: Config,
                          previous_workbook: str | Path | None = None) -> pd.DataFrame:
    """Choose the small validation set (brief §20) from metadata.

    Categories: fixed IDs from ``validation_set.fixed_ids`` (calibration and
    previous top template); high-resolution dimer with two chains in the ASU;
    ASU monomer whose assembly is a dimer; ligand-bound; apo; cryo-EM; and a
    disordered-lid entry taken from ``previous_workbook``'s EntranceLid sheet
    or ``validation_set.extra_ids``. Categories that cannot be filled are
    returned with status NOT RUN rather than dropped.
    """
    f = entry_features(meta, cfg)
    chosen: list[dict] = []
    taken: set[str] = set()

    def add(cat: str, pid: str | None, why: str, status: Status = Status.OK) -> None:
        chosen.append({"Category": cat, "PDB ID": pid, "Reason": why, "Status": status.value})
        if pid:
            taken.add(pid)

    for pid in cfg.get("validation_set.fixed_ids", []) or []:
        pid = pid.upper()
        if pid in meta:
            add("fixed (config)", pid, "listed in validation_set.fixed_ids")
        else:
            add("fixed (config)", pid, "not found in RCSB results for this accession", Status.NOT_RUN)

    if len(f):
        xray = f[f["Method"].str.contains("X-RAY", na=False)].sort_values("Resolution (Å)")
        em = f[f["Method"].str.contains("ELECTRON MICROSCOPY", na=False)].sort_values("Resolution (Å)")
    else:
        xray = em = f

    def pick(cat: str, pool: pd.DataFrame, why: str) -> None:
        if not len(pool):
            add(cat, None, "no entry satisfies the criterion", Status.NOT_RUN)
            return
        fresh = pool[~pool["PDB ID"].isin(taken)]
        r = (fresh if len(fresh) else pool).iloc[0]
        add(cat, r["PDB ID"], f"{why}; {r['Method']} {r['Resolution (Å)']} Å")

    if len(f):
        pick("high-resolution dimer (2 chains in ASU)",
             xray[(xray["Target chains in ASU"] == 2) & (xray["Assembly chains (homomer)"] == 2)],
             "best-resolution X-ray entry with 2 target chains in ASU and assembly of 2")
        pick("ASU monomer, dimer assembly",
             xray[(xray["Target chains in ASU"] == 1) & (xray["Assembly chains (homomer)"] == 2)],
             "1 target chain in ASU, author assembly of 2")
        pick("ligand-bound", xray[xray["Ligands of interest"] != ""], "has a ligand of interest")
        pick("apo", xray[xray["Ligands of interest"] == ""], "no ligand of interest")
        pick("cryo-EM", em, "best-resolution cryo-EM entry")

    lid_pid, why = None, ""
    if previous_workbook and Path(previous_workbook).exists():
        try:
            lid = pd.read_excel(previous_workbook, sheet_name="EntranceLid")
            dis = lid[lid["Lid state"].astype(str).str.startswith("disordered")]
            dis = dis[~dis["PDB ID"].isin(taken)] if len(dis[~dis["PDB ID"].isin(taken)]) else dis
            if len(dis):
                lid_pid, why = str(dis["PDB ID"].iloc[0]), f"lid disordered in {Path(previous_workbook).name}"
        except Exception as ex:  # noqa: BLE001
            why = f"could not read previous workbook: {ex}"
    extra = [x.upper() for x in cfg.get("validation_set.extra_ids", []) or []]
    if lid_pid:
        add("lid disordered", lid_pid, why)
    elif not extra:
        add("lid disordered", None, why or "set validation_set.previous_workbook or extra_ids", Status.NOT_RUN)
    for pid in extra:
        add("extra (config)", pid, "listed in validation_set.extra_ids",
            Status.OK if pid in meta else Status.NOT_RUN)
    return pd.DataFrame(chosen)


def validation_ids(selection: pd.DataFrame, meta: dict) -> list[str]:
    """Unique, available PDB IDs from a validation-set selection (input order kept)."""
    out: list[str] = []
    for pid in selection["PDB ID"].dropna():
        if pid in meta and pid not in out:
            out.append(pid)
    return out


__all__: list[Any] = ["run_sanity_checks", "critical_failures", "sanity_overview",
                      "select_validation_set", "validation_ids", "entry_features"]
