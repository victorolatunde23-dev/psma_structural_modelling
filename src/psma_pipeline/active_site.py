"""Active-site definition (UniProt features + configured literature residues)
and per-chain B-factor / occupancy analysis of those residues.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .config import parse_residue_code
from .gemmi_utils import author_id, one_letter, unp_maps
from .status import IssueLog, Status
from .uniprot import site_features


def active_site_residues(entry: dict, useq: str, site_cfg: dict,
                         issues: IssueLog | None = None
                         ) -> tuple[dict[int, str], pd.DataFrame]:
    """Union of UniProt site features and configured literature residues.

    Parameters
    ----------
    entry:
        UniProtKB JSON entry.
    useq:
        Canonical sequence of that entry.
    site_cfg:
        ``config.active_site`` (``uniprot_feature_types``, ``max_feature_span``,
        ``literature_sites``).

    Returns
    -------
    (sites, definition)
        ``sites``: ``{UniProt number: role text}``, sorted.
        ``definition``: one row per configured or UniProt residue with its
        source and ``Status`` (``OK`` or ``FAIL`` when dropped because the
        amino acid does not match the canonical sequence).
    """
    sites: dict[int, list[str]] = defaultdict(list)
    rows = []
    for f in site_features(entry, site_cfg.get("uniprot_feature_types", []),
                           int(site_cfg.get("max_feature_span", 4))):
        for n in range(f["start"], f["end"] + 1):
            sites[n].append(f"UniProt {f['type'].lower()}{': ' + f['label'] if f['label'] else ''}")
            rows.append({"UniProt residue": f"{useq[n - 1]}{n}" if n <= len(useq) else str(n),
                         "UniProt number": n, "Source": "UniProt feature",
                         "Role": f"{f['type']}: {f['label']}".rstrip(": "),
                         "Status": Status.OK.value, "Note": ""})
    for code, role in (site_cfg.get("literature_sites") or {}).items():
        aa, n = parse_residue_code(code)
        if n <= len(useq) and useq[n - 1] == aa:
            sites[n].append(f"literature: {role}")
            rows.append({"UniProt residue": code, "UniProt number": n, "Source": "literature (config)",
                         "Role": role, "Status": Status.OK.value, "Note": ""})
        else:
            found = f"{useq[n - 1]}{n}" if n <= len(useq) else "out of range"
            note = f"configured {code} but canonical sequence has {found}; dropped"
            rows.append({"UniProt residue": code, "UniProt number": n, "Source": "literature (config)",
                         "Role": role, "Status": Status.FAIL.value, "Note": note})
            if issues is not None:
                issues.add("active-site definition", Status.FAIL, note)
    out = {n: "; ".join(dict.fromkeys(v)) for n, v in sorted(sites.items())}
    return out, pd.DataFrame(rows)


def active_site_analysis(pdb: str, ctx: dict, sites: dict[int, str], useq: str,
                         outliers: dict, backbone: set[str]) -> list[dict]:
    """B-factors and occupancy of every active-site residue in every target chain.

    Normalised B = (residue mean B − chain mean B) / chain SD (population SD),
    computed within each chain so structures at different resolutions are
    comparable. Not meaningful for most cryo-EM models.

    Returns
    -------
    list of row dicts (one per site residue per target chain); residues not
    modelled get ``Modelled = False``.
    """
    coords = ctx["coords"]
    model = coords["model"]
    l2u = unp_maps(ctx)
    rows = []
    for sub in sorted(coords["target_subs"]):
        auth = coords["auth_of_sub"].get(sub)
        if auth is None:
            continue
        chain = model[auth]
        residues = [r for r in chain if r.subchain == sub and r.label_seq is not None]
        all_b = [a.b_iso for r in residues for a in r]
        if not all_b:
            continue
        mu = sum(all_b) / len(all_b)
        sd = (sum((b - mu) ** 2 for b in all_b) / len(all_b)) ** 0.5
        by_unp = {l2u[sub][r.label_seq]: r for r in residues if r.label_seq in l2u.get(sub, {})}
        for n, role in sites.items():
            r = by_unp.get(n)
            base = {"PDB ID": pdb, "Model": coords.get("model_number", 1), "Chain": auth,
                    "Chain (label)": sub, "Entity": coords["ent_of_sub"].get(sub),
                    "UniProt residue": f"{useq[n - 1]}{n}",
                    "Role": role, "Method": ctx["method"], "Resolution (Å)": ctx["resolution"]}
            if r is None:
                rows.append({**base, "Modelled": False, "Calculation status": Status.OK.value,
                             "Warning": "residue not modelled"})
                continue
            bs = [a.b_iso for a in r]
            sc = [a.b_iso for a in r if a.name not in backbone]
            occ = [a.occ for a in r]
            mb = sum(bs) / len(bs)
            match = one_letter(r.name) == useq[n - 1]
            rows.append({
                **base, "Modelled": True,
                "Residue in model": f"{r.name}{r.seqid.num}",
                "Author residue number": author_id(r),
                "Label seq": r.label_seq,
                "Matches UniProt": match,
                "Atoms": len(r),
                "Mean B": round(mb, 1),
                "Side-chain mean B": round(sum(sc) / len(sc), 1) if sc else None,
                "Chain mean B": round(mu, 1),
                "Normalised B (z)": round((mb - mu) / sd, 2) if sd > 0 else None,
                "Min occupancy": round(min(occ), 2),
                "Mean occupancy": round(sum(occ) / len(occ), 2),
                "Alternate conformations": any(a.altloc != "\0" for a in r),
                "Validation outliers": ",".join(sorted(
                    outliers.get((auth, r.seqid.num, r.seqid.icode.strip()), []))),
                "Calculation status": Status.OK.value if match else Status.WARNING.value,
                "Warning": "" if match else
                f"model has {one_letter(r.name)} at UniProt {n}, canonical {useq[n - 1]}; "
                "excluded from active-site summaries",
            })
    return rows
