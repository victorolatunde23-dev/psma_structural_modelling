"""Ligand and ion binding sites in the deposited model (asymmetric unit + lattice).

Milestone 3 adds interaction fingerprints, sub-pocket mapping and ligand
clustering to this module.

Contact rules
    * metals (``ligand_classes.metals``) use ``contacts.metal`` (first shell);
    * anions (``ligand_classes.anions``) use ``contacts.anion``;
    * everything else uses ``contacts.ligand``.
    Glycans are handled by :mod:`psma_pipeline.glycans`; additives are skipped
    unless they are metals.

Symmetry
    Contacts with crystal-lattice images are included and marked with ``*``.
    In this milestone a lattice contact may be biological (the dimer partner)
    or crystal packing; Milestone 2 separates the two by building the
    biological assembly.
"""

from __future__ import annotations

from .gemmi_utils import (author_id, mark_contact, neighbor_search, res_key, residue_label,
                          uniprot_mismatch, unp_maps)
from .ligands import ADDITIVE, GLYCAN, classify_component, contact_cutoff, is_metal, is_metal_or_anion
from .status import Status


def binding_site_analysis(pdb: str, ctx: dict, outliers: dict, classes: dict,
                          contacts_cfg: dict, useq: str) -> list[dict]:
    """One row per ligand / ion instance in the analysed model.

    Parameters
    ----------
    pdb:
        PDB ID.
    ctx:
        Per-entry context from :func:`psma_pipeline.survey.process_entry`
        (``coords``, ``segs``, ...).
    outliers:
        PDBe residue outliers keyed by (author chain, author number, icode).
    classes, contacts_cfg:
        ``config.ligand_classes`` and the ``contacts`` section.
    useq:
        Canonical UniProt sequence (to flag residues that disagree with it).

    Returns
    -------
    list of row dicts. ``Waters in contact`` counts unique water molecules;
    ``Ligand atom–water pairs`` keeps the original (pair) count.
    """
    import gemmi

    coords = ctx["coords"]
    st, model = coords["st"], coords["model"]
    ns = neighbor_search(st, model, float(contacts_cfg.get("neighbour_search_radius", 6.0)))
    l2u = unp_maps(ctx)
    tsubs = coords["target_subs"]
    rows: list[dict] = []

    for ch in model:
        for res in ch:
            if res.is_water() or res.entity_type == gemmi.EntityType.Polymer:
                continue
            comp = res.name.upper()
            cls = classify_component(comp, classes)
            if cls == GLYCAN:
                continue
            if cls == ADDITIVE and not is_metal(comp, classes):
                continue
            own = res_key(ch, res)
            cutoff = contact_cutoff(comp, classes, contacts_cfg)
            shell = is_metal_or_anion(comp, classes)
            contacts: dict[str, float] = {}
            other_het: dict[str, float] = {}
            coordination: list[tuple[str, float]] = []
            donors: list[tuple[str, float]] = []
            env_b: list[float] = []
            water_pairs, water_ids, sym_contact = 0, set(), False
            outlier_hits, mismatches = set(), set()
            for atom in res:
                for m in ns.find_atoms(atom.pos, "\0", radius=cutoff):
                    cra = m.to_cra(model)
                    key = res_key(cra.chain, cra.residue)
                    d, sym = mark_contact(st, atom.pos, m, cra)
                    if key == own and not sym:
                        continue
                    if d > cutoff:
                        continue
                    if cra.residue.is_water():
                        water_pairs += 1
                        water_ids.add((key, sym, round(d, 2) if sym else 0))
                        if shell:
                            coordination.append((f"HOH{cra.residue.seqid.num} O", d))
                            donors.append((f"HOH{cra.residue.seqid.num} O{_alt(cra.atom)}", d))
                        continue
                    lab = residue_label(cra.chain.name, cra.residue, l2u, tsubs, sym)
                    if cra.residue.entity_type == gemmi.EntityType.Polymer:
                        contacts[lab] = min(d, contacts.get(lab, 99))
                        env_b.append(cra.atom.b_iso)
                        sym_contact |= sym
                        ok = outliers.get((cra.chain.name, cra.residue.seqid.num,
                                           cra.residue.seqid.icode.strip()))
                        if ok:
                            outlier_hits.add(f"{lab}: {','.join(sorted(ok))}")
                        mm = uniprot_mismatch(cra.residue, l2u, tsubs, useq)
                        if mm:
                            mismatches.add(f"{mm}/{cra.chain.name}")
                    else:
                        other_het[lab] = min(d, other_het.get(lab, 99))
                    if shell:
                        coordination.append((f"{lab} {cra.atom.name}", d))
                        if cra.atom.element.name.upper() in DONOR_ELEMENTS:
                            donors.append((f"{lab} {cra.atom.name}{_alt(cra.atom)}", d))

            lig_b = [a.b_iso for a in res]
            occ = [a.occ for a in res]
            mean_lb = sum(lig_b) / len(lig_b)
            mean_eb = sum(env_b) / len(env_b) if env_b else None
            warn = []
            if mismatches:
                warn.append("contact residues differ from UniProt: " + ", ".join(sorted(mismatches)))
            if not contacts:
                warn.append("no protein residue within cut-off")
            rows.append({
                "PDB ID": pdb, "Model": coords.get("model_number", 1),
                "Component": comp, "Class": cls,
                "Chain": ch.name, "Chain (label)": res.subchain,
                "Entity": coords["ent_of_sub"].get(res.subchain),
                "Residue no.": author_id(res),
                "Atoms": len(res), "Cutoff (Å)": cutoff,
                "Mean occupancy": round(sum(occ) / len(occ), 2),
                "Ligand mean B": round(mean_lb, 1),
                "Contact-atom mean B": round(mean_eb, 1) if mean_eb else None,
                "B ratio ligand/environment": round(mean_lb / mean_eb, 2) if mean_eb else None,
                "N protein residues": len(contacts),
                "Protein residues (UniProt/chain)": ", ".join(
                    sorted(contacts, key=lambda s: (s.split("/")[1], s))),
                "Other ligands/ions in contact": ", ".join(sorted(other_het)),
                "Waters in contact": len(water_ids),
                "Ligand atom–water pairs": water_pairs,
                "Symmetry-mate contact": sym_contact,
                "Coordination number": len(coordination) if shell else None,
                "Coordination (atom: Å)": "; ".join(
                    f"{a}: {d:.2f}" for a, d in sorted(coordination, key=lambda t: t[1]))
                    if coordination else "",
                "Donor coordination number": len(donors) if shell else None,
                "Donor atoms (N/O/S: Å)": "; ".join(
                    f"{a}: {d:.2f}" for a, d in sorted(donors, key=lambda t: t[1])) if donors else "",
                "Validation outliers in site": "; ".join(sorted(outlier_hits)),
                "Residue/UniProt mismatches": ", ".join(sorted(mismatches)),
                "Calculation status": (Status.WARNING if warn else Status.OK).value,
                "Warning": "; ".join(warn),
            })
    return rows


DONOR_ELEMENTS = {"N", "O", "S"}


def _alt(atom) -> str:
    """`` (alt X)`` for atoms with an alternate location, else ``""``."""
    return f" (alt {atom.altloc})" if atom.altloc not in ("\0", "", " ") else ""


def coordination_distances(row: dict, column: str = "Coordination (atom: Å)") -> list[tuple[str, float]]:
    """Parse a coordination column back into ``[(partner, distance), ...]``.

    ``Coordination (atom: Å)`` lists every atom within the metal cut-off
    (including carbons such as Asp CG); ``Donor atoms (N/O/S: Å)`` lists only
    potential ligand atoms (N, O, S, waters) and is the one to use for
    coordination geometry.
    """
    out = []
    for part in str(row.get(column) or "").split(";"):
        if ":" in part:
            a, d = part.rsplit(":", 1)
            try:
                out.append((a.strip(), float(d)))
            except ValueError:
                continue
    return out

