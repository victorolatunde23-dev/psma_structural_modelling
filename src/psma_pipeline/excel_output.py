"""Workbook, CSV and Parquet output; data dictionary; workbook comparison.

Formatting rules applied to every sheet:

* bold, wrapped, coloured header row; header frozen; auto-filter;
* readable column widths;
* status cells coloured (PASS/OK green, WARNING amber, FAIL red,
  NOT RUN / NOT APPLICABLE grey);
* missing values are written as empty cells (documented in the dictionary).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("psma_pipeline.excel_output")

MISSING_CONVENTION = ("Empty cell = value not available or not applicable. Check the row's "
                      "'Calculation status' / 'Warning' columns (or RunIssues) for the reason.")

# (sheet regex, column regex, description, units, numbering scheme)
_DICT: list[tuple[str, str, str, str, str]] = [
    # ---- identifiers used on many sheets --------------------------------------
    (".*", r"PDB ID", "PDB entry identifier", "", ""),
    (".*", r"Model", "Model number in the mmCIF file (1 = first model)", "", ""),
    (".*", r"Model analysed", "Model number analysed for this entry", "", ""),
    (".*", r"Entity", "mmCIF entity ID", "", ""),
    (".*", r"Chain \(auth\)|Chain", "Author chain ID (auth_asym_id)", "", "author"),
    (".*", r"Chain \(label\)|Chains \(label\)", "mmCIF label_asym_id(s)", "", "label"),
    (".*", r"Chains \(auth\)", "Author chain IDs of the component", "", "author"),
    (".*", r"Calculation status", "OK / WARNING / FAIL / NOT RUN / NOT APPLICABLE for this row", "", ""),
    (".*", r"Warnings?", "Problems recorded for this row (empty = none)", "", ""),
    (".*", r"Status", "PASS / OK / WARNING / FAIL / NOT RUN / NOT APPLICABLE", "", ""),
    (".*", r"Method", "Experimental method(s) reported by RCSB", "", ""),
    (".*", r"Resolution \(Å\)", "Reported resolution", "Å", ""),
    (".*", r"Item", "Parameter or manifest key", "", ""),
    (".*", r"Value", "Value of the parameter / manifest key", "", ""),
    # ---- Summary -----------------------------------------------------------------
    ("Summary", r"Title", "Entry title", "", ""),
    ("Summary", r"Release date", "Initial release date", "YYYY-MM-DD", ""),
    ("Summary", r"R-free", "R-free reported by RCSB (X-ray only)", "fraction", ""),
    ("Summary", r"Space group", "Crystal space group (H-M symbol)", "", ""),
    ("Summary", r"Target chains \(auth\)", "Author chains of the target protein", "", "author"),
    ("Summary", r"Construct.*", "Per entity: UniProt range, non-UniProt N/C extensions, substitutions", "", "UniProt"),
    ("Summary", r"Missing residues \(UniProt numbering\)", "Unmodelled target residues per chain", "", "UniProt"),
    ("Summary", r"Missing internal residues", "Unmodelled residues between the first and last modelled residue", "", "UniProt"),
    ("Summary", r"(\w+) ASU", "Count of this ion in the asymmetric unit (deposited model)", "count", ""),
    ("Summary", r"(\w+) assembly", "Count of this ion in the chosen biological assembly (generator expansion)", "count", ""),
    ("Summary", r"Ligands of interest", "Ligands not classified as ion/glycan/additive: COMP(count in ASU)", "", ""),
    ("Summary", r"Glycans", "Glycan components and oligosaccharides: COMP(count in ASU)", "", ""),
    ("Summary", r"Glycan chains ASU", "Branched oligosaccharide chains in the ASU", "count", ""),
    ("Summary", r"Glycan chains assembly", "Branched oligosaccharide chains in the assembly", "count", ""),
    ("Summary", r"Glycan residues assembly.*", "Single sugar residues (non-branched entities) in the assembly", "count", ""),
    ("Summary", r"Additives / buffer", "Crystallisation / buffer components: COMP(count)", "", ""),
    ("Summary", r"Other polymer entities", "Non-target polymer entities (e.g. antibody chains)", "", ""),
    ("Summary", r"Assembly used", "Assembly ID analysed (and its RCSB 'details')", "", ""),
    ("Summary", r"Assembly choice reason", "Why this assembly was chosen (author → software → first)", "", ""),
    ("Summary", r"Assembly copies source", "How 'Target copies in assembly' was obtained", "", ""),
    ("Summary", r"RCSB assembly chain count", "Polymer chains in the chosen assembly according to RCSB", "count", ""),
    ("Summary", r"Polymer entities", "Number of polymer entities in the entry", "count", ""),
    ("Summary", r"RCSB oligomeric state", "RCSB global-symmetry oligomeric state of the chosen assembly", "", ""),
    ("Summary", r"Target copies in ASU", "Target chains in the asymmetric unit", "count", ""),
    ("Summary", r"Target copies in assembly", "Target copies in the chosen biological assembly", "count", ""),
    ("Summary", r"Monomer / dimer", "Oligomeric state of the target in the chosen assembly", "", ""),
    ("Summary", r"Assembly note", "Relation between ASU and assembly content", "", ""),
    ("Summary|TemplateRanking", r"Clashscore", "MolProbity clashscore (wwPDB validation; RCSB, PDBe fallback)", "clashes / 1000 atoms", ""),
    ("Summary|TemplateRanking", r"Ramachandran outliers \(%\)", "Ramachandran outliers (wwPDB validation)", "%", ""),
    ("Summary", r"Rotamer outliers \(%\)", "Side-chain rotamer outliers (wwPDB validation)", "%", ""),
    ("Summary", r"RSRZ outliers \(%\)", "Residues with RSRZ > 2 (X-ray only)", "%", ""),
    ("Summary", r"Clashscore percentile \(similar resolution\)", "PDBe percentile vs entries of similar resolution (higher = better)", "percentile", ""),
    ("Summary", r"Active-site residues modelled \(best chain\)", "Fraction of active-site residues modelled in the best chain", "fraction", ""),
    ("Summary", r"Active-site mean normalised B", "Mean within-chain B z-score of active-site residues (mutated residues excluded)", "z", ""),
    ("Summary", r"Active-site partial occupancy", "Any active-site atom with occupancy < 1", "", ""),
    ("Summary", r"Active-site residues differing from UniProt", "Active-site residues whose model amino acid differs from UniProt", "", "UniProt"),
    ("Summary", r"Glycosylated Asn \(UniProt\)", "Asn residues carrying an observed N-glycan", "", "UniProt"),
    ("Summary", r"Max ligand B ratio", "Highest ligand/environment B ratio among ligands of interest (indirect indicator, not map fit)", "ratio", ""),
    ("Summary", r"Entrance lid state", "Entry-level lid call (mixed = chains disagree)", "", ""),
    ("Summary", r"Entrance lid confidence", "Confidence of the lid call", "", ""),
    ("Summary", r"Arene-site contact.*", "A ligand of interest contacts an arene-site residue", "", "UniProt"),
    ("Summary", r"Largest ligand \(atoms\)", "Heavy atoms modelled in the largest ligand of interest", "count", ""),
    # ---- TemplateRanking ------------------------------------------------------------
    ("TemplateRanking", r"Rank", "Position within the ranking mode (1 = best)", "", ""),
    ("TemplateRanking", r"Score.*", "1 − weighted mean of scaled criterion ranks", "0–1", ""),
    ("TemplateRanking", r"Ranking mode", "Configured ranking mode", "", ""),
    ("TemplateRanking", r"component: .*", "Scaled rank for this criterion (0 = best, 1 = worst)", "0–1", ""),
    ("TemplateRanking", r"Criteria used", "Criteria available for this entry, with weights", "", ""),
    ("TemplateRanking", r"Score explanation", "Raw value → scaled component for each criterion", "", ""),
    ("TemplateRanking", r"Best UniProt coverage", "Highest UniProt coverage among target chains", "fraction", ""),
    ("TemplateRanking", r"Fewest internal missing", "Fewest unmodelled internal residues among target chains", "count", ""),
    ("TemplateRanking", r"Active-site completeness", "Best per-chain fraction of active-site residues modelled", "fraction", ""),
    # ---- Chains ------------------------------------------------------------------------
    ("Chains", r"Description", "Entity description", "", ""),
    ("Chains", r"Entity length", "Residues in the entity sequence (incl. tags)", "count", "label_seq"),
    ("Chains", r"Observed residues", "Entity residues with coordinates", "count", ""),
    ("Chains", r"UniProt range \(SIFTS\)", "UniProt segments mapped by SIFTS", "", "UniProt"),
    ("Chains", r"UniProt coverage", "Fraction of the UniProt sequence covered by the construct", "fraction", ""),
    ("Chains", r"N-terminal extension.*", "Entity residues before the first UniProt-mapped residue (tags)", "sequence", ""),
    ("Chains", r"C-terminal extension.*", "Entity residues after the last UniProt-mapped residue", "sequence", ""),
    ("Chains", r"Substitutions vs UniProt", "Point differences, <UniProt aa><number><entity aa>", "", "UniProt"),
    ("Chains", r"Mutation \(author annotation\)", "Depositor's mutation annotation", "", ""),
    ("Chains", r"Internal gaps / insertions", "Deletions, insertions or inconsistent SIFTS segments", "", "UniProt"),
    ("Chains", r"Missing residues \(UniProt numbering\)", "Unmodelled UniProt-mapped residues", "", "UniProt"),
    ("Chains", r"N missing \(UniProt\)", "Number of unmodelled UniProt-mapped residues", "count", ""),
    ("Chains", r"Missing internal \(loop\) residues", "Unmodelled residues between first and last modelled residue", "", "UniProt"),
    ("Chains", r"N missing internal", "Number of internal unmodelled residues", "count", ""),
    ("Chains", r"Missing non-UniProt \(tag\) residues", "Unmodelled residues outside UniProt segments", "count", ""),
    ("Chains", r"Missing residues \(entity label_seq\)", "All unmodelled residues", "", "label_seq"),
    ("Chains", r"Entity sequence", "Canonical one-letter entity sequence", "", "label_seq"),
    # ---- Ligands -------------------------------------------------------------------------
    ("Ligands", r"Component", "PDB chemical component ID", "", ""),
    ("Ligands", r"Name", "Component name", "", ""),
    ("Ligands|BindingSites|LigandModelFit", r"Class", "ion of interest / glycan / additive / buffer / ligand of interest", "", ""),
    ("Ligands", r"Count in ASU", "Instances in the asymmetric unit", "count", ""),
    ("Ligands", r"Count in assembly", "Instances in the chosen assembly", "count", ""),
    ("Ligands", r"Assembly", "Assembly ID used for the assembly count", "", ""),
    # ---- BindingSites ------------------------------------------------------------------
    ("BindingSites", r"Component", "PDB chemical component ID", "", ""),
    ("BindingSites", r"Residue no\.", "Author residue number of the ligand/ion", "", "author"),
    ("BindingSites", r"Atoms", "Modelled atoms in the ligand/ion", "count", ""),
    ("BindingSites", r"Cutoff \(Å\)", "Contact cut-off used", "Å", ""),
    ("BindingSites", r"Mean occupancy", "Mean atomic occupancy of the ligand/ion", "fraction", ""),
    ("BindingSites", r"Ligand mean B", "Mean B-factor of the ligand/ion", "Å²", ""),
    ("BindingSites", r"Contact-atom mean B", "Mean B-factor of contacting protein atoms", "Å²", ""),
    ("BindingSites", r"B ratio ligand/environment", "Ligand mean B / contact-atom mean B (indirect indicator only)", "ratio", ""),
    ("BindingSites", r"N protein residues", "Protein residues within the cut-off", "count", ""),
    ("BindingSites", r"Protein residues \(UniProt/chain\)", "Contacting residues: <aa><UniProt no.>/<author chain>; * = symmetry image", "", "UniProt (target) / author (other)"),
    ("BindingSites", r"Other ligands/ions in contact", "Non-protein, non-water partners", "", "author"),
    ("BindingSites", r"Waters in contact", "Unique water molecules within the cut-off (M1 fix)", "count", ""),
    ("BindingSites", r"Ligand atom–water pairs", "Ligand atom–water pairs within the cut-off (original count)", "count", ""),
    ("BindingSites", r"Symmetry-mate contact", "Contact with a crystal-lattice image", "", ""),
    ("BindingSites", r"Coordination number", "All atoms within the metal/anion cut-off, incl. carbons (original definition)", "count", ""),
    ("BindingSites", r"Donor coordination number", "N/O/S atoms (incl. water O) within the metal/anion cut-off", "count", ""),
    ("BindingSites", r"Donor atoms \(N/O/S: Å\)", "Potential ligand atoms and distances; '(alt X)' marks alternate locations", "Å", "UniProt / author"),
    ("BindingSites", r"Coordination \(atom: Å\)", "Coordinating atoms and distances", "Å", "UniProt / author"),
    ("BindingSites", r"Validation outliers in site", "Contacting residues with wwPDB validation outliers", "", "UniProt"),
    ("BindingSites", r"Residue/UniProt mismatches", "Contacting target residues whose amino acid differs from UniProt", "", "UniProt"),
    # ---- Glycosylation ---------------------------------------------------------------------
    ("Glycosylation", r"Asn \(UniProt\)", "Glycosylated Asn", "", "UniProt"),
    ("Glycosylation", r"Asn \(author\)", "Glycosylated Asn", "", "author"),
    ("Glycosylation", r"Root sugar", "Sugar linked to Asn ND2: COMP + author number", "", "author"),
    ("Glycosylation", r"Root sugar chain \(label\)", "label_asym_id of the root sugar", "", "label"),
    ("Glycosylation", r"C1-ND2 \(Å\)", "Sugar C1 – Asn ND2 distance", "Å", ""),
    ("Glycosylation", r"Root sugar mean B", "Mean B-factor of the root sugar", "Å²", ""),
    # ---- ActiveSite ----------------------------------------------------------------------------
    ("ActiveSite|ActiveSiteDefinition", r"UniProt residue", "Canonical amino acid + UniProt number", "", "UniProt"),
    ("ActiveSiteDefinition", r"UniProt number", "UniProt residue number", "", "UniProt"),
    ("ActiveSiteDefinition", r"Source", "UniProt feature or literature (config)", "", ""),
    ("ActiveSiteDefinition", r"Note", "Why a configured residue was dropped", "", ""),
    ("ActiveSite|ActiveSiteDefinition", r"Role", "Functional role(s) of the residue", "", ""),
    ("ActiveSite", r"Modelled", "Residue has coordinates in this chain", "", ""),
    ("ActiveSite", r"Residue in model", "Residue name + author number in the model", "", "author"),
    ("ActiveSite", r"Author residue number", "auth_seq_id + insertion code", "", "author"),
    ("ActiveSite", r"Label seq", "label_seq_id", "", "label_seq"),
    ("ActiveSite", r"Matches UniProt", "Model amino acid equals canonical UniProt amino acid", "", ""),
    ("ActiveSite", r"Atoms", "Modelled atoms in the residue", "count", ""),
    ("ActiveSite", r"Mean B", "Mean B-factor of the residue", "Å²", ""),
    ("ActiveSite", r"Side-chain mean B", "Mean B-factor of side-chain atoms", "Å²", ""),
    ("ActiveSite", r"Chain mean B", "Mean B-factor of the whole chain", "Å²", ""),
    ("ActiveSite", r"Normalised B \(z\)", "(residue mean B − chain mean B) / chain SD", "z", ""),
    ("ActiveSite", r"Min occupancy", "Lowest atomic occupancy in the residue", "fraction", ""),
    ("ActiveSite", r"Mean occupancy", "Mean atomic occupancy", "fraction", ""),
    ("ActiveSite", r"Alternate conformations", "Residue has alternate locations", "", ""),
    ("ActiveSite", r"Validation outliers", "wwPDB outlier types for the residue", "", ""),
    # ---- Validation --------------------------------------------------------------------------
    ("Validation", r"RCSB clashscore", "Clashscore (RCSB validation summary)", "clashes / 1000 atoms", ""),
    ("Validation", r"RCSB rama_outliers_pct", "Ramachandran outliers (RCSB)", "%", ""),
    ("Validation", r"RCSB rotamer_outliers_pct", "Rotamer outliers (RCSB)", "%", ""),
    ("Validation", r"RCSB rsrz_outliers_pct", "RSRZ outliers (RCSB; X-ray only)", "%", ""),
    ("Validation", r"RCSB dcc_rfree", "R-free recomputed by DCC (RCSB)", "fraction", ""),
    ("Validation", r"PDBe \S+ percentile \(vs similar resolution\)", "PDBe percentile vs similar resolution (higher = better)", "percentile", ""),
    ("Validation", r"PDBe \S+ percentile \(whole PDB\)", "PDBe percentile vs whole PDB (higher = better)", "percentile", ""),
    ("Validation", r"PDBe \S+", "Raw metric value from PDBe", "as metric", ""),
    ("Validation", r"Residues with outliers \(PDBe\)", "Residues with any validation outlier (model 1)", "count", ""),
    # ---- Variability -------------------------------------------------------------------------
    ("Variability", r"Reference", "Reference chain for superposition (PDB/author chain)", "", ""),
    ("Variability", r"Common CA", "CA atoms shared with the reference", "count", "UniProt"),
    ("Variability", r"Global CA RMSD \(Å\)", "CA RMSD after global superposition", "Å", ""),
    ("Variability", r"Largest deviation", "UniProt residue with the largest deviation (distance)", "Å", "UniProt"),
    ("Variability", r".+ CA RMSD \(Å\)", "Region CA RMSD after the global fit", "Å", "UniProt"),
    ("Variability", r".+ residues compared", "Residues of the region present in both chains", "count", ""),
    # ---- EntranceLid ---------------------------------------------------------------------------
    ("EntranceLid", r"Lid state", "open / closed / disordered / not in construct / ambiguous / undetermined", "", ""),
    ("EntranceLid", r"Lid confidence", "high (all votes agree) / medium / low", "", ""),
    ("EntranceLid", r"Lid votes", "Majority votes / available votes", "", ""),
    ("EntranceLid", r"Lid residues modelled", "Modelled lid residues / lid length", "", "UniProt"),
    ("EntranceLid", r"Lid helicity \(CA i,i\+3\)", "Fraction of helical CA(i)–CA(i+3) distances in the lid", "fraction", ""),
    ("EntranceLid", r"Vote: .*", "Vote of one classifier signal", "", ""),
    ("EntranceLid", r"Lid PC1 score", "Projection of superposed lid coordinates on PC1", "Å", ""),
    ("EntranceLid", r"Cluster", "Two-means cluster name on PC1", "", ""),
    ("EntranceLid", r"Calibration label \(literature\)", "Literature lid state of a calibration structure", "", ""),
    ("EntranceLid", r"Lid RMSD to .*", "Lid CA RMSD after superposition on non-lid residues", "Å", ""),
    # ---- LidVsLigand / Assemblies / CrossCheck -------------------------------------------------
    ("LidVsLigand", r"Ligand size", "Largest ligand of interest, heavy-atom bins", "", ""),
    ("LidVsLigand", r"open|closed", "Number of entries with this lid state", "count", ""),
    ("Assemblies", r"Assembly", "Assembly ID", "", ""),
    ("Assemblies", r"Used in summary", "This assembly was chosen for the Summary", "", ""),
    ("Assemblies", r"Defined by", "RCSB 'details' (author_defined_assembly, software_defined_assembly, …)", "", ""),
    ("Assemblies", r"Method details", "Assembly method (e.g. PISA)", "", ""),
    ("Assemblies", r"Oligomeric details", "RCSB oligomeric description", "", ""),
    ("Assemblies", r"Total polymer chains", "Polymer chains in the assembly", "count", ""),
    ("Assemblies", r"Target copies \(gemmi\)", "Target copies from the assembly generators", "count", ""),
    ("Assemblies", r"Target state", "Oligomeric state of the target in this assembly", "", ""),
    ("Assemblies", r"Symmetry", "Global symmetry symbol", "", ""),
    ("Assemblies", r"Stoichiometry", "Global stoichiometry", "", ""),
    ("Assemblies", r"RCSB oligomeric state", "RCSB oligomeric state", "", ""),
    ("Assemblies", r"Selection reason", "Why this assembly was chosen (chosen row only)", "", ""),
    ("CrossCheck", r"In (RCSB|PDBe|UniProt)", "Entry present in this source", "", ""),
    ("CrossCheck", r"(RCSB|PDBe|UniProt) resolution", "Resolution reported by this source", "Å", ""),
    ("CrossCheck", r"(RCSB|PDBe|UniProt) method", "Method reported by this source", "", ""),
    ("CrossCheck", r"PDBe chains \(UniProt range\)", "PDBe chain = UniProt range", "", "author / UniProt"),
    ("CrossCheck", r"UniProt chains", "UniProt cross-reference chains and ranges", "", "UniProt"),
    ("CrossCheck", r"Flags", "Discrepancies between sources ('consistent' if none)", "", ""),
    # ---- Milestone 2: interfaces ------------------------------------------------------------------
    ("Interfaces|CrystalContacts|InterfaceContacts", r"Source", "biological assembly or crystal lattice", "", ""),
    ("Interfaces|CrystalContacts", r"Assembly", "Assembly analysed (ASU if the assembly was unavailable)", "", ""),
    ("Interfaces|CrystalContacts|InterfaceContacts", r"Protomer [12]", "Protomer: <label_asym>|<operator> in the assembly / lattice", "", "label"),
    ("Interfaces|CrystalContacts|InterfaceContacts", r"Chain [12]", "Author chain of the protomer", "", "author"),
    ("Interfaces|CrystalContacts", r"Chain \(label\) [12]", "label_asym_id of the protomer", "", "label"),
    ("Interfaces|CrystalContacts", r"Copy [12]", "Assembly / lattice operator generating the protomer", "", ""),
    ("Interfaces|CrystalContacts|InterfaceContacts", r"Interface ID", "<PDB>:<protomer 1>-<protomer 2> (lattice interfaces marked 'lattice')", "", ""),
    ("Interfaces|CrystalContacts", r"Atom contacts", "Inter-protomer heavy-atom pairs within the contact cut-off", "count", ""),
    ("Interfaces|CrystalContacts", r"Residue contacts", "Inter-protomer residue pairs in contact", "count", ""),
    ("Interfaces|CrystalContacts", r"Interface residues [12] \(UniProt\)", "Interface residues of the protomer", "", "UniProt"),
    ("Interfaces|CrystalContacts", r"N interface residues [12]", "Interface residues of the protomer", "count", ""),
    ("Interfaces|CrystalContacts|Summary", r"Interface residue count", "Distinct UniProt residues in the interface (both protomers, homodimer union)", "count", "UniProt"),
    ("Interfaces|CrystalContacts", r"Non-UniProt interface residues", "Contacting residues without a SIFTS mapping (tags), kept apart", "count", ""),
    ("Interfaces|CrystalContacts", r"H-bonds \(distance-based\)", "Donor–acceptor N/O pairs within hbond_max (no angle criterion)", "count", ""),
    ("Interfaces|CrystalContacts", r"Salt bridges", "Acidic O – basic N pairs within salt_bridge_max", "count", ""),
    ("Interfaces|CrystalContacts", r"Hydrophobic contacts", "Hydrophobic side-chain C–C pairs within hydrophobic_max", "count", ""),
    ("Interfaces|CrystalContacts", r"Aromatic contacts", "Aromatic ring pairs with centroids within aromatic_centroid_max", "count", ""),
    ("Interfaces|CrystalContacts", r"BSA total \(Å²\)", "SASA(1) + SASA(2) − SASA(complex), protein heavy atoms", "Å²", ""),
    ("Interfaces|CrystalContacts", r"BSA protomer [12] \(Å²\)", "Surface of this protomer buried by the partner", "Å²", ""),
    ("Interfaces|CrystalContacts|Summary", r"Dimer interface area \(Å²\)", "Interface area = BSA total / 2", "Å²", ""),
    ("Interfaces|CrystalContacts", r"SASA protomer [12] \(Å²\)", "SASA of the isolated protomer", "Å²", ""),
    ("Interfaces|CrystalContacts", r"Fraction buried [12]", "BSA of the protomer / its SASA", "fraction", ""),
    ("Interfaces|CrystalContacts", r"(Non-)?[Pp]olar fraction of BSA", "Share of BSA from N/O (polar) or C (non-polar) atoms", "fraction", ""),
    ("Interfaces|CrystalContacts", r"SASA method", "freesasa or shrake_rupley (numpy fallback)", "", ""),
    ("Interfaces|CrystalContacts", r"Interface (mean|max) normalised B", "Within-chain B z-score of interface residues", "z", ""),
    ("Interfaces|CrystalContacts", r"Validation outliers at interface", "Interface residues with wwPDB validation outliers", "", "UniProt"),
    ("Interfaces|CrystalContacts", r"Glycans near interface", "Glycan residues within the neighbourhood distance of interface atoms", "", "author"),
    ("Interfaces|CrystalContacts", r"Ligands/ions near interface", "Other het groups within the neighbourhood distance of interface atoms", "", "author"),
    ("Interfaces|CrystalContacts", r"Interface conservation", "Pointer to variant mapping (pockets/waters/variants workbook)", "", ""),
    ("Interfaces|CrystalContacts|Summary", r"Canonical interface similarity", "Jaccard similarity of the interface residue set to the canonical interface", "0–1", "UniProt"),
    ("Interfaces", r"Contact-pair similarity", "Jaccard similarity of unordered UniProt contact pairs to the canonical pairs", "0–1", "UniProt"),
    ("Interfaces", r"Fraction of canonical interface recovered", "|interface ∩ canonical| / |canonical|", "fraction", ""),
    ("Interfaces|Summary", r"Noncanonical interface flag", "Interface differs from the canonical dimer (low Jaccard or small BSA)", "", ""),
    ("Interfaces", r"Noncanonical reason", "Which noncanonical criterion triggered", "", ""),
    ("Interfaces", r"Interface RMSD \(Å\)", "CA RMSD of canonical interface residues after superposing protomer 1 on the reference", "Å", ""),
    ("Interfaces", r"Protomer-2 rotation \(°\)", "Rotation needed to re-fit protomer 2 after the protomer-1 superposition", "degrees", ""),
    ("Interfaces", r"Protomer-2 translation \(Å\)", "Translation needed to re-fit protomer 2", "Å", ""),
    ("Interfaces", r"Protomer assignment", "Which protomer was matched to the reference protomer 1", "", ""),
    ("Interfaces", r"Interface residues compared", "CA atoms used for the interface RMSD", "count", ""),
    ("Interfaces|Summary", r"Interface cluster", "Hierarchical cluster of interface residue sets (1 − Jaccard, distance threshold)", "", ""),
    ("Interfaces", r"In main cluster", "Interface belongs to the largest cluster", "", ""),
    ("Interfaces", r"Main interface", "Largest-BSA target interface of the entry (used in Summary and comparison)", "", ""),
    ("Interfaces", r"Missing canonical interface residues [12]", "Canonical interface residues without coordinates in this protomer", "", "UniProt"),
    ("Interfaces|Summary", r"Interface complete", "All canonical interface residues are modelled in both protomers", "", ""),
    ("CrystalContacts", r"Lattice operator", "Space-group operation and lattice shift (e.g. 2_565)", "", ""),
    ("CrystalContacts", r"Matches canonical dimer", "Lattice interface reproduces the canonical dimer (Jaccard ≥ threshold)", "", ""),
    ("InterfaceContacts", r"res_[ab]", "Residue label (aa + UniProt number, or resname + author number '(auth)')", "", "UniProt / author"),
    ("InterfaceContacts", r"unp_[ab]", "UniProt number", "", "UniProt"),
    ("InterfaceContacts", r"resname_[ab]", "Residue name", "", ""),
    ("InterfaceContacts", r"auth_[ab]", "Author residue number + insertion code", "", "author"),
    ("InterfaceContacts", r"min_dist", "Minimum heavy-atom distance", "Å", ""),
    ("InterfaceContacts", r"atom_contacts", "Atom pairs within the contact cut-off", "count", ""),
    ("InterfaceContacts", r"hbonds|salt_bridges|hydrophobic|aromatic", "Typed interactions of this residue pair", "count", ""),
    ("InterfaceContacts", r"closest_atoms", "Atom names of the closest pair", "", ""),
    ("InterfaceComparison", r"Interface [AB]", "Interface IDs compared", "", ""),
    ("InterfaceComparison", r"Residue Jaccard|Contact-pair Jaccard", "Jaccard similarity", "0–1", "UniProt"),
    ("InterfaceComparison", r"Fraction of A recovered in B", "|A ∩ B| / |A| (residue sets)", "fraction", ""),
    ("CanonicalInterface", r"Item|Value", "Canonical interface definition and reference", "", "UniProt"),
    # ---- Milestone 2: local model fit ----------------------------------------------------------------
    ("LocalModelFit", r"Group", "Residue group (active site, lid, interface, ligands, ions, …)", "", ""),
    ("LocalModelFit", r"Method kind", "xray / em / other (selects the metric and rule)", "", ""),
    ("LocalModelFit", r"Residues in group", "Residues in the group for this entry", "count", ""),
    ("LocalModelFit", r"Residues with metric", "Residues with RSCC (X-ray) or Q-score (EM) in the report", "count", ""),
    ("LocalModelFit", r"Metric", "RSCC (X-ray) or Q-score (EM): never compared with each other", "", ""),
    ("LocalModelFit", r"(Median|Min) RSCC", "Real-space correlation coefficient", "0–1", ""),
    ("LocalModelFit", r"(Median|Min) Q-score", "Q-score (map–model resolvability)", "−1–1", ""),
    ("LocalModelFit", r"Median RSRZ", "Real-space R Z-score (X-ray)", "z", ""),
    ("LocalModelFit", r"Lowest-supported residue", "Residue with the lowest metric", "", "author"),
    ("LocalModelFit", r"Residues (strong|acceptable|weak)", "Residues per support category", "count", ""),
    ("LocalModelFit", r"Group support", "Category from the group statistic with the method-specific rule", "", ""),
    ("LocalModelFit|LigandModelFit", r"(Group|Local) support score", "Category as number (3 strong … 0 unavailable)", "0–3", ""),
    ("LigandModelFit", r"Component", "PDB chemical component ID", "", ""),
    ("LigandModelFit", r"Residue no\. \(author\)", "Author residue number of the ligand/ion", "", "author"),
    ("LigandModelFit", r"Alt\. conformation", "Alternate location code", "", ""),
    ("LigandModelFit", r"RSCC|RSRZ|Q-score", "Validation-report fit metric of the ligand/ion", "", ""),
    ("LigandModelFit", r"Occupancy \(avg\)", "Average occupancy", "fraction", ""),
    ("LigandModelFit", r"Clashes|Geometry outliers|Chirality outliers", "Counts from the validation report", "count", ""),
    ("LigandModelFit", r"B ratio ligand/environment \(indirect\)", "Indirect indicator only (not map fit)", "ratio", ""),
    ("LigandModelFit", r"Local support", "Method-specific support category", "", ""),
    ("EMMapFit", r"Schema", "wwPDB validation XML schema file", "", ""),
    ("EMMapFit", r"emdb_id|EMDB-resolution|Q_score|atom_inclusion_.*|contour_level_primary_map|.*fsc_resolution.*|.*percentile-qrelative",
     "EM map / model metric copied from the validation report (Entry attribute of the same name)", "as reported", ""),
    ("EMMapFit", r"Per-chain mean Q-score", "Mean residue Q-score per author chain", "", "author"),
    ("EMMapFit", r"Primary map|Half maps", "Availability according to the EMDB API ('unknown' if not reported)", "", ""),
    # ---- Milestone 2: summary helpers and rankings --------------------------------------------------
    ("Summary", r"Canonical dimer in crystal lattice", "Lattice operators reproducing the canonical dimer (for entries whose assembly lacks it)", "", ""),
    ("Summary", r"(Active-site|Ligand|Interface|Entrance-lid) local support", "Group support category (LocalModelFit)", "", ""),
    ("Summary", r"(Active-site|Ligand|Interface|Entrance-lid) local support score", "Group support as number for ranking (3 … 0)", "0–3", ""),
    ("Summary", r"Ligand local support \(worst ligand\)", "Lowest category among ligands of interest (LigandModelFit)", "", ""),
    ("Summary", r"Lowest-supported important residue", "Lowest-metric residue over all analysed groups", "", "author"),
    ("Summary", r"Local-fit availability", "Validation-report residues available", "", ""),
    ("Summary", r"Local-fit warning", "Groups with weak support", "", ""),
    ("Summary", r"Interface validation warning", "Validation outliers at the main interface", "", ""),
    ("Summary", r"Substitutions vs UniProt \(all chains\)", "All point differences from UniProt (engineered or natural)", "", "UniProt"),
    ("Summary", r"N substitutions vs UniProt", "Number of distinct substitutions", "count", ""),
    ("Summary", r"Has other polymer chains", "Non-target polymer entities present (e.g. antibodies)", "", ""),
    ("Summary", r"Glycosylated Asn count", "Distinct observed N-glycosylation sites", "count", ""),
    ("Summary", r"Zn site complete", "Zn in assembly = expected per protomer × protomers", "", ""),
    ("Summary", r"Validation XML schema", "wwPDB validation XML schema used (or 'unavailable')", "", ""),
    ("Summary", r"EM Q-score", "Entry-level Q-score (cryo-EM)", "−1–1", ""),
    ("TemplateRanking", r"RankingMode|Mode description", "Ranking mode and its purpose (config ranking.modes)", "", ""),
    ("TemplateRanking", r"Entries eligible", "Entries passing the mode's filters", "count", ""),
    # ---- Milestone 3: pockets, waters, variants --------------------------------------------------
    ("SubpocketDefinition|SubpocketOccupancy|LigandInteractions", r"Sub-pockets?", "Sub-pocket name(s) (config/psma_active_site.yaml)", "", ""),
    ("SubpocketDefinition", r"Residue", "Configured residue code", "", "UniProt"),
    ("SubpocketDefinition|LigandInteractions", r"UniProt number", "Residue number in the canonical sequence", "", "UniProt"),
    ("PocketFingerprints|SubpocketOccupancy|LigandClusters|LigandChemistry", r"Component", "PDB chemical component ID", "", ""),
    ("PocketFingerprints", r"Residue no\. \(author\)", "Author residue number of the ligand", "", "author"),
    ("SubpocketDefinition", r"Reason", "Why a residue was dropped (sequence mismatch)", "", ""),
    ("PocketFingerprints|SubpocketOccupancy|LigandInteractions|LigandClusters", r"Instance|Ligand instance",
     "Ligand instance: PDB:component:author chain+number|assembly copy", "", "author"),
    ("PocketFingerprints", r"Copy", "Assembly operator that generated this copy", "", ""),
    ("PocketFingerprints", r"Heavy atoms", "Heavy atoms of the ligand instance", "count", ""),
    ("PocketFingerprints", r"Residues contacted", "Protein residues with a heavy atom ≤ contact_max of the ligand", "count", ""),
    ("PocketFingerprints", r"Contacted residues \(UniProt\)", "Binary residue fingerprint", "", "UniProt"),
    ("PocketFingerprints", r"Interaction bits", "Interaction-type fingerprint '<residue>:<type>' (distance-only rules)", "", "UniProt"),
    ("PocketFingerprints", r"N .*", "Number of ligand-atom interactions of this type (distance-only; 'possible' = not geometry-checked)", "count", ""),
    ("PocketFingerprints|LigandClusters", r"Zn-coordinating atoms", "Ligand N/O/S atoms ≤ metal_max of a Zn", "count", ""),
    ("PocketFingerprints", r"Metal coordination", "Ligand atom – metal pairs", "", "author"),
    ("PocketFingerprints|LigandClusters", r"Occupied sub-pockets", "Sub-pockets with ≥ 1 contacted residue", "", ""),
    ("PocketFingerprints", r"Contacts partner protomer", "Ligand also contacts another target protomer of the assembly", "", ""),
    ("PocketFingerprints", r"Chains contacted", "Author chains contacted", "", "author"),
    ("PocketFingerprints", r"Representative", "Instance with most contacts for its (entry, component): used for similarity/clustering", "", ""),
    ("PocketFingerprints|LigandClusters|LigandConditionedRanking", r"Entrance lid state", "Entrance-lid call from notebook 01", "", ""),
    ("SubpocketOccupancy", r"Occupied", "Ligand contacts ≥ 1 residue of the sub-pocket", "", ""),
    ("SubpocketOccupancy", r"Contacted residues", "Sub-pocket residues contacted", "", "UniProt"),
    ("SubpocketOccupancy", r"Defined residues", "Validated residues in the sub-pocket", "count", ""),
    ("SubpocketOccupancy", r"Modelled in ligand chain", "Sub-pocket residues with a modelled CA in the ligand's chain", "count", ""),
    ("LigandInteractions", r"Ligand atom|Protein atom", "Atom names ('(ring)' = ring centroid; water bridge shows the HOH)", "", "author"),
    ("LigandInteractions", r"Protein residue \(UniProt\)", "Interacting residue", "", "UniProt"),
    ("LigandInteractions", r"Author residue", "Author residue number", "", "author"),
    ("LigandInteractions", r"Protomer", "Protomer of the interacting residue", "", ""),
    ("LigandInteractions", r"Partner protomer", "Residue belongs to a different protomer from the ligand copy", "", ""),
    ("LigandInteractions", r"Distance \(Å\)", "Atom–atom or centroid distance", "Å", ""),
    ("LigandInteractions", r"Interaction type", "Distance-rule interaction class (see fingerprints.py)", "", ""),
    ("PocketSimilarity", r"Ligand [AB]", "Representative ligand instances compared", "", "author"),
    ("PocketSimilarity", r"Residue Jaccard", "Jaccard of contacted-residue sets", "0–1", ""),
    ("PocketSimilarity", r"Interaction Tanimoto", "Tanimoto of interaction-type fingerprints (used for clustering)", "0–1", ""),
    ("PocketSimilarity", r"Weighted cosine", "Cosine of distance-weighted residue vectors", "0–1", ""),
    ("PocketSimilarity", r"Sub-pocket Tanimoto", "Tanimoto of occupied sub-pocket sets", "0–1", ""),
    ("LigandClusters", r"Hierarchical cluster", "Cluster at the chosen distance threshold (average linkage on 1 − Tanimoto)", "", ""),
    ("LigandClusters", r"DBSCAN cluster", "DBSCAN label (−1 = noise)", "", ""),
    ("ClusterDiagnostics", r"Method", "Clustering method", "", ""),
    ("ClusterDiagnostics", r"Threshold \(1 − Tanimoto\)", "Distance threshold (hierarchical) or eps (DBSCAN)", "", ""),
    ("ClusterDiagnostics", r"Clusters", "Number of clusters", "count", ""),
    ("ClusterDiagnostics", r"Silhouette", "Silhouette score (precomputed distances)", "−1–1", ""),
    ("ClusterDiagnostics", r"Chosen", "Threshold used for LigandClusters", "", ""),
    ("ClusterDiagnostics", r"Noise points", "DBSCAN points labelled noise", "count", ""),
    ("ClusterDiagnostics", r"Largest clusters", "Sizes of the largest DBSCAN clusters", "count", ""),
    ("LigandChemistry|LigandClusters", r"Ligand name|Formula|Formula weight|Formal charge \(CCD\)",
     "Chemical Component Dictionary definition (RCSB)", "", ""),
    ("LigandChemistry", r"Canonical SMILES|Isomeric SMILES|InChI|InChIKey", "CCD descriptor", "", ""),
    ("LigandChemistry", r"Definition status", "CCD definition retrieved", "", ""),
    ("LigandChemistry", r"Heavy atoms|H-bond donors|H-bond acceptors|Rotatable bonds|Aromatic rings", "RDKit descriptor (missing without RDKit)", "count", ""),
    ("LigandChemistry", r"Molecular weight", "RDKit average molecular weight", "Da", ""),
    ("LigandChemistry", r"Formal charge \(RDKit\)", "Net formal charge from the SMILES", "e", ""),
    ("LigandChemistry", r"Descriptor status|Descriptor note", "RDKit descriptor calculation status", "", ""),
    ("ConservedWaters|WaterMembers", r"Water cluster|Cluster", "Water cluster (W1 = most conserved)", "", ""),
    ("ConservedWaters", r"Sites with water|Eligible sites", "Superposed target chains with a member / eligible in total", "count", ""),
    ("ConservedWaters", r"Conservation fraction", "Sites with water / eligible sites", "fraction", ""),
    ("ConservedWaters", r"Centroid [xyz] \(ref\)", "Cluster centroid in the reference site's frame", "Å", ""),
    ("ConservedWaters", r"Median protein partners", "Median protein N/O atoms ≤ hbond_max", "count", ""),
    ("ConservedWaters", r"Frequent partner residues \(UniProt\)", "Most frequent H-bond partner residues (sites)", "", "UniProt"),
    ("ConservedWaters", r"Zn-bound fraction|Zn-bridging fraction", "Members ≤ zn_bridge_max of ≥ 1 / both Zn", "fraction", ""),
    ("ConservedWaters", r"Mean B \(Å²\)|Mean occupancy", "Mean of member waters", "", ""),
    ("ConservedWaters", r"Recommendation", "retain / consider / remove for modelling and MD (rule in waters.py)", "", ""),
    ("ConservedWaters", r"Reason", "Basis of the recommendation", "", ""),
    ("ConservedWaters", r"Entries", "Sites (PDB/chain) containing the water", "", "author"),
    ("WaterMembers", r"Water residue", "Author residue number of the water", "", "author"),
    ("WaterMembers", r"B|Occupancy", "Deposited B factor / occupancy of the water", "", ""),
    ("WaterMembers", r"Partners", "Protein N/O atoms ≤ hbond_max", "count", ""),
    ("WaterMembers", r"Partner residues", "Residues of those atoms", "", "UniProt"),
    ("WaterMembers", r"Zn contacts", "Zn ions ≤ zn_bridge_max", "count", ""),
    ("WaterMembers", r"Nearest Zn \(Å\)", "Distance to the nearest Zn of the site", "Å", ""),
    ("WaterSites", r"Reference", "Reference site for superposition", "", "author"),
    ("WaterSites", r"Superposition CA", "CA atoms used for superposition", "count", ""),
    ("WaterSites", r"Superposition RMSD \(Å\)", "CA RMSD after superposition", "Å", ""),
    ("WaterSites", r"Waters in region", "Waters within region_radius of the site", "count", ""),
    ("Variants|VariantStructureMapping", r"Source|Source type", "Variant source and kind (natural variant, mutagenesis, consequence type)", "", ""),
    ("Variants|VariantStructureMapping", r"Variant ID", "Source identifier (UniProt VAR_, dbSNP, …)", "", ""),
    ("Variants", r"Start|End", "Variant span", "", "UniProt"),
    ("Variants", r"Reference aa|Variant aa", "Amino acids as given by the source", "", ""),
    ("Variants", r"Description|Cross-references|Clinical significance", "As given by the source", "", ""),
    ("Variants", r"Evidence count", "Evidence records in the source", "count", ""),
    ("Variants", r"Status reason", "Why a variant was not mapped", "", ""),
    ("Variants|VariantStructureMapping", r"Position", "Substituted residue", "", "UniProt"),
    ("Variants|VariantStructureMapping", r"Substitution", "Reference–position–variant (one-letter)", "", "UniProt"),
    ("VariantStructureMapping", r"Reference structure", "Assembly and protomer used for structural context", "", ""),
    ("VariantStructureMapping", r"Modelled in reference", "Residue has atoms in the reference protomer", "", ""),
    ("VariantStructureMapping", r"Fraction of structures modelling residue", "Target chains with a modelled CA / all chains analysed", "fraction", ""),
    ("VariantStructureMapping", r"SASA \(Å²\)", "Residue SASA in the assembly", "Å²", ""),
    ("VariantStructureMapping", r"RSA", "Relative SASA (Tien et al. 2013 max ASA)", "fraction", ""),
    ("VariantStructureMapping", r"Normalised B", "Residue mean B as z-score within the protomer", "z", ""),
    ("VariantStructureMapping", r"Atom contacts \(≤4\.5 Å\)", "Heavy-atom contacts to other residues", "count", ""),
    ("VariantStructureMapping", r"Distance to (catalytic pocket|ligand|Zn|partner protomer) \(Å\)", "Minimum heavy-atom distance in the reference assembly (catalytic pocket = literature residues + Zn ligands + sub-pockets)", "Å", ""),
    ("VariantStructureMapping", r"CA RMSF across structures \(Å\)", "CA RMSF over all superposed target chains (≥ 3)", "Å", ""),
    ("VariantStructureMapping", r"Sequon effect", "N-X-S/T sequon removed or created", "", "UniProt"),
    ("VariantStructureMapping", r"Categories", "Structural location categories", "", ""),
    ("VariantStructureMapping", r"Hypothesis", "Location-based hypothesis only – not a prediction of effect", "", ""),
    ("VariantStructureMapping", r"AlphaFold pLDDT|AlphaFold status", "AlphaFold confidence (Milestone 4)", "", ""),
    ("VariantSummary", r"Group|Item|Count|Examples", "Counts per status / category", "", ""),
    ("LigandConditionedRanking", r"Reference ligand|Reference instance|Reference SMILES", "Reference used for similarities", "", ""),
    ("LigandConditionedRanking", r"Required sub-pockets|Desired lid state", "Ranking request", "", ""),
    ("LigandConditionedRanking", r"Best instance|Component", "Entry's ligand instance most similar to the reference", "", "author"),
    ("LigandConditionedRanking", r"Pocket-contact similarity", "Tanimoto of contacted residues vs the reference instance", "0–1", ""),
    ("LigandConditionedRanking", r"Interaction-fingerprint similarity", "Tanimoto of interaction bits vs the reference", "0–1", ""),
    ("LigandConditionedRanking", r"Required sub-pockets occupied", "Fraction of required sub-pockets occupied", "fraction", ""),
    ("LigandConditionedRanking", r"Ligand chemical similarity", "Morgan r2 2048-bit Tanimoto vs the reference SMILES (RDKit)", "0–1", ""),
    ("LigandConditionedRanking", r"Zinc-coordination similarity", "1 − |ΔZn-coordinating atoms| / max", "0–1", ""),
    ("LigandConditionedRanking", r"Entrance-lid state match", "1 = desired lid state, 0 = other state, empty = unknown", "", ""),
    ("LigandConditionedRanking", r"Missing residues near pocket", "Sub-pocket residues without CA (best chain)", "count", ""),
    ("LigandConditionedRanking", r"Conserved waters present", "Fraction of 'retain' conserved waters present", "fraction", ""),
    # ---- Milestone 4: AlphaFold comparison and membrane context --------------------------------------
    (".*", r"Model ID", "Predicted-model identifier (AFDB:<model entity>-v<version>, CF:<job>:rank_NNN for AlphaFold2 via ColabFold, MODEL:<label>)", "", ""),
    (r"AF\w+|Membrane\w+|ConstructContext", r"Chain", "Author chain ID in the model / structure", "", "author"),
    ("AFModels", r"Source", "AlphaFold DB, ColabFold (AlphaFold2) or user model", "", ""),
    ("AFModels", r"Model kind", "monomer (one target chain) or multimer (two or more)", "", ""),
    ("AFModels", r"Version", "AlphaFold DB release of the model", "", ""),
    ("AFModels", r"Model rank", "Rank given by ColabFold (rank_NNN)", "", ""),
    ("AFModels", r"Model", "AlphaFold2 model type and model/seed tag from the result files", "", ""),
    ("AFModels", r"Target chains", "Chains aligned to the target sequence", "", "author"),
    ("AFModels", r"Residues \(target chains\)", "Residues mapped to UniProt over target chains", "count", ""),
    ("AFModels|AFChains", r"UniProt range", "Mapped UniProt residues", "", "UniProt"),
    ("AFModels", r"Identity to UniProt \(min\)", "Lowest sequence identity of a target chain to UniProt (aligned residues)", "fraction", ""),
    ("AFModels", r"Mean pLDDT", "Mean CA pLDDT over target chains", "0–100", ""),
    ("AFModels", r"pLDDT .*\(%\)", "Percentage of residues in the pLDDT band", "%", ""),
    ("AFModels|AFRankings", r"Mean pLDDT (ectodomain|TM|active site|entrance lid)", "Mean CA pLDDT over the region (TM from the membrane topology)", "0–100", "UniProt"),
    ("AFModels", r"pTM|ipTM", "Global confidence reported by AlphaFold2 (ipTM for multimers)", "0–1", ""),
    ("AFModels", r"pLDDT in B-factor", "B-factor column holds pLDDT", "", ""),
    ("AFModels", r"AFDB canonical sequence", "AlphaFold DB model sequence equals the current UniProt canonical sequence", "", ""),
    ("AFModels", r"PAE status", "Result of the critical pae_indexing check", "", ""),
    ("AFModels", r"File", "Model file in the cache (originals are never modified)", "", ""),
    ("AFModels", r"SHA-256", "Hash of the model file", "", ""),
    ("AFModels", r"Best template", "First experimental reference (TemplateRanking order) with a valid comparison", "", ""),
    ("AFModels|AFRankings", r"(Core|All-common) CA RMSD to best template \(Å\)", "See AFvsExperimental (core = pLDDT ≥ confident_plddt)", "Å", ""),
    ("AFModels|AFRankings", r"TM-score to best template|GDT-TS-like score to best template", "See AFvsExperimental", "0–1", ""),
    ("AFModels|AFRankings", r"(Zn-ligand|Active-site) side-chain RMSD \(Å\)", "Side-chain heavy-atom RMSD after local superposition on the site CA atoms", "Å", ""),
    ("AFModels", r"pLDDT–deviation Spearman", "Spearman correlation of pLDDT with per-residue deviation (negative = well calibrated)", "", ""),
    ("AFModels", r"Entrance lid state", "Lid call(s) of the model chains (Milestone 1 classifier)", "", ""),
    ("AFModels|AFInterfaces|AFRankings", r"Interface PAE \(Å\)", "Mean PAE over interface residue pairs, both directions", "Å", "UniProt"),
    ("AFModels|AFInterfaces|AFRankings", r"Canonical interface Jaccard", "Residue Jaccard to the experimental canonical interface (assessment workbook)", "0–1", ""),
    ("AFInterfaces", r"Fraction of canonical interface recovered", "Canonical interface residues present in the predicted interface", "fraction", ""),
    ("AFModels|AFInterfaces", r"Dimer confidence", "Confident-dimer rule: confident / uncertain / low confidence (design §4)", "", ""),
    ("AFInterfaces", r"Dimer confidence reason", "Which criteria passed or failed", "", ""),
    ("AFInterfaces|AFRankings", r"ipTM", "AlphaFold2-multimer interface pTM of the complex", "0–1", ""),
    ("AFModels|AFRankings|MembranePlacement", r"TM in slab", "Fraction of TM residues inside the fitted membrane", "fraction", ""),
    ("AFModels|AFRankings|MembranePlacement", r"TM-ectodomain PAE \(Å\)", "Mean PAE between the TM helix and the ectodomain core (both directions); high = orientation not determined", "Å", ""),
    ("AFModels|AFRankings|MembranePlacement", r"Ectodomain atoms in slab", "Ectodomain heavy atoms inside the membrane slab", "count", ""),
    ("AFModels|AFRankings|MembranePlacement", r"C2 axis to membrane normal \(°\)", "Angle between the dimer two-fold axis and the TM-fitted membrane normal", "°", ""),
    ("AFModels|MembranePlacement", r"Orientation reliability", "supported / weak / unsupported (see Reliability reasons)", "", ""),
    ("AFChains", r"Residues", "Polymer residues in the chain", "count", ""),
    ("AFChains", r"Mapped residues|Unmapped residues", "Residues aligned / not aligned to UniProt", "count", ""),
    ("AFChains", r"Identity to UniProt", "Identity over aligned residues", "fraction", ""),
    ("AFChains", r"Target", "Chain treated as the target (identity ≥ min_identity_target)", "", ""),
    ("AFResidueConfidence", r"UniProt number", "Residue", "", "UniProt"),
    ("AFResidueConfidence", r"Residue", "Model amino acid (one-letter)", "", ""),
    ("AFResidueConfidence|AFRegionRMSD", r"pLDDT|Mean pLDDT", "Predicted local distance difference test (CA)", "0–100", ""),
    ("AFResidueConfidence", r"Confidence band", "very high / confident / low / very low (alphafold.plddt_bands)", "", ""),
    ("AFResidueConfidence", r"Regions", "Configured regions containing the residue", "", ""),
    ("AFResidueConfidence", r"Fraction of experimental chains covering residue", "Target chains whose SIFTS range covers the residue", "fraction", ""),
    ("AFResidueConfidence", r"Deviation from best template \(Å\)|Deviation reference", "CA deviation after the core fit, and the reference used", "Å", ""),
    ("AFResidueConfidence", r"Membrane depth \(Å\)", "CA position along the membrane normal (0 = mid-plane, < 0 cytoplasmic)", "Å", ""),
    ("AFResidueConfidence|MembraneTopology", r"Membrane label", "Topology label used: I cytoplasmic, O non-cytoplasmic, M TM helix, B β-strand, - unknown", "", ""),
    ("AFRegionConfidence", r"Region", "Configured region (domains, flexible regions, active site, Zn ligands, sequons, topology, interface)", "", ""),
    ("AFRegionConfidence", r"Residues in region|Residues modelled", "Residues defined / with a CA in the model", "count", ""),
    ("AFRegionConfidence", r"Mean pLDDT|Min pLDDT", "pLDDT over modelled residues", "0–100", ""),
    ("AFRegionConfidence", r"Fraction pLDDT < confident", "Residues below alphafold.confident_plddt", "fraction", ""),
    ("AFRegionConfidence", r"Intra-region PAE \(Å\)", "Mean PAE within the region", "Å", ""),
    ("AFDomainPAE", r"Chain \((aligned|scored)\)|Domain \((aligned|scored)\)", "PAE block: rows aligned on, columns scored", "", "author"),
    ("AFDomainPAE", r"(Mean|Median) PAE \(Å\)", "Statistic over the PAE block", "Å", ""),
    ("AFvsExperimental|AFRegionRMSD", r"Reference PDB ID|Reference chain", "Experimental reference chain", "", "author"),
    ("AFvsExperimental", r"Common CA", "Residues with CA in both", "count", ""),
    ("AFvsExperimental", r"Confident-core residues", "Common residues with pLDDT ≥ confident_plddt (superposition set)", "count", ""),
    ("AFvsExperimental", r"Core CA RMSD \(Å\)", "CA RMSD over the confident core after fitting on it", "Å", ""),
    ("AFvsExperimental", r"All-common CA RMSD after core fit \(Å\)", "CA RMSD over all common residues, core fit", "Å", ""),
    ("AFvsExperimental", r"Fraction within \d+ Å", "Common residues within the cutoff after the core fit", "fraction", ""),
    ("AFvsExperimental", r"GDT-TS-like score", "Mean of the 1/2/4/8 Å fractions (fixed core superposition)", "0–1", ""),
    ("AFvsExperimental", r"pLDDT–deviation Spearman", "Spearman correlation of pLDDT with deviation", "", ""),
    ("AFvsExperimental", r"TM-score \(reference-normalised\)", "TM-align score normalised by the reference (tmtools); empty = NOT RUN", "0–1", ""),
    ("AFvsExperimental", r"(Active-site|Zn-ligand) side-chain RMSD \(Å\)", "Side-chain heavy-atom RMSD after local superposition on the site CA", "Å", ""),
    ("AFvsExperimental", r"(Active-site|Zn-ligand) side-chain atoms compared", "Atoms present in both", "count", ""),
    ("AFRegionRMSD", r"Region", "Configured region", "", ""),
    ("AFRegionRMSD", r"Common residues", "Region residues with CA in both", "count", ""),
    ("AFRegionRMSD", r"CA RMSD after core fit \(Å\)|Max CA deviation \(Å\)", "Region deviation after the global core fit", "Å", ""),
    ("AFEntranceLid", r"Lid reference", "Chain used as the lid superposition reference", "", ""),
    ("AFRankings|AFRankingTop", r"Ranking mode", "alphafold.ranking.modes entry", "", ""),
    ("AFRankings", r"Mode description", "Description of the ranking mode", "", ""),
    ("AFRankingTop", r"Models eligible", "Models passing the mode's filters", "count", ""),
    ("AlphaFoldChecks", r"Check", "Sanity-check name (Milestone 4)", "", ""),
    ("AlphaFoldChecks", r"Detail", "What was checked / found", "", ""),
    ("MembraneTopology", r"UniProt number", "Residue", "", "UniProt"),
    ("MembraneTopology", r"Residue", "UniProt amino acid", "", ""),
    ("MembraneTopology", r"UniProt label|DeepTMHMM label", "Per-residue topology from the source (I/O/M/B/S/P/-)", "", ""),
    ("MembraneTopology", r"Sources agree", "UniProt and DeepTMHMM labels agree (S → unknown, P → inside)", "", ""),
    ("TopologyComparison", r"Source", "Topology source or comparison", "", ""),
    ("TopologyComparison", r"TM segments|TM ranges", "Transmembrane segments", "", "UniProt"),
    ("TopologyComparison", r"N-terminus", "Side of the N-terminus", "", ""),
    ("TopologyComparison", r"Residues labelled", "Residues with a known label", "count", ""),
    ("TopologyComparison", r"Residue agreement|TM residue Jaccard", "Agreement between UniProt and DeepTMHMM", "fraction", ""),
    ("TopologyComparison|ConstructContext", r"Note", "Explanation", "", ""),
    ("MembranePlacement", r"Placement method", "TM fit (tmfold) or C2-axis constrained", "", ""),
    ("MembranePlacement", r"Chains placed", "Target chains included in the fit", "", "author"),
    ("MembranePlacement", r"Loop side agreement", "Fraction of cytoplasmic / extracellular residues on their side", "fraction", ""),
    ("MembranePlacement", r"Membrane thickness \(Å\)", "Fitted hydrophobic thickness", "Å", ""),
    ("MembranePlacement|AFModels|AFRankings", r"TM helix tilt \(°\)", "Mean angle between TM helix axes and the membrane normal", "°", ""),
    ("MembranePlacement", r"TM pLDDT", "Mean pLDDT of TM residues", "0–100", ""),
    ("MembranePlacement", r"Ectodomain lowest atom above outer plane \(Å\)", "Clearance of the ectodomain core (negative = penetrates the membrane)", "Å", ""),
    ("MembranePlacement", r"Ectodomain atoms on cytoplasmic side", "Ectodomain heavy atoms below the inner plane", "count", ""),
    ("MembranePlacement", r"Sequons on wrong side", "Extracellular N-X-S/T Asn CA atoms not above the outer plane", "count", "UniProt"),
    ("MembranePlacement", r"Active-site height above membrane \(Å\)", "Mean Zn-ligand CA height above the outer plane", "Å", ""),
    ("MembranePlacement", r"C2 rotation angle \(°\)", "Rotation superposing chain 1 on chain 2 (180 = two-fold)", "°", ""),
    ("MembranePlacement", r"TM-fit normal to final normal \(°\)", "Change made by the C2 constraint", "°", ""),
    ("MembranePlacement", r"Membrane normal \(x, y, z\)|Membrane centre \(x, y, z\)", "Membrane frame in model coordinates (normal points cytoplasm → outside)", "Å", ""),
    ("MembranePlacement", r"Reliability reasons", "Why the orientation is weak / unsupported", "", ""),
    ("ConstructContext", r"UniProt range \(SIFTS\)", "Construct span of the experimental chain", "", "UniProt"),
    ("ConstructContext", r"Fraction of full length", "Construct span / UniProt length", "fraction", ""),
    ("ConstructContext", r"Covers (cytoplasmic|transmembrane|extracellular)", "Fraction of the topology region within the expressed construct (SIFTS range, includes unresolved residues)", "fraction", ""),
    ("ConstructContext", r"Modelled (cytoplasmic|transmembrane|extracellular)", "Fraction of the topology region actually modelled (construct minus missing residues)", "fraction", ""),
    ("ConstructContext", r"Modelled residues", "Residues modelled in the chain (construct minus missing residues)", "count", "UniProt"),
    ("AFInterfaceContacts", r"Interface", "Protomer pair of the predicted interface", "", ""),
    ("MembraneMDReadiness", r"Item", "Blocker between membrane-ready inputs and an MD-ready system", "", ""),
    ("MembraneMDReadiness", r"Detail", "Finding and the milestone that resolves it", "", ""),
    # ---- Milestone 5: model preparation ---------------------------------------------------------------
    (r"Prep\w+|Hybrid\w+|ResidueSources|Junctions|Repairs|TemplateHetero|Glycans|Zn\w+|Protonation|Microstates|Clashes", r"Chain", "Chain of the prepared hybrid model", "", "author"),
    ("HybridSegments", r"Start|End", "Segment boundaries", "", "UniProt"),
    ("HybridSegments|ResidueSources", r"Source", "Where the coordinates come from (experimental template, AF2 anchor / gap / side chain / reverted substitution)", "", ""),
    ("HybridSegments", r"Residues", "Residues in the segment", "count", ""),
    ("HybridSegments", r"Fit RMSD \(Å\)", "Superposition RMSD used to place an AF2 segment (global dimer fit or local flank fit)", "Å", ""),
    ("HybridSegments", r"Mean pLDDT", "Mean AF2 pLDDT of an AF2-derived segment", "0–100", ""),
    ("ResidueSources|Repairs|Protonation", r"UniProt number", "Residue", "", "UniProt"),
    ("ResidueSources|Protonation", r"Residue", "Residue type (one-letter or three-letter)", "", ""),
    ("ResidueSources", r"pLDDT \(AF2\)", "AF2 pLDDT of the residue in the anchor model", "0–100", ""),
    ("Junctions", r"Junction", "Residue pair joined across a source change", "", "UniProt"),
    ("Junctions", r"Type", "Kind of graft", "", ""),
    ("Junctions", r"Graft shifted by \(residues\)", "Template residues replaced by AF2 to find the best N-terminal graft point", "count", ""),
    ("Junctions", r"Peptide C–N \(Å\)", "C(i)–N(i+1) distance at the junction (ideal ≈ 1.33 Å)", "Å", ""),
    ("Repairs", r"Repair", "Change made to a template residue", "", ""),
    ("Repairs", r"Template residue|UniProt residue", "Residue in the template / in UniProt", "", ""),
    ("Repairs", r"Atoms from AF2", "Atoms taken from the locally fitted AF2 residue", "count", ""),
    ("TemplateHetero", r"Component", "Chemical component of the template hetero group", "", ""),
    ("TemplateHetero", r"Template group", "label_asym|residue|assembly copy of the group in the template", "", ""),
    ("TemplateHetero", r"Nearest chain", "Hybrid chain the group belongs to (nearest protein atom / linked Asn)", "", ""),
    ("TemplateHetero|Glycans", r"Decision", "kept / removed (hetero), core GlcNAc / not glycosylated (glycans)", "", ""),
    ("TemplateHetero|Glycans", r"Note", "Reason for the decision", "", ""),
    ("TemplateHetero", r"Atoms", "Heavy atoms in the group", "count", ""),
    ("TemplateHetero", r"Linked Asn \(UniProt\)", "Asn carrying a core sugar", "", "UniProt"),
    ("Glycans|SequonObservations", r"Asn \(UniProt\)", "Asn of an N-X-S/T sequon", "", "UniProt"),
    ("Glycans|SequonObservations", r"Sequon", "N-X-S/T motif", "", ""),
    ("Glycans|SequonObservations", r"Entries with glycan", "PDB entries with a glycan on this Asn (survey Glycosylation sheet)", "count", ""),
    ("SequonObservations", r"Entries covering residue", "PDB entries whose construct includes the Asn", "count", ""),
    ("Glycans|SequonObservations", r"Observed fraction", "Entries with glycan / entries covering the residue", "fraction", ""),
    ("SequonObservations", r"Example entries", "Up to six entries with the glycan (donors for transplanting)", "", ""),
    ("Glycans", r"Source", "Template, or donor entry/chain of a transplanted core GlcNAc", "", ""),
    ("ZnSite|ZnContacts", r"Zn", "Zn ion (chain/ZN<residue number>)", "", ""),
    ("ZnSite|ZnContacts", r"Site", "Zn1 / Zn2 (assigned by the configured ligands)", "", ""),
    ("ZnSite", r"Protein ligands", "Coordinating protein atoms with distances", "Å", "UniProt"),
    ("ZnSite", r"Expected ligands", "Configured ligands of the site (psma_active_site.yaml)", "", "UniProt"),
    ("ZnSite", r"Coordination number", "N/O/S atoms within contacts.metal", "count", ""),
    ("ZnSite", r"Waters coordinating", "Water oxygens within contacts.metal", "count", ""),
    ("ZnSite", r"Mean ligand distance \(Å\)", "Mean Zn–protein ligand distance", "Å", ""),
    ("ZnSite", r"Zn–Zn distance \(Å\)", "Distance to the other Zn of the same chain", "Å", ""),
    ("ZnSite", r"Bridging water/hydroxide", "Water oxygens within waters.zn_bridge_max of both Zn", "count", ""),
    ("ZnContacts", r"Residue|Atom|Kind", "Coordinating atom", "", "UniProt"),
    ("ZnContacts", r"Distance \(Å\)", "Zn–donor distance", "Å", ""),
    ("ZnContacts", r"In range", "Within sanity.zn_ligand_distance_range", "", ""),
    ("MetalTreatmentOptions", r"Treatment", "Value for preparation.zn_md_treatment", "", ""),
    ("MetalTreatmentOptions", r"Model|Needs|Keeps geometry|Bridging water / hydroxide|Effort", "Property of the Zn MD treatment", "", ""),
    ("MetalTreatmentOptions", r"Chosen", "Selected in the configuration", "", ""),
    ("Protonation", r"pKa", "Predicted pKa (PROPKA) or default value", "pH units", ""),
    ("Protonation", r"pKa source", "PROPKA / default pKa / fixed (Zn ligand, disulfide)", "", ""),
    ("Protonation", r"pH", "Target pH", "", ""),
    ("Protonation", r"State|Alternative state", "AMBER residue name of the default / alternative protonation state", "", ""),
    ("Protonation", r"Reason", "Why the state was chosen", "", ""),
    ("Microstates", r"Microstate|Change from default|File", "Protonation microstate file (default + one alternative each)", "", ""),
    ("Clashes", r"Atom \d", "Clashing atom (chain/residue:atom)", "", "UniProt"),
    ("Clashes", r"Distance \(Å\)", "Heavy-atom distance", "Å", ""),
    ("Clashes", r"Source \d", "Coordinate source of each atom", "", ""),
    ("PrepSummary", r"Item|Value", "Run summary: template, anchor, fit, counts, protonation, Zn treatment, output folder", "", ""),
    ("PrepReadiness", r"Item", "Blocker between the prepared model and an MD-ready system", "", ""),
    ("PrepReadiness", r"Detail", "Finding and the milestone that resolves it", "", ""),
    # ---- Milestone 6 part A: Zn parameters (MCPB.py + PySCF) ---------------------------------------
    ("AmberTools", r"Tool|Path|Available", "AmberTools executable found on PATH", "", ""),
    ("SiteGeometry|QMSummary|TleapCheck", r"Item|Value", "Site geometry / QM summary / tleap build summary", "", ""),
    ("QMSummary", r"Model", "MCPB.py model (small: Seminario force constants; large: RESP charges)", "", ""),
    ("FrcmodBond", r"Bond", "Atom types of a metal-site bond (MCPB.py new types M*, Y*)", "", ""),
    ("FrcmodBond", r"k \(kcal/mol/Å²\)", "Seminario bond force constant", "kcal/mol/Å²", ""),
    ("FrcmodBond", r"r0 \(Å\)", "Equilibrium bond length (QM-optimised small model)", "Å", ""),
    ("FrcmodAngl", r"Angle", "Atom types of a metal-site angle", "", ""),
    ("FrcmodAngl", r"k \(kcal/mol/rad²\)", "Seminario angle force constant", "kcal/mol/rad²", ""),
    ("FrcmodAngl", r"θ0 \(°\)", "Equilibrium angle", "°", ""),
    ("FrcmodBond|FrcmodAngl", r"Comment", "frcmod comment (method)", "", ""),
    ("FrcmodNonb", r"Type|R\* \(Å\)|ε \(kcal/mol\)", "Lennard-Jones parameters of the new metal types", "", ""),
    ("SiteResidues", r"Residue", "MCPB.py renamed metal-site residue (mol2 template)", "", ""),
    ("SiteResidues", r"Atoms", "Atoms in the residue template", "count", ""),
    ("SiteResidues", r"Net charge", "Sum of RESP charges of the residue (the whole site sums to an integer)", "e", ""),
    ("SiteResidues", r"Metal-site atoms \(type, RESP charge\)", "Zn and ligating atoms with their new types and charges", "e", ""),
    ("ZnBonds", r"Atom \d", "Bonded atoms (residue as renamed by MCPB.py, number, atom)", "", "UniProt"),
    ("ZnBonds", r"Types", "Atom types of the bond", "", ""),
    ("ZnBonds", r"Model distance \(Å\)", "Distance in the prepared model (experimental Zn site)", "Å", ""),
    ("ZnBonds", r"MCPB r0 \(Å\)", "Equilibrium length from the QM-optimised small model", "Å", ""),
    ("ZnBonds", r"MCPB k \(kcal/mol/Å²\)", "Seminario force constant", "kcal/mol/Å²", ""),
    ("ZnBonds", r"r0 − model \(Å\)", "QM equilibrium minus experimental distance", "Å", ""),
    # ---- Milestone 6 part B: membrane system build and minimisation -------------------------------
    ("SoluteSummary|MergeSummary|TleapBuild|SystemSummary", r"Item|Value", "Solute / merge / tleap build / OpenMM system summary", "", ""),
    ("ZnTemplateMap", r"Chain", "Chain of the Zn ion", "", ""),
    ("ZnTemplateMap", r"Template", "MCPB.py Zn residue (parameterised on chain A) assigned by matching ligand residues", "", ""),
    ("ZnTemplateMap", r"Ligand residues", "Protein residues with an atom within 2.6 Å of the Zn", "", "UniProt"),
    ("GlycanLinks", r"Chain", "Chain of the glycosylated Asn", "", ""),
    ("GlycanLinks", r"Asn \(UniProt\)", "Glycosylated Asn (renamed NLN, GLYCAM)", "", "UniProt"),
    ("GlycanLinks", r"GLYCAM residue seq|Asn seq", "Sequential residue number in system.pdb (used by the tleap bond command)", "", "sequential"),
    ("GlycanLinks", r"C1–ND2 \(Å\)", "GlcNAc C1 – Asn ND2 distance (bonded in tleap)", "Å", ""),
    ("Composition", r"Component", "lipid / water / ion / other molecules from packmol-memgen", "", ""),
    ("Composition", r"Molecules", "Molecules kept after clash removal", "count", ""),
    ("Composition", r"Removed \(clash\)", "Whole molecules removed for a heavy atom within the clash cut-off of the solute", "count", ""),
    ("Composition", r"Types", "Residue composition of the kept molecules with counts", "", ""),
    ("Minimisation", r"Stage", "Minimisation / smoke-run stage", "", ""),
    ("Minimisation", r"Potential energy \(kJ/mol\)", "OpenMM potential energy", "kJ/mol", ""),
    ("Minimisation", r"Finite", "Energy is finite (no NaN / overflow)", "", ""),
    ("Minimisation", r"Max force \(kJ/mol/nm\)", "Largest per-atom force", "kJ/mol/nm", ""),
    ("Minimisation", r"Protein heavy-atom RMSD from start \(Å\)", "RMSD of protein heavy atoms from the built coordinates", "Å", ""),
    ("Minimisation", r"Zn–ligand distances \(Å\)", "All bonded Zn–ligand distances (both sites)", "Å", ""),
    ("Minimisation", r"Max Zn–ligand change \(Å\)", "Largest change of a Zn–ligand distance from the start", "Å", ""),
    ("Minimisation", r"Temperature \(K\)", "Instantaneous temperature at the end of the stage", "K", ""),
    ("Minimisation", r"Box \(Å\)", "Periodic box edge lengths", "Å", ""),
    ("StartForces", r"Atom index", "0-based atom index in system.prmtop", "", ""),
    ("StartForces", r"Residue|Atom", "Residue / atom name in the built system", "", ""),
    ("StartForces", r"Residue number \(tleap\)", "Sequential residue number in system.prmtop", "", "sequential"),
    ("StartForces", r"Kind", "protein / lipid / water / ion / metal site, glycan or other", "", ""),
    ("StartForces", r"Force \(kJ/mol/nm\)", "Force on the atom in the built system before any minimisation", "kJ/mol/nm", ""),
    ("MDReadiness", r"Item", "Item between the built system and equilibration (OVERALL: READY if nothing FAILs)", "", ""),
    ("MDReadiness", r"Detail", "Finding, assumption or the check that failed", "", ""),
    # ---- run-level sheets ------------------------------------------------------------------------
    ("RunIssues", r"Stage", "Pipeline stage that recorded the issue", "", ""),
    ("RunIssues", r"Message", "Issue description", "", ""),
    ("RunIssues", r"Detail", "Additional key=value details", "", ""),
    ("SanityChecks", r"Check", "Sanity-check name", "", ""),
    ("SanityChecks", r"Critical", "A FAIL in this check stops the full run", "", ""),
    ("SanityChecks", r"Detail", "What was checked / found", "", ""),
    ("Software", r"Software", "Package or executable", "", ""),
    ("Software", r"Kind", "python package / executable", "", ""),
    ("Software", r"Available", "Importable / found on PATH", "", ""),
    ("Software", r"Version", "Installed version (packages only)", "", ""),
    ("Software", r"Location", "Module name or executable path", "", ""),
    ("ValidationSet", r"Category", "Validation-set category", "", ""),
    ("ValidationSet", r"Reason", "Why the entry was selected", "", ""),
    ("DataDictionary", r"Sheet|Column|Description|Units|Numbering", "Data-dictionary field", "", ""),
    ("WorkbookComparison", r"Sheet|Kind|Key|Column|Old|New|Detail", "Workbook comparison field", "", ""),
]
_COMPILED = [(re.compile(f"^(?:{s})$"), re.compile(f"^(?:{c})$"), d, u, n) for s, c, d, u, n in _DICT]


SHEET_ALIASES = {"SystemChecks": ["SanityChecks"], "SystemIssues": ["RunIssues"], "ZincChecks": ["SanityChecks"], "ZincIssues": ["RunIssues"], "HybridMembrane": ["MembranePlacement"], "PrepChecks": ["SanityChecks"], "PrepIssues": ["RunIssues"],
                 "VariantAlphaFold": ["VariantStructureMapping"], "AFInterfaces": ["Interfaces"], "AFInterfaceContacts": ["InterfaceContacts"],
                 "AFEntranceLid": ["EntranceLid"], "AFRankings": ["TemplateRankings", "TemplateRanking"],
                 "AFRankingTop": ["RankingTop", "TemplateRanking"], "AlphaFoldIssues": ["RunIssues"],
                 "AlphaFoldChecks": ["SanityChecks"], "AssessmentSummary": ["Summary"], "TemplateRankings": ["TemplateRanking", "Summary"],
                 "RankingTop": ["TemplateRanking"], "AssessmentIssues": ["RunIssues"],
                 "LigandConditionedRanking": ["TemplateRanking", "Summary"], "PocketIssues": ["RunIssues"], "PocketChecks": ["SanityChecks"],
                 "LigandClusters": ["PocketFingerprints"], "SubpocketOccupancy": ["PocketFingerprints"]}


def describe_column(sheet: str, column: str) -> dict[str, str]:
    """Data-dictionary entry for a column (sheet-specific entries win over generic ones;
    derived sheets such as AssessmentSummary fall back to their parent sheet)."""
    best = None
    for name in [sheet] + SHEET_ALIASES.get(sheet, []):
        for s, c, d, u, n in _COMPILED:
            if c.match(str(column)) and s.match(name):
                specific = s.pattern != "^(?:.*)$"
                if best is None or (specific and not best[0]):
                    best = (specific, d, u, n)
        if best and best[0]:
            break
    if best is None:
        return {"Description": "UNDOCUMENTED", "Units": "", "Numbering": ""}
    return {"Description": best[1], "Units": best[2], "Numbering": best[3]}


def data_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (sheet, column) with description, units and numbering scheme."""
    rows = []
    for sheet, df in tables.items():
        for col in df.columns:
            rows.append({"Sheet": sheet, "Column": str(col), **describe_column(sheet, str(col))})
    rows.append({"Sheet": "(all)", "Column": "(missing values)", "Description": MISSING_CONVENTION,
                 "Units": "", "Numbering": ""})
    return pd.DataFrame(rows)


STATUS_FILLS = {"PASS": "C6EFCE", "OK": "C6EFCE", "WARNING": "FFEB9C", "FAIL": "FFC7CE",
                "NOT RUN": "E7E6E6", "NOT APPLICABLE": "E7E6E6"}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Consistent missing values: NaN/None/'' -> empty cell; tuples/sets -> strings."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: ", ".join(map(str, v)) if isinstance(v, (set, tuple, list)) else v)
    return out.replace({np.nan: None})


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame], add_dictionary: bool = True) -> Path:
    """Write sheets to ``path`` with consistent formatting and a DataDictionary sheet.

    Sheet names longer than 31 characters are truncated (Excel limit) and the
    truncation is logged.
    """
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    sheets = dict(sheets)
    if add_dictionary:
        sheets["DataDictionary"] = data_dictionary({k: v for k, v in sheets.items() if k != "DataDictionary"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            sname = name[:31]
            if sname != name:
                log.warning("sheet name %r truncated to %r", name, sname)
            df = _clean(df if isinstance(df, pd.DataFrame) else pd.DataFrame(df))
            df.to_excel(xw, sheet_name=sname, index=False)
            ws = xw.sheets[sname]
            ws.freeze_panes = "B2"
            if ws.max_row > 1 and ws.max_column >= 1:
                ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="305496")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for i, col in enumerate(df.columns, start=1):
                vals = [str(col)] + [str(v) for v in df[col].head(200) if v is not None]
                width = min(max(len(v) for v in vals) + 2, 60)
                ws.column_dimensions[get_column_letter(i)].width = max(width, 10)
                if re.search(r"status|^Flags$", str(col), re.I) and ws.max_row > 1:
                    rng = f"{get_column_letter(i)}2:{get_column_letter(i)}{ws.max_row}"
                    for word, colour in STATUS_FILLS.items():
                        ws.conditional_formatting.add(rng, CellIsRule(
                            operator="equal", formula=[f'"{word}"'],
                            fill=PatternFill("solid", fgColor=colour, bgColor=colour)))
    return path


def write_tables(folder: Path, sheets: dict[str, pd.DataFrame], csv: bool = True,
                 parquet: bool = True) -> list[Path]:
    """Write each table as CSV and/or Parquet; returns written paths.

    Parquet is skipped (with a log warning) if pyarrow/fastparquet is missing.
    """
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pq_ok = parquet
    for name, df in sheets.items():
        df = _clean(df)
        if csv:
            p = folder / f"{name}.csv"
            df.to_csv(p, index=False)
            written.append(p)
        if pq_ok:
            try:
                p = folder / f"{name}.parquet"
                df.astype({c: "string" for c in df.columns if df[c].dtype == object}).to_parquet(p, index=False)
                written.append(p)
            except ImportError:
                log.warning("Parquet output skipped: install pyarrow")
                pq_ok = False
    return written


# ---------------------------------------------------------------------------
# Workbook comparison (milestone regression check)
# ---------------------------------------------------------------------------
KEY_COLUMNS = ["PDB ID", "Model", "Chain", "Chain (auth)", "Entity", "Component", "Residue no.",
               "UniProt residue", "Assembly", "Asn (UniProt)", "Root sugar", "Item", "Ligand size"]


def _keys(df: pd.DataFrame) -> list[str]:
    return [k for k in KEY_COLUMNS if k in df.columns]


def _equal(a: object, b: object, tol: float) -> bool:
    """Numbers within ``tol``; strings equal, or equal as unordered ', '-separated lists."""
    if (a is None or (isinstance(a, float) and np.isnan(a))) and \
       (b is None or (isinstance(b, float) and np.isnan(b))):
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        sa, sb = str(a).strip(), str(b).strip()
        return sa == sb or sorted(sa.split(", ")) == sorted(sb.split(", "))


def compare_workbooks(old: str | Path, new: str | Path, sheets: Iterable[str] | None = None,
                      tol: float = 1e-6, max_cells: int = 50) -> pd.DataFrame:
    """Differences between two workbooks, sheet by sheet.

    Reports added/removed sheets and columns, row-count changes, and value
    differences on shared columns for rows matched by key columns (PDB ID,
    chain, component, …). Use it at the end of each milestone against the
    previous workbook.

    Returns a table with columns Sheet, Kind, Key, Column, Old, New, Detail.
    """
    o, n = pd.read_excel(old, sheet_name=None), pd.read_excel(new, sheet_name=None)
    rows: list[dict] = []
    names = list(sheets) if sheets else sorted(set(o) | set(n))
    for s in names:
        if s not in n:
            rows.append({"Sheet": s, "Kind": "sheet removed"}); continue
        if s not in o:
            rows.append({"Sheet": s, "Kind": "sheet added"}); continue
        a, b = o[s], n[s]
        for c in a.columns.difference(b.columns):
            rows.append({"Sheet": s, "Kind": "column removed", "Column": c})
        for c in b.columns.difference(a.columns):
            rows.append({"Sheet": s, "Kind": "column added", "Column": c})
        if len(a) != len(b):
            rows.append({"Sheet": s, "Kind": "row count", "Old": len(a), "New": len(b)})
        keys = [k for k in _keys(a) if k in b.columns]
        if s == "Notes":
            continue
        if not len(a) and not len(b):
            rows.append({"Sheet": s, "Kind": "empty in both"}); continue
        if not keys:
            rows.append({"Sheet": s, "Kind": "not compared", "Detail": "no key columns"}); continue
        # repeated keys (e.g. several oligosaccharide rows per entry) are matched by occurrence,
        # after sorting on the shared columns so that row order does not matter
        shared = [c for c in a.columns if c in b.columns]
        a = a.loc[a[shared].astype(str).sort_values(shared).index]
        b = b.loc[b[shared].astype(str).sort_values(shared).index]
        a = a.assign(_occ=a.groupby(keys, dropna=False).cumcount())
        b = b.assign(_occ=b.groupby(keys, dropna=False).cumcount())
        a2 = a.set_index(keys + ["_occ"])
        b2 = b.set_index(keys + ["_occ"])
        common = a2.index.intersection(b2.index)
        if len(common) < len(a2):
            rows.append({"Sheet": s, "Kind": "rows only in old", "Old": len(a2) - len(common)})
        if len(common) < len(b2):
            rows.append({"Sheet": s, "Kind": "rows only in new", "New": len(b2) - len(common)})
        shown = 0
        for c in a2.columns.intersection(b2.columns):
            diff = [k for k in common if not _equal(a2.at[k, c], b2.at[k, c], tol)]
            if diff:
                rows.append({"Sheet": s, "Kind": "values differ", "Column": c,
                             "Detail": f"{len(diff)} of {len(common)} rows"})
                for k in diff[: max(0, max_cells - shown)]:
                    *kk, occ = k if isinstance(k, tuple) else (k,)
                    label = (kk[0] if len(kk) == 1 else str(tuple(kk))) + (f" #{occ + 1}" if occ else "")
                    rows.append({"Sheet": s, "Kind": "cell", "Key": label, "Column": c,
                                 "Old": a2.at[k, c], "New": b2.at[k, c]})
                    shown += 1
    return pd.DataFrame(rows, columns=["Sheet", "Kind", "Key", "Column", "Old", "New", "Detail"])
