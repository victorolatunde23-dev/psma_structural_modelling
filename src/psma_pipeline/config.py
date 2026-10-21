"""Configuration loading, merging and validation.

All target-specific knowledge (accession, active site, regions, ligand classes,
thresholds) lives in YAML files under ``config/``. Functions in this package
receive values from a :class:`Config`; none of them hard-code Q04609.

Input contract
    ``load_config(path=None, overrides=None)``; ``path`` defaults to
    ``config/default_config.yaml`` of the project. ``overrides`` is a nested
    dict (or a YAML path) merged on top.

Output contract
    A :class:`Config` whose ``raw`` dict contains the merged main config plus
    three loaded target files under the keys ``active_site``, ``regions`` and
    ``ligand_classes``.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = Path(os.environ.get("PSMA_CONFIG_DIR", PROJECT_ROOT / "config"))
RESIDUE_CODE = re.compile(r"^([A-Z])(\d+)$")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as ex:
        raise ConfigError(f"cannot parse {path}: {ex}") from ex
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at top level")
    return data


def parse_residue_code(code: str) -> tuple[str, int]:
    """Split a residue code such as ``"W541"`` into ``("W", 541)``.

    Raises
    ------
    ConfigError
        If the code is not one upper-case letter followed by digits.
    """
    m = RESIDUE_CODE.match(str(code).strip())
    if not m:
        raise ConfigError(f"invalid residue code {code!r}; expected e.g. 'W541'")
    return m.group(1), int(m.group(2))


def validate_residue_codes(codes: Iterable[str], sequence: str) -> tuple[list[str], list[dict]]:
    """Check residue codes against a canonical sequence (UniProt numbering).

    Returns
    -------
    (valid, dropped)
        ``valid`` keeps the input order; ``dropped`` holds one dict per
        rejected code with the reason (wrong amino acid or out of range).
    """
    valid, dropped = [], []
    for code in codes:
        aa, n = parse_residue_code(code)
        if n < 1 or n > len(sequence):
            dropped.append({"Residue": code, "Reason": f"position {n} outside sequence (1-{len(sequence)})"})
        elif sequence[n - 1] != aa:
            dropped.append({"Residue": code,
                            "Reason": f"sequence has {sequence[n - 1]}{n}, not {aa}{n}"})
        else:
            valid.append(code)
    return valid, dropped


@dataclass
class Config:
    """Merged configuration with convenience accessors.

    Attributes
    ----------
    raw:
        The complete merged configuration.
    config_dir:
        Folder the target files were read from.
    project_root:
        Base for relative paths in ``paths``.
    sources:
        Files that contributed to this configuration (for the manifest).
    """

    raw: dict
    config_dir: Path
    project_root: Path
    sources: list[str] = field(default_factory=list)

    # ---- generic access -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Return ``raw['a']['b']`` for ``dotted='a.b'`` (``default`` if absent)."""
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def require(self, dotted: str) -> Any:
        """Like :meth:`get` but raise :class:`ConfigError` when the key is missing."""
        sentinel = object()
        val = self.get(dotted, sentinel)
        if val is sentinel:
            raise ConfigError(f"missing configuration key: {dotted}")
        return val

    # ---- frequently used values -----------------------------------------
    @property
    def accession(self) -> str:
        return str(self.require("target.accession")).upper()

    def path(self, key: str) -> Path:
        """Absolute path for ``paths.<key>``; created if it does not exist."""
        p = Path(self.require(f"paths.{key}"))
        p = p if p.is_absolute() else self.project_root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_dir(self) -> Path:
        return self.path("cache")

    @property
    def outputs_dir(self) -> Path:
        return self.path("outputs")

    def output_file(self, key: str) -> Path:
        """Output path for ``outputs.<key>`` with ``{accession}`` filled in."""
        return self.outputs_dir / str(self.require(f"outputs.{key}")).format(accession=self.accession)

    def endpoint(self, name: str) -> str:
        return str(self.require(f"endpoints.{name}"))

    @property
    def ligand_classes(self) -> dict:
        return self.raw["ligand_classes"]

    @property
    def active_site(self) -> dict:
        return self.raw["active_site"]

    @property
    def regions(self) -> dict:
        return self.raw["regions"]

    @property
    def lid(self) -> dict:
        """Entrance-lid settings with ``range`` as a tuple, or ``{}`` if not configured."""
        lid = copy.deepcopy(self.regions.get("entrance_lid") or {})
        if lid:
            lid["range"] = tuple(lid["range"])
            lid["known_states"] = {str(k).upper(): v for k, v in (lid.get("known_states") or {}).items()}
        return lid

    @property
    def flexible_regions(self) -> dict[str, tuple[int, int]]:
        return {k: tuple(v) for k, v in (self.regions.get("flexible_regions") or {}).items()}

    def with_overrides(self, overrides: dict) -> "Config":
        """Return a new Config with ``overrides`` deep-merged into ``raw``."""
        return Config(_deep_merge(self.raw, overrides), self.config_dir,
                      self.project_root, self.sources + ["<overrides>"])


def _check(raw: dict) -> None:
    """Structural validation of the merged configuration."""
    acc = (raw.get("target") or {}).get("accession")
    if not acc or not isinstance(acc, str):
        raise ConfigError("target.accession must be a non-empty string")
    for sect in ("active_site", "regions"):
        sa = (raw.get(sect) or {}).get("accession")
        if sa and str(sa).upper() != acc.upper():
            raise ConfigError(f"{sect} file is for {sa}, but target.accession is {acc}")
    lc = raw.get("ligand_classes") or {}
    for key in ("ions_of_interest", "glycans", "additives", "metals"):
        if not isinstance(lc.get(key), list):
            raise ConfigError(f"ligand_classes.{key} must be a list")
    for ids_key in ("fixed_ids", "extra_ids"):
        for v in (raw.get("validation_set") or {}).get(ids_key) or []:
            if not isinstance(v, str):
                raise ConfigError(f"validation_set.{ids_key} contains {v!r}; quote PDB IDs "
                                  "in YAML (e.g. '1E10' would otherwise be read as a number)")
    for st in ((raw.get("regions") or {}).get("entrance_lid") or {}).get("known_states", {}):
        if not isinstance(st, str):
            raise ConfigError(f"entrance_lid.known_states key {st!r} must be a quoted PDB ID")
    for code in ((raw.get("active_site") or {}).get("literature_sites") or {}):
        parse_residue_code(code)
    policy = (raw.get("structure") or {}).get("model_policy", "first")
    if policy not in ("first", "all"):
        raise ConfigError("structure.model_policy must be 'first' or 'all'")
    for name, mode in ((raw.get("ranking") or {}).get("modes") or {}).items():
        for c in mode.get("criteria") or []:
            if c.get("direction") not in ("lower", "higher"):
                raise ConfigError(f"ranking mode {name}: criterion {c} needs direction lower|higher")
            if float(c.get("weight", 1)) < 0:
                raise ConfigError(f"ranking mode {name}: negative weight in {c}")


def load_config(path: str | Path | None = None,
                overrides: dict | str | Path | None = None) -> Config:
    """Load the main configuration and the target files it references.

    Parameters
    ----------
    path:
        Main YAML file. Defaults to ``config/default_config.yaml``.
    overrides:
        Nested dict, or path to a YAML file, merged on top of the main file
        before the target files are loaded (so ``target.*_file`` can be
        overridden too).

    Raises
    ------
    ConfigError
        On a missing file, a parse error or an inconsistent configuration.
    """
    main = Path(path) if path else DEFAULT_CONFIG_DIR / "default_config.yaml"
    raw = _read_yaml(main)
    sources = [str(main)]
    if overrides is not None:
        if isinstance(overrides, (str, Path)):
            sources.append(str(overrides))
            overrides = _read_yaml(Path(overrides))
        else:
            sources.append("<overrides>")
        raw = _deep_merge(raw, overrides)
    cdir = main.parent
    for key, file_key in (("active_site", "active_site_file"), ("regions", "regions_file"),
                          ("ligand_classes", "ligand_classes_file")):
        fname = (raw.get("target") or {}).get(file_key)
        if not fname:
            raise ConfigError(f"target.{file_key} is not set")
        fpath = Path(fname) if Path(fname).is_absolute() else cdir / fname
        raw[key] = _deep_merge(_read_yaml(fpath), raw.get(key) or {})
        sources.append(str(fpath))
    for key in ("ions_of_interest", "glycans", "additives", "metals", "anions"):
        if key in raw["ligand_classes"]:
            raw["ligand_classes"][key] = [str(x).upper() for x in raw["ligand_classes"][key]]
    _check(raw)
    root = PROJECT_ROOT if main.parent == DEFAULT_CONFIG_DIR else main.parent.parent
    return Config(raw=raw, config_dir=cdir, project_root=root, sources=sources)
