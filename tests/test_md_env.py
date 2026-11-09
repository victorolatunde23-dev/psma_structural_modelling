"""The MD-environment runner: notebook 05 runs scripts/run_md_setup.py with a separate Python."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from psma_pipeline import md_env
from test_system_build_offline import FAKE_PACKMOL, FAKE_TLEAP_SYS, prepared_with_glycan, zinc_dir

REPO = Path(__file__).resolve().parents[1]


def fake_env(tmp_path: Path) -> Path:
    """An 'environment' whose bin/ holds this Python and fake tleap / packmol-memgen / MCPB.py."""
    prefix = tmp_path / "env"
    b = prefix / "bin"
    b.mkdir(parents=True)
    (b / "python").symlink_to(sys.executable)
    paths = [str(REPO / "tests"), str(REPO / "src")]
    for name, body in (("packmol-memgen", FAKE_PACKMOL), ("tleap", FAKE_TLEAP_SYS), ("MCPB.py", "#!{python}\n")):
        p = b / name
        p.write_text(body.replace("{python}", sys.executable).replace("{paths!r}", repr(paths)))
        p.chmod(0o755)
    return prefix


def test_run_stage_system_round_trip(tmp_path, monkeypatch):
    prefix = fake_env(tmp_path)
    assert md_env.env_ready(str(prefix))
    out = tmp_path / "outputs"
    zinc_dir(out / "md_setup" / "zinc_mcpb")
    prep = prepared_with_glycan(tmp_path / "microstate_00_default.pdb")
    monkeypatch.setenv("PATH", os.environ["PATH"])
    res = md_env.run_stage("system", out, tmp_path / "cache", repo=REPO, prefix=str(prefix),
                           extra=["--prepared", str(prep), "--no-md", "--tag", "t"])
    t = res["tables"]
    assert res["returncode"] == 0 and res["extra"]["critical_ok"]      # NOT RUN is not a critical FAIL
    assert t["SystemChecks"].set_index("Check").loc["tleap_build", "Status"] == "PASS"
    assert t["MDReadiness"].iloc[-1]["Item"] == "OVERALL" and "Issues" in t
    assert Path(res["extra"]["saved"]["workbook"]).exists()


def test_run_stage_reports_errors(tmp_path):
    prefix = fake_env(tmp_path)
    res = md_env.run_stage("zinc", tmp_path / "outputs", repo=REPO, prefix=str(prefix),
                           extra=["--prepared", str(tmp_path / "missing.pdb")])
    assert res["returncode"] == 1


def test_local_machine_defaults(tmp_path, monkeypatch):
    import platform as pf
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setenv("PSMA_MDENV", str(tmp_path / "env"))
    assert not md_env.in_colab() and md_env.default_prefix() == str(tmp_path / "env")
    monkeypatch.setattr(pf, "system", lambda: "Windows")
    import pytest
    with pytest.raises(RuntimeError, match="WSL2"):
        md_env.mamba_platform()
    monkeypatch.setattr(pf, "system", lambda: "Darwin")
    monkeypatch.setattr(pf, "machine", lambda: "arm64")
    assert md_env.mamba_platform() == "osx-arm64"


def test_local_overrides(tmp_path):
    from psma_pipeline.colab import local_overrides
    import pytest
    assert local_overrides(None) == {}
    with pytest.raises(FileNotFoundError, match="outputs"):
        local_overrides(tmp_path)
    (tmp_path / "outputs").mkdir()
    o = local_overrides(tmp_path)
    assert o["paths"]["outputs"] == str(tmp_path / "outputs") and (tmp_path / "cache").is_dir()
