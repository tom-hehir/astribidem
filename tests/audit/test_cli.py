"""The ``astribidem`` command line: audit, audit-sweep and summary."""

import dataclasses
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from astribidem import (
    EntitywiseCrossmatchConfig,
    build_edges,
    match_uids,
    resolve,
    write_edges,
)
from astribidem.audit import audit_edges, restrict_radius
from astribidem.audit.cli import main


@pytest.fixture(scope="module")
def edges():
    rng = np.random.default_rng(0)
    side = 60 / 3600
    surveys = {
        name: (150 + rng.uniform(0, side, n), rng.uniform(0, side, n))
        for name, n in (("a", 150), ("b", 100), ("c", 60))
    }
    return build_edges(
        surveys,
        radius_arcsec=2.0,
        dedupe_radius_arcsec=1.0,
        radius_arcsec_overrides={("a", "c"): 3.0},
    )


@pytest.fixture
def edges_dir(edges, tmp_path):
    write_edges(edges, tmp_path / "edges")
    return tmp_path / "edges"


def run(capsys, *argv):
    assert main([str(arg) for arg in argv]) == 0
    return capsys.readouterr().out


def test_audit_prints_tables_and_saves_the_audit(capsys, edges, edges_dir, tmp_path):
    out = run(capsys, "audit", edges_dir, "--save", tmp_path / "audit.json")
    for title in ("Surveys", "Pairs", "Rows with candidates", "Component sizes"):
        assert title in out
    saved = json.loads((tmp_path / "audit.json").read_text())
    assert saved == json.loads(json.dumps(dataclasses.asdict(audit_edges(edges))))


def test_audit_narrows_the_radius(capsys, edges, edges_dir, tmp_path):
    run(
        capsys,
        "audit",
        edges_dir,
        "--radius-override",
        "a+b=0.5",
        "--save",
        tmp_path / "audit.json",
    )
    saved = json.loads((tmp_path / "audit.json").read_text())
    expected = audit_edges(
        restrict_radius(edges, 2.0, {("a", "b"): 0.5, ("a", "c"): 3.0})
    )
    assert saved == json.loads(json.dumps(dataclasses.asdict(expected)))


def test_audit_sweep_audits_each_radius(capsys, edges, edges_dir, tmp_path):
    out = run(
        capsys,
        "audit-sweep",
        edges_dir,
        "--radius",
        "1",
        "0.5",
        "--radius-override",
        "a+c=2.5",
        "--save",
        tmp_path / "sweep.json",
    )
    assert "median sep" in out
    saved = json.loads((tmp_path / "sweep.json").read_text())
    expected = [
        audit_edges(restrict_radius(edges, radius, {("a", "c"): 2.5}))
        for radius in (0.5, 1.0)
    ]
    assert saved == json.loads(
        json.dumps([dataclasses.asdict(audit) for audit in expected])
    )
    # The held pair appears once, not once per radius.
    assert sum(line.startswith("a+c ") for line in out.splitlines()) == 3


def test_summary_reads_the_recorded_settings(capsys, edges, tmp_path):
    index = resolve(edges, EntitywiseCrossmatchConfig())
    pq.write_table(index, tmp_path / "index.parquet")
    out = run(capsys, "summary", tmp_path / "index.parquet")
    assert f"Rows: {index.num_rows:,}" in out
    for title in ("Summary", "Settings", "Dedupe"):
        assert f"\n{title}\n" in out
    assert "EntitywiseCrossmatchConfig" in out


def test_summary_of_a_uid_index(capsys, tmp_path):
    index = match_uids({"a": [1, 2, 3], "b": [3, 1]}, join="inner")
    pq.write_table(index, tmp_path / "uids.parquet")
    out = run(capsys, "summary", tmp_path / "uids.parquet")
    assert "exact_uid" in out and "Summary" not in out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["audit", "{edges}", "--radius", "5"], "exceeds the built radius"),
        (["audit", "{edges}", "--radius-override", "a+z=1"], "invalid survey pair"),
        (["audit", "{missing}"], "No such file"),
    ],
)
def test_errors_exit_with_a_message(capsys, edges_dir, tmp_path, argv, message):
    paths = {"edges": edges_dir, "missing": tmp_path / "missing"}
    with pytest.raises(SystemExit) as exit_info:
        main([arg.format(**paths) for arg in argv])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_malformed_override_is_a_usage_error(capsys, edges_dir):
    with pytest.raises(SystemExit) as exit_info:
        main(["audit", str(edges_dir), "--radius-override", "a=1"])
    assert exit_info.value.code == 2
    assert "SURVEY+SURVEY=RADIUS" in capsys.readouterr().err
