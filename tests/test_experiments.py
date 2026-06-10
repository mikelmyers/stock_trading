"""Experiment-registry tests: append-only ledger, trial counting, rendering."""

import json

import pytest

import training.experiments as ex


def test_log_and_count(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RENDERED", tmp_path / "EXPERIMENTS.md")
    ledger = tmp_path / "ledger.jsonl"
    ex.log_experiment("baseline", {"objective": "win"}, "adopt",
                      metric="r=+0.0278", ledger=ledger)
    ex.log_experiment("si features", {"objective": "win", "feat": "+si"},
                      "reject", ledger=ledger)
    rows = ex.load(ledger)
    assert len(rows) == 2 and ex.trial_count(ledger) == 2
    assert rows[0]["config_hash"] != rows[1]["config_hash"]


def test_invalid_verdict_rejected(tmp_path):
    with pytest.raises(ValueError):
        ex.log_experiment("x", {}, "maybe", ledger=tmp_path / "l.jsonl")


def test_render_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RENDERED", tmp_path / "EXPERIMENTS.md")
    ledger = tmp_path / "ledger.jsonl"
    ex.log_experiment("trial one", {}, "inconclusive", notes="needs walk-forward",
                      ledger=ledger)
    text = (tmp_path / "EXPERIMENTS.md").read_text()
    assert "trial one" in text and "1 experiments logged" in text


def test_dataset_hash_detects_change(tmp_path):
    d = tmp_path / "data.parquet"
    d.write_bytes(b"abc" * 100)
    h1 = ex._dataset_hash(d)
    d.write_bytes(b"xyz" * 100)
    assert ex._dataset_hash(d) != h1
    assert ex._dataset_hash(tmp_path / "nope.parquet").startswith("missing:")


def test_cli_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ex, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(ex, "RENDERED", tmp_path / "EXPERIMENTS.md")
    ex.main(["log", "--name", "cli test", "--verdict", "adopt",
             "--config", json.dumps({"a": 1})])
    ex.main(["list"])
    out = capsys.readouterr().out
    assert "cli test" in out and "1 total trials" in out
