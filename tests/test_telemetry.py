"""Tests for the paper-trade telemetry: fill-quality ledger, paginated order
fetch, reconciliation drift detection, and the daily report — all against
faked broker responses (no network)."""

import pandas as pd
import pytest

from training.fill_quality import (
    compute_fill_quality,
    fetch_all_orders,
    summarize,
    update_ledger,
)


def _order(oid, sym, fill, qty=10, leg=None, submitted="2026-06-10T14:00:00Z"):
    return {"id": oid, "symbol": sym, "side": "buy", "filled_avg_price": str(fill),
            "filled_qty": str(qty), "filled_at": submitted, "submitted_at": submitted,
            "legs": [leg] if leg else []}


def _executions(rows):
    return pd.DataFrame(rows, columns=["order_id", "ticker", "submitted_at",
                                       "qty", "entry_signal", "stop"])


class TestFillQuality:
    def test_slippage_in_r_and_pct(self):
        # signal 100, stop 95 (risk 5), filled at 100.50 -> +0.10R, +0.5%
        ex = _executions([("o1", "AAA", "2026-06-10T14:00:00Z", 10, 100.0, 95.0)])
        fq = compute_fill_quality(ex, [_order("o1", "AAA", 100.50)])
        assert fq.iloc[0]["slippage_r"] == pytest.approx(0.10)
        assert fq.iloc[0]["slippage_pct"] == pytest.approx(0.5)

    def test_unfilled_orders_skipped(self):
        ex = _executions([("o1", "AAA", "t", 10, 100.0, 95.0)])
        order = _order("o1", "AAA", 100.0)
        order["filled_avg_price"] = None
        assert compute_fill_quality(ex, [order]).empty

    def test_ledger_update_is_idempotent(self, tmp_path):
        ex = _executions([("o1", "AAA", "t", 10, 100.0, 95.0)])
        fq = compute_fill_quality(ex, [_order("o1", "AAA", 100.2)])
        ledger = tmp_path / "fq.csv"
        update_ledger(fq, ledger)
        merged = update_ledger(fq, ledger)   # second run must not duplicate
        assert len(merged) == 1

    def test_summary_flags_excess_cost(self):
        # +0.05R entry slippage -> ~0.10R round trip >> 0.019R modeled
        ex = _executions([("o1", "AAA", "t", 10, 100.0, 95.0)])
        fq = compute_fill_quality(ex, [_order("o1", "AAA", 100.25)])
        s = summarize(fq)
        assert "EXCEEDS" in s["verdict"]


class TestFetchAllOrders:
    def test_paginates_past_the_500_cap(self):
        # 1200 orders served in pages of 500: the old single call lost 700.
        all_orders = [_order(f"o{i}", "AAA", 100.0,
                             submitted=f"2026-06-{10 - i // 200:02d}T{23 - (i % 200) // 10:02d}:00:00Z")
                      for i in range(1200)]

        def fake_get(path, **params):
            until = params.get("until")
            pool = [o for o in all_orders if not until or o["submitted_at"] < until]
            return pool[:params["limit"]]

        got = fetch_all_orders(fake_get, page_limit=500)
        assert len(got) == 1200

    def test_single_page_stops(self):
        calls = []

        def fake_get(path, **params):
            calls.append(1)
            return [_order("o1", "AAA", 100.0)]

        assert len(fetch_all_orders(fake_get)) == 1
        assert len(calls) == 1


class TestReconcileDrift:
    def _fake_get(self, orders, positions):
        def get(path, **params):
            if path == "/v2/orders":
                return orders
            if path == "/v2/positions":
                return positions
            raise AssertionError(path)
        return get

    def test_orphan_position_detected(self, monkeypatch, tmp_path):
        import training.reconcile as rec
        monkeypatch.setattr(rec, "EXEC_LOG", tmp_path / "none.csv")
        monkeypatch.setattr(rec, "LOG", tmp_path / "none2.csv")
        monkeypatch.setattr("training.alpaca_exec.EXEC_LOG", tmp_path / "none.csv")
        g = rec.gather(self._fake_get(
            [], [{"symbol": "ZZZZ", "current_price": "5.00"}]))
        assert g["orphan_positions"] == ["ZZZZ"]
        assert rec.reconcile(self._fake_get(
            [], [{"symbol": "ZZZZ", "current_price": "5.00"}]), strict=True) == 1

    def test_clean_state_passes_strict(self, monkeypatch, tmp_path):
        import training.reconcile as rec
        ex = tmp_path / "executions.csv"
        pd.DataFrame([{"order_id": "o1", "ticker": "AAA", "stop": 95.0,
                       "model_p": 0.6}]).to_csv(ex, index=False)
        monkeypatch.setattr(rec, "EXEC_LOG", ex)
        monkeypatch.setattr(rec, "LOG", tmp_path / "none.csv")
        sell = {"side": "sell", "status": "filled", "filled_avg_price": "104.0"}
        orders = [_order("o1", "AAA", 100.0, leg=sell)]
        assert rec.reconcile(self._fake_get(orders, []), strict=True) == 0
        g = rec.gather(self._fake_get(orders, []))
        assert g["closed"][0][3] == pytest.approx(0.8)   # (104-100)/5


def test_daily_report_composes(monkeypatch, tmp_path):
    import training.daily_report as dr
    import training.reconcile as rec
    monkeypatch.setattr(rec, "EXEC_LOG", tmp_path / "none.csv")
    monkeypatch.setattr(rec, "LOG", tmp_path / "none2.csv")
    monkeypatch.setattr(dr, "FQ_LEDGER", tmp_path / "nofq.csv")

    def fake_get(path, **params):
        if path == "/v2/account":
            return {"equity": "10000", "buying_power": "10000", "status": "ACTIVE"}
        if path == "/v2/orders":
            return []
        if path == "/v2/positions":
            return []
        raise AssertionError(path)

    text = dr.build_report(fake_get)
    assert "DAILY OPS REPORT" in text
    assert "logs and broker agree" in text
    assert "no closed trades yet" in text
