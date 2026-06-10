#!/usr/bin/env bash
# Daily local paper-trade loop. DRY-RUN by default; pass --live to submit orders.
#   ./scripts/paper_trade.sh            # generate book + connect + dry-run
#   ./scripts/paper_trade.sh --live     # ...and actually submit to Alpaca paper
# Secrets come from .env (gitignored) or your shell environment -- never committed.
set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env if present (keeps the secret on your machine only).
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${APCA_API_KEY_ID:?set APCA_API_KEY_ID (e.g. in .env -- see .env.example)}"
: "${APCA_API_SECRET_KEY:?set APCA_API_SECRET_KEY (NEVER commit this)}"
export APCA_API_BASE_URL="${APCA_API_BASE_URL:-https://paper-api.alpaca.markets}"

LIVE=""; [ "${1:-}" = "--live" ] && LIVE="--live"
RISK="${RISK:-1.0}"   # %% equity risk per trade (the validated 1% default)

echo "== 1/6  verify Alpaca paper connection =="
python -m training.alpaca_exec --selftest

echo "== 2/6  reconcile BEFORE trading (drift blocks new entries) =="
# If broker state has drifted from the logs (unmatched fills, unmanaged
# positions), stop here instead of stacking new trades on a corrupt record.
python -m training.reconcile --strict

echo "== 3/6  manage exits on OPEN positions (sell winners/laggards per the rules) =="
python -m training.manage_exits $LIVE

echo "== 4/6  generate today's book (FRESH data) =="
python -m training.live_signals --refresh --top 10 --risk "$RISK"

echo "== 5/6  ${LIVE:-(dry-run -- no orders sent)}  submit new entries =="
python -m training.alpaca_exec --from-log --risk "$RISK" $LIVE

echo "== 6/6  fill quality + final reconcile =="
python -m training.fill_quality || true
python -m training.reconcile || true

echo "== daily ops report =="
python -m training.daily_report || true
