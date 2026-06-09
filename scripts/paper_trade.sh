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

echo "== 1/3  generate today's book (FRESH data) =="
python -m training.live_signals --refresh --top 10 --risk "$RISK"

echo "== 2/3  verify Alpaca paper connection =="
python -m training.alpaca_exec --selftest

echo "== 3/3  ${LIVE:-(dry-run -- no orders sent)}  submit =="
python -m training.alpaca_exec --from-log --risk "$RISK" $LIVE
