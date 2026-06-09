#!/usr/bin/env bash
# Runner system intraday loop (low-float momentum). DRY-RUN by default.
#   ./scripts/runner.sh                 # live data, DRY-RUN loop (no orders)
#   ./scripts/runner.sh --live          # paper-submit loop
# Secrets from .env (gitignored). Needs the Alpaca SIP feed for reliable low-float data.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${APCA_API_KEY_ID:?set APCA_API_KEY_ID (.env)}"
: "${APCA_API_SECRET_KEY:?set APCA_API_SECRET_KEY (NEVER commit)}"
export APCA_API_BASE_URL="${APCA_API_BASE_URL:-https://paper-api.alpaca.markets}"

LIVE=""; [ "${1:-}" = "--live" ] && LIVE="--live"
EQUITY="${EQUITY:-500}"; LEVERAGE="${LEVERAGE:-1.0}"; EPISODE="${EPISODE:-ep001}"; INTERVAL="${INTERVAL:-120}"

python -m runner.run --loop $LIVE --equity "$EQUITY" --leverage "$LEVERAGE" \
  --episode "$EPISODE" --interval "$INTERVAL"
