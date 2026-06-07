# Swing Trading Research Agent

A decision-support agent for swing trading breakouts. It does **not** place trades. It scans the market, scores setups, sizes positions, generates exit plans, tracks your positions daily, and builds a trust score over time so risk can scale up gradually.

Built around the **Core Breakout** pattern: volatility compression → resistance ceiling → volume breakout.

---

## What It Does

| Agent | Role |
|-------|------|
| **Scout** | Detects squeeze + ceiling + volume breakout setups |
| **Context** | Checks relative strength vs SPY and sector ETF |
| **Probability** | Backtests historical breakouts on the same ticker |
| **Risk** | Sizes positions to a strict max-risk cap ($10 to start) |
| **Exit Manager** | Hard stops, scale-out ladder, ATR trailing, time stops |
| **Teacher** | Post-trade review and trust score calibration |
| **Orchestrator** | Combines all agents into one trade sheet |

---

## Requirements

- Python 3.10+
- Internet access (pulls data from Yahoo Finance via `yfinance`)

---

## Installation

```powershell
cd C:\Users\moder\trading_agent
pip install -r requirements.txt
```

---

## Quick Start

```powershell
# 1. Scan the market for breakouts
python cli.py scan

# 2. Deep-dive one ticker
python cli.py analyze NVDA

# 3. After YOU enter the trade in your broker, tell the agent to track it
python cli.py track NVDA

# 4. Run daily (after market close) to check exits
python cli.py monitor

# 5. Check your stats
python cli.py status
```

---

## All Commands

### `scan` — Find breakouts across the watchlist

Scans large, mid, and small caps plus any tickers in `universe.txt`. Returns only setups that pass all breakout criteria.

```powershell
python cli.py scan
```

---

### `analyze TICKER` — Full trade sheet on one stock

Runs all agents and prints a complete report: setup score, market context, historical probability, position sizing, and exit plan.

```powershell
python cli.py analyze AMD
python cli.py analyze AMD --save    # also saves to reports/
```

---

### `sheet TICKER` — Generate and save a trade sheet

Same as analyze, but always saves JSON + TXT to the `reports/` folder.

```powershell
python cli.py sheet PLTR
```

---

### `track TICKER` — Start tracking a live position

**You** execute the trade in your broker. The agent only tracks it and tells you when to exit.

**Auto mode** (uses current breakout analysis):

```powershell
python cli.py track NVDA
```

**Manual mode** (you entered at a different price):

```powershell
python cli.py track NVDA --entry 130.50 --stop 127.00 --shares 2.85
```

If you omit `--shares`, the agent calculates shares to keep risk within your trust tier cap.

---

### `monitor` — Daily exit check

Run this once per day after the market closes. The agent checks every open position for:

1. Hard stop hit
2. Trailing stop hit (active after 1R or first scale-out)
3. Scale-out targets (33% at 1R, 33% at 2R)
4. Time stops (no follow-through by day 7/10)
5. Max hold (14 days)

```powershell
python cli.py monitor
```

When a position closes, the agent prompts you to rate the trade (see `review` below).

---

### `status` — Portfolio and trust overview

```powershell
python cli.py status
```

Shows trust score, risk tier, open positions, recent closed trades, and total P&L.

---

### `close TICKER` — Manually close a position

Use when you exit before the agent signals it.

```powershell
python cli.py close NVDA --price 135.20
python cli.py close NVDA --price 135.20 --reason EARLY_EXIT
```

---

### `review TICKER` — Rate a closed trade

Your feedback feeds the trust score. Run after every closed trade.

```powershell
python cli.py review NVDA --fidelity 0.9 --aligned yes --notes "Clean breakout, followed plan"
python cli.py review AMD --fidelity 0.5 --aligned no --notes "Exited early out of fear"
```

| Flag | Meaning |
|------|---------|
| `--fidelity` | How clean was the setup? `0.0` (terrible) to `1.0` (textbook) |
| `--aligned` | Did you follow the exit plan? `yes` or `no` |
| `--notes` | Optional free-text notes |

---

### `history` — Full trade log

```powershell
python cli.py history
```

---

### `full` — Scan + monitor in one pass

```powershell
python cli.py full
```

---

## Daily Workflow

This is the intended loop once you are live:

```
Morning (optional)          After market close
─────────────────          ──────────────────
python cli.py scan         python cli.py monitor
python cli.py analyze X    python cli.py status
```

### Step-by-step: taking a trade

1. **Find a setup**
   ```powershell
   python cli.py scan
   ```

2. **Review the full trade sheet**
   ```powershell
   python cli.py analyze SOUN --save
   ```
   Read the report. Check recommendation (`TRADE`, `WATCH`, or `PASS`), composite score, and exit plan.

3. **Execute in your broker** (Robinhood, E*Trade, etc.)
   - Buy the shares shown in the trade sheet
   - Set a mental or broker stop at the stop-loss price
   - Do NOT risk more than the `Actual risk` line shows

4. **Tell the agent you entered**
   ```powershell
   python cli.py track SOUN
   ```
   Or with manual prices if your fill differed:
   ```powershell
   python cli.py track SOUN --entry 5.12 --stop 4.75
   ```

5. **Check daily**
   ```powershell
   python cli.py monitor
   ```
   - `[HOLD]` — stay in the trade
   - `[SCALE OUT]` — sell the indicated shares at the target
   - `[EXIT]` — close the remaining position

6. **If you exit manually**, tell the agent:
   ```powershell
   python cli.py close SOUN --price 5.85
   ```

7. **Rate the trade** (builds trust score):
   ```powershell
   python cli.py review SOUN --fidelity 0.85 --aligned yes
   ```

---

## Trust Score and Risk Tiers

The agent starts at **0% trust** with a **$10 max risk** per trade. As you complete trades and submit honest feedback, the trust score rises and risk caps increase.

| Trust Score | Tier | Max Risk/Trade |
|-------------|------|----------------|
| 0–29% | Sandbox | $10 |
| 30–59% | Learning | $25 |
| 60–79% | Trusted | $50 |
| 80–100% | Proven | $100 |

Trust is calculated from three pillars:
- **Setup fidelity** — how clean your rated setups are
- **Execution alignment** — whether you followed the exit plan
- **Expectancy** — win rate and average R-multiple

---

## Exit Plan Rules

Every tracked trade gets a pre-defined management sheet:

| Rule | Detail |
|------|--------|
| Hard stop | 2% below broken resistance |
| Scale-out 1 | Sell 33% of position at 1R |
| Scale-out 2 | Sell 33% at 2R |
| Runner trail | Remaining 34% trails at 2× ATR below high |
| Time stop | Exit if <25% of max profit by day 10 |
| Max hold | 14 days |

---

## Configuration

Edit `config.py` to customize:

- `MAX_RISK_PER_TRADE` — starting sandbox cap ($10)
- `RISK_TIERS` — trust score → max risk mapping
- `WATCHLIST` — default tickers by market cap
- `SCALE_OUT_LEVELS` — profit-taking ladder
- `TRAILING_STOP_ATR_MULT` — runner trail multiplier
- `MAX_HOLDING_DAYS` / `TIME_STOP_DAYS` — time rules

### Adding more tickers

Edit `universe.txt` (one ticker per line):

```
COIN,Mid Cap
MARA,Small Cap
SMCI,Mega/Large Cap
```

---

## Project Structure

```
trading_agent/
├── cli.py              # Command-line interface (main entry point)
├── core.py             # Business logic
├── config.py           # Settings and watchlist
├── data.py             # Yahoo Finance data layer
├── state.py            # Position tracking + trust score (trade_state.json)
├── reports.py          # Trade sheet formatting and export
├── universe.txt        # Extra tickers to scan
├── trade_state.json    # Auto-created: your positions and history
├── reports/            # Auto-created: saved trade sheets
└── agents/
    ├── scout.py        # Breakout detection
    ├── context.py      # Market/sector tailwind
    ├── probability.py  # Historical edge estimation
    ├── risk.py         # Position sizing
    ├── exit_manager.py # Exit and scale-out engine
    ├── teacher.py      # Post-trade review
    └── orchestrator.py # Combines all agents
```

---

## Important Notes

- **This is research tooling, not financial advice.** Probabilities are based on historical patterns, not guarantees.
- **You execute all trades manually.** The agent never connects to a broker.
- **yfinance data can lag or fail.** If a ticker errors, retry later or check the symbol.
- **Run `monitor` after market close** (4:00 PM ET) for accurate daily candle data.
- **Always rate closed trades** with `review` — this is how the agent learns your edge.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named ...` | Run `pip install -r requirements.txt` from the project folder |
| `Cannot auto-track` | Setup criteria not met — use `--entry --stop --shares` for manual tracking |
| `Already tracking` | Run `close` first, then `track` again |
| `database is locked` (yfinance) | Transient error — re-run the command |
| No breakouts found | Normal — breakouts are episodic, not daily |

---

## Training the Agent

This system has **two layers of learning**. They work together but train different things.

### Setup Types (expand when one pattern loses)

The agent supports multiple setup patterns. Each is trained independently — losing patterns get **disabled** automatically.

| Setup | File | Pattern |
|-------|------|---------|
| Core Breakout | `agents/setups/breakout.py` | Squeeze → resistance → volume surge |
| MA Pullback | `agents/setups/ma_pullback.py` | Uptrend → dip to 21 EMA → bounce |

To add a new setup: create `agents/setups/your_setup.py`, register it in `agents/setups/registry.py`, re-run `train`.

### Layer 1: Mass Historical Simulation (`train`)

Runs thousands of backtests on **real Yahoo Finance OHLCV data** — not synthetic prices, not an LLM fine-tune.

```powershell
# 10,000 simulations (default)
python cli.py train

# 50,000 simulations, 5 years of history
python cli.py train --simulations 50000 --years 5

# 100,000 simulations with parallel workers
python cli.py train -n 100000 --years 5 --workers 8

# Only real historical setups (no bootstrap resampling)
python cli.py train -n 5000 --no-bootstrap
```

#### How it works

```
Phase 1   Download real daily OHLCV for ~68 liquid tickers (3-5 years)
Phase 2   Walk every day in history → find actual breakout setups
          → simulate forward with real subsequent bars + exit rules
Phase 3   Bootstrap resample to reach 10k/50k/100k (adds slippage noise)
Phase 4   Calibrate thresholds → save learned_params.json
```

| Output | What it teaches |
|--------|-----------------|
| `learned_params.json` | Min setup score, composite cutoff, expectancy threshold |
| `training/results/*.json` | Full stats: win rate, drawdown, score edge map |
| Orchestrator at runtime | Uses learned cutoffs for TRADE/WATCH/PASS decisions |

#### What 10k vs 100k means

| Simulations | Real setups (typical) | Bootstrapped | Best for |
|-------------|----------------------|--------------|----------|
| 1,000 | ~25-50 | rest | Quick test |
| 10,000 | ~25-50 | ~9,950 | Default calibration |
| 50,000 | ~25-50 | ~49,950 | Stable threshold tuning |
| 100,000 | ~25-50 | ~99,950 | Stress-test robustness |

**Important:** With strict breakout rules, only ~25-50 *real* setups appear per 2-3 years across 68 tickers. The bootstrap phase resamples those real outcomes with slippage noise to reach 10k+. This is statistically valid for calibration but is not 10k independent market events.

To get more real (non-bootstrapped) samples:
- Increase `--years 5`
- Add tickers to `universe.txt`
- Loosen rules in `config.py` (not recommended without testing)

#### Re-train after changes

Re-run `train` whenever you change:
- Breakout rules in `agents/scout.py`
- Exit rules in `agents/exit_manager.py`
- Watchlist or universe

### Layer 2: Live Trust Score (`review`)

Your real trades + honest feedback train the **risk scaling** system:

```powershell
python cli.py review NVDA --fidelity 0.9 --aligned yes
```

This does NOT change breakout detection. It controls how much money the Risk Agent allows per trade ($10 → $100).

### Is this machine learning?

**No neural network training.** The agents are deterministic Python rules. "Training" means:

1. **Calibrating thresholds** — what setup score actually predicts wins on real data
2. **Building statistics** — win rate, expectancy, per-ticker edge
3. **Earning trust** — your live trade feedback unlocks larger position sizes

---

## Roadmap (not yet implemented)

- Options swing setups (defined-risk spreads)
- Intraday entry timing (15m/1h charts for swing entries)
- LangGraph orchestration loop with critique/revise cycle
- Broker CSV import for automatic fill tracking