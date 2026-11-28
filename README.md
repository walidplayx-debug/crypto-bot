# 📡 crypto-bot

**English** · [العربية](README.ar.md)

A signal bot that ranks **18 large coins** (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, TRX, LINK, AVAX, DOT, LTC, BCH, ATOM, ETC, XLM, UNI, NEAR) with a **coin-rotation (momentum)** strategy and publishes its signal on a web page once a day, after the daily candle closes.

📊 **Live signal:** https://walidplayx-debug.github.io/crypto-bot/ · summary in [`RESULTS.md`](RESULTS.md)

## The strategy

Every few days the coins are ranked by how much they rose over the last N days:

- **buy the strongest** (only if their momentum is positive),
- **short the weakest** (only if its momentum is negative),
- hold until the next review; no leverage, no stop-loss.

The idea: coins that are rising tend to keep rising for a while, and falling coins keep falling (momentum).

Three optional improvements are part of the search: **volatility sizing** (calmer coins get bigger weights), a **volume universe** (only the 10 coins with the most trading volume over the last 30 days, which also reduces survivorship bias), and an **emergency stop** (close a position that is down 15% or 25%, kept only if it clearly helps). The page shows what each one did.

The settings were chosen from 2,160 combinations using **training (until 2023-01-01) and validation (until 2024-07-01) data only**. The hidden test after that was never used for any decision. On the go-live day the settings are **frozen** in [`docs/live.json`](docs/live.json), and the live result only counts days after that day: a true paper-trading test.

## No cheating

- Positions are decided on a day's **close** and held from the **next day's open**.
- Every change of position pays the 0.05% taker fee plus 0.03% slippage, and a funding cost is charged for every hour held.
- The strategy was also run on completely random prices and found nothing profitable, which confirms it can't see the future.

## How it runs

GitHub Actions ([`.github/workflows/evolve.yml`](.github/workflows/evolve.yml)) runs once a day after the daily close (00:00 UTC): it downloads the new daily candles from Binance (free, no API key), computes the signal, and updates the page. Because GitHub's schedule is often late, it has several backup times that day and skips itself once the day's signal is done.

The coin list lives in [`bot/config.py`](bot/config.py) (`ROTATION_SYMBOLS`). Changing it starts a new live test. Note: the list holds coins that are big *today*, so the backtest has some survivorship bias (coins that died are missing); the live test does not.

```bash
pip install -r requirements.txt
python -m bot.fetch     # download / update daily candles
python -m bot.signals   # compute the signal, write docs/ and RESULTS.md
```

## Research history

Before the rotation strategy, a **genetic algorithm** searched indicator-rule strategies (26 indicators, 3 timeframes, limit orders, worst-year scoring, 3 independent searches per strategy, loss analysis). After about 80 runs it found nothing that held up on the hidden test, so it was stopped. Its code is kept in [`research/`](research/) (`python -m bot.fetch --hourly`, then `python -m research.evolve`, `python -m research.report`; note that the report overwrites `docs/data.json`).

> ⚠️ This is a learning and experimentation project. Results on historical data and paper trading are no guarantee of future performance, and nothing here is financial advice.
