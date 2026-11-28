# 📡 crypto-bot

**English** · [العربية](README.ar.md)

A signal bot for five coins (**BTC · ETH · SOL · BNB · DOGE**) that runs a **coin-rotation (momentum)** strategy and publishes its live signal on a web page, updated every hour.

📊 **Live signal:** https://walidplayx-debug.github.io/crypto-bot/ · summary in [`RESULTS.md`](RESULTS.md)

## The strategy

Every few days the coins are ranked by how much they rose over the last N days:

- **buy the strongest** (only if their momentum is positive),
- **short the weakest** (only if its momentum is negative),
- hold until the next review; no leverage, no stop-loss.

The idea: coins that are rising tend to keep rising for a while, and falling coins keep falling (momentum).

The settings were chosen from 240 combinations using **training (until 2023-01-01) and validation (until 2024-07-01) data only**. The hidden test after that was never used for any decision. On the go-live day the settings are **frozen** in [`docs/live.json`](docs/live.json), and the live result only counts days after that day: a true paper-trading test.

## No cheating

- Positions are decided on a day's **close** and held from the **next day's open**.
- Every change of position pays the 0.05% taker fee plus 0.03% slippage, and a funding cost is charged for every hour held.
- The strategy was also run on completely random prices and found nothing profitable, which confirms it can't see the future.

## How it runs

GitHub Actions ([`.github/workflows/evolve.yml`](.github/workflows/evolve.yml)) runs every hour: it downloads new candles from Binance (free, no API key), computes the signal, and updates the page. Each run starts the next one, because GitHub's own schedule is often hours late.

```bash
pip install -r requirements.txt
python -m bot.fetch     # download / update candles
python -m bot.signals   # compute the signal, write docs/ and RESULTS.md
```

## Research history

Before the rotation strategy, a **genetic algorithm** searched indicator-rule strategies (26 indicators, 3 timeframes, limit orders, worst-year scoring, 3 independent searches per strategy, loss analysis). After about 80 runs it found nothing that held up on the hidden test, so it was stopped. Its code is kept in [`research/`](research/) (`python -m research.evolve`, `python -m research.report`; note that the report overwrites `docs/data.json`).

> ⚠️ This is a learning and experimentation project. Results on historical data and paper trading are no guarantee of future performance, and nothing here is financial advice.
