# 🤖 crypto-bot

**English** · [العربية](README.ar.md)

A bot that teaches itself trading strategies with a **genetic algorithm**, on historical data for five coins:
**BTC · ETH · SOL · BNB · DOGE**.

We never tell it how to trade. It gets the indicators, invents strategies, tests them, keeps the best, then mixes and mutates them generation after generation.

📊 **Results:** [`RESULTS.md`](RESULTS.md) and the [live results page](https://walidplayx-debug.github.io/crypto-bot/) (GitHub Pages)

## How it works

1. **Data:** 1-hour candles from Binance (free, no API key). 4-hour candles are built from them.
2. **Indicators (26):** RSI, MACD, Bollinger, EMA 20/50/200, ATR, ADX, Stochastic, CCI, MFI, OBV, volume, Donchian channels, volatility…
3. **Market filters (learned from the loss analysis):** a strategy may switch on "only trade with the EMA 200 trend" and/or "skip low-volatility markets".
4. **A strategy:** up to 2 long conditions and up to 2 short conditions, plus a stop-loss and take-profit in ATR multiples, and a maximum trade duration.
4. **Three timeframes:** 1h, 4h, and 1h with a 4h trend filter.
5. **Two kinds of strategy:**
   - **Universal:** one strategy that has to work on all five coins at once. Harder to find, but the strongest evidence that it isn't luck.
   - **Per coin:** a separate strategy for each coin.

## No cheating

- Decisions are made on a candle's **close**; the trade enters at the **next candle's open**.
- If the stop and the target are both hit inside the same candle, it **counts as a loss**.
- If price gaps through the stop, the worse price is used.
- Entries are **limit orders**: placed at (or better than) the signal candle's close, filled only if the next candle really trades there, otherwise cancelled. Entries and take-profits pay the 0.02% maker fee; stop-losses and time exits pay the 0.05% taker fee plus 0.03% slippage. A funding cost is charged for every hour held.
- **Data split:**
  - Training: until 2023-01-01
  - Validation: until 2024-07-01
  - **Hidden test:** from then until today. The bot never sees it; its numbers are for reporting only.

The bot was also run on a completely random price series and found no consistently "profitable" strategy, which confirms it can't see the future.

## Scoring that resists luck

- The biggest 5% of winning trades are ignored when a strategy is scored.
- The training period is cut into years; each coin is judged on its **2 worst years**, so a strategy must survive bull, bear and sideways markets.
- At most 2 conditions per direction, with a penalty per condition.
- A new champion must be better on **both** training and validation, by a clear margin.

## How often does it trade?

At least ~75 trades a year per coin (about one every 5 days). Across the five coins that is roughly **one signal a day**. Trades stay open at most 5 days.

## When does it stop?

- Every strategy is searched **3 times independently**, from different random starts. If all three find similar, profitable strategies, the pattern is probably real; if each finds something different, it was luck.
- A search stops after **48 hours** without a better champion. When every search has stopped, the page shows 🏁.

## Second strategy: coin rotation 🔄

Instead of "when do I buy?", it asks "which coin is strongest right now?". Every day (or every few days) the coins are ranked by how much they rose over the last N days: buy the strongest, optionally short the weakest, and stay in cash when the market filter is negative. A small grid of 240 settings is tested on training + validation only, and the hidden test is just reported.

## Why does it lose?

For every champion the page shows a loss analysis on **training + validation trades only**: results by trend, volatility, direction, time of day and how trades ended, plus how many stopped-out trades were in profit first. The hidden test is never analysed, so improvements made from it cannot leak the test.

## Suggested leverage

Leverage is sized so that hitting the stop loses **only 2%** of the account:
`leverage = 2% ÷ stop distance`, capped at 10x.

## Running it

- **Automatic:** every 6 hours via GitHub Actions. It downloads new data, continues evolving from where it stopped, and updates the results.
- **Manual:** in the Actions tab, choose `evolve` and press **Run workflow**.
- **Locally:**
  ```bash
  pip install -r requirements.txt
  python -m bot.fetch
  python -m bot.evolve --variant 4h --minutes 10
  python -m bot.report
  ```

## Settings

All settings live in [`bot/config.py`](bot/config.py): coins, fees, split dates, risk per trade, population size.

> ⚠️ This is a learning and experimentation project. Results on historical data are no guarantee of future performance, and nothing here is financial advice.
