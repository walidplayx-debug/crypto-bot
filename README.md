# 🤖 crypto-bot

**English** · [العربية](README.ar.md)

A bot that teaches itself trading strategies with a **genetic algorithm**, on historical data for five coins:
**BTC · ETH · SOL · BNB · DOGE**.

We never tell it how to trade. It gets the indicators, invents strategies, tests them, keeps the best, then mixes and mutates them generation after generation.

📊 **Results:** [`RESULTS.md`](RESULTS.md) and the [live results page](https://walidplayx-debug.github.io/crypto-bot/) (GitHub Pages)

## How it works

1. **Data:** 1-hour candles from Binance (free, no API key). 4-hour candles are built from them.
2. **Indicators (26):** RSI, MACD, Bollinger, EMA 20/50/200, ATR, ADX, Stochastic, CCI, MFI, OBV, volume, Donchian channels, volatility…
3. **A strategy:** up to 3 long conditions and up to 3 short conditions, plus a stop-loss and take-profit in ATR multiples, and a maximum trade duration.
4. **Three timeframes:** 1h, 4h, and 1h with a 4h trend filter.
5. **Two kinds of strategy:**
   - **Universal:** one strategy that has to work on all five coins at once. Harder to find, but the strongest evidence that it isn't luck.
   - **Per coin:** a separate strategy for each coin.

## No cheating

- Decisions are made on a candle's **close**; the trade enters at the **next candle's open**.
- If the stop and the target are both hit inside the same candle, it **counts as a loss**.
- If price gaps through the stop, the worse price is used.
- 0.05% fee and 0.03% slippage on both entry and exit, plus a funding cost for every hour held.
- **Data split:**
  - Training: until 2023-01-01
  - Validation: until 2024-07-01
  - **Hidden test:** from then until today. The bot never sees it; its numbers are for reporting only.

The bot was also run on a completely random price series and found no consistently "profitable" strategy, which confirms it can't see the future.

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
