"""Download (and incrementally update) 1-hour candles from Binance's public data API.

No API key needed. Data is stored in data/<SYMBOL>_1h.csv.gz; 4-hour candles are
built from it locally, so both timeframes always come from the same source.
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from .config import DATA_DIR, ROTATION_SYMBOLS, START_MONTH, SYMBOLS

HOSTS = [
    "https://data-api.binance.vision",  # public market-data mirror (works from GitHub runners)
    "https://api.binance.com",
    "https://api1.binance.com",
]
HOUR_MS = 3_600_000
STEP_MS = {"1h": HOUR_MS, "1d": 24 * HOUR_MS}
COLS = ["time", "open", "high", "low", "close", "volume"]


def _get(path):
    last = None
    for host in HOSTS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(host + path, headers={"User-Agent": "crypto-bot"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read())
            except Exception as e:  # try again, then next host
                last = e
                time.sleep(1 + attempt)
    raise RuntimeError(f"all hosts failed for {path}: {last}")


def _download(symbol, start_ms, interval="1h"):
    rows = []
    now_ms = int(time.time() * 1000)
    while start_ms < now_ms:
        batch = _get(f"/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_ms}&limit=1000")
        if not batch:
            break
        rows += [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in batch]
        start_ms = int(batch[-1][0]) + STEP_MS[interval]
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return pd.DataFrame(rows, columns=COLS)


def path_for(symbol, interval="1h"):
    return DATA_DIR / f"{symbol}_{interval}.csv.gz"


def update(symbol, interval="1h"):
    p = path_for(symbol, interval)
    if p.exists():
        old = pd.read_csv(p)
        start = int(old["time"].iloc[-1])  # re-download last candle (it may have been open)
    else:
        old = pd.DataFrame(columns=COLS)
        start = int(datetime.strptime(START_MONTH, "%Y-%m").replace(tzinfo=timezone.utc).timestamp() * 1000)
    new = _download(symbol, start, interval)
    df = pd.concat([old, new]) if len(old) else new
    df = df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    # drop the still-open candle so no unfinished data is ever stored
    now_ms = int(time.time() * 1000)
    df = df[df["time"] + STEP_MS[interval] <= now_ms]
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(p, index=False, compression="gzip")
    first = pd.to_datetime(df["time"].iloc[0], unit="ms")
    print(f"{symbol}: {len(df)} candles from {first:%Y-%m-%d} (+{len(new)} downloaded)", flush=True)


def load(symbol, timeframe):
    """Return OHLCV DataFrame indexed by UTC open time for '1h' or '4h'."""
    df = pd.read_csv(path_for(symbol))
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    df = df.astype(float)
    if timeframe == "1h":
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample("4h", origin="epoch").agg(agg)
    counts = df["close"].resample("4h", origin="epoch").count()
    return out[counts == 4].dropna()  # only complete 4h candles


if __name__ == "__main__":
    if sys.argv[1:2] == ["--hourly"]:          # 1h candles for the archived research
        for s in SYMBOLS:
            update(s, "1h")
    else:                                       # daily candles for the rotation signal
        for s in ROTATION_SYMBOLS:
            try:
                update(s, "1d")
            except Exception as e:              # one bad coin must not stop the others
                print(f"{s}: skipped ({e})", flush=True)
