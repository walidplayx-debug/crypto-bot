"""Datasets, strategy logic and the backtester.

Honesty rules built into the backtest:
  * a signal is computed at the CLOSE of bar i and entered at the OPEN of bar i+1
  * stop-loss / take-profit come from ATR known at bar i
  * if stop and target are both touched in the same candle, the STOP is assumed
  * if price gaps through the stop, the worse open price is used
  * fees + slippage on both sides and a funding cost for every hour held
  * thresholds are percentiles measured on TRAINING data only
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .fetch import load
from .indicators import add_higher_tf, features

PCTS = np.arange(101) / 100.0


@dataclass
class Dataset:
    symbol: str
    variant: str
    index: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    X: np.ndarray            # features (bars x n_features)
    names: list
    qtable: np.ndarray       # n_features x 101 thresholds from training data
    bar_hours: int
    train: tuple             # (start, end) bar index ranges
    valid: tuple
    test: tuple

    def col(self, name):
        return self.names.index(name)


def build(symbol, variant):
    tf = "4h" if variant == "4h" else "1h"
    df = load(symbol, tf)
    f, atr = features(df)
    if variant == "1h+4h":
        f = add_higher_tf(f, load(symbol, "4h"))
    names = list(f.columns)
    X = f.to_numpy(np.float64)
    idx = df.index
    t_end = int(idx.searchsorted(pd.Timestamp(C.TRAIN_END, tz="UTC")))
    v_end = int(idx.searchsorted(pd.Timestamp(C.VALID_END, tz="UTC")))
    start = int(np.argmax(np.isfinite(X).all(axis=1)))  # first bar with every feature ready
    qtable = np.nanquantile(X[start:t_end], PCTS, axis=0).T
    return Dataset(symbol, variant, idx, df["open"].to_numpy(), df["high"].to_numpy(),
                   df["low"].to_numpy(), df["close"].to_numpy(), atr.to_numpy(), X, names,
                   qtable, 4 if tf == "4h" else 1, (start, t_end), (t_end, v_end), (v_end, len(df)))


# ---------------------------------------------------------------- strategy

def side_mask(d, rules):
    """rules: list of [feature_name, op ('<' or '>'), percentile 1..99]"""
    if not rules:
        return None
    m = np.ones(len(d.c), bool)
    for name, op, p in rules:
        j = d.col(name)
        x = d.X[:, j]
        thr = d.qtable[j, int(p)]
        m &= (x < thr) if op == "<" else (x > thr)
    return m


def signals(d, g):
    """+1 long / -1 short / 0 none for every bar (decided at bar close)."""
    s = np.zeros(len(d.c), np.int8)
    none = np.zeros(len(d.c), bool)
    ml = side_mask(d, g["long"]); ml = none if ml is None else ml
    ms = side_mask(d, g["short"]); ms = none if ms is None else ms
    s[ml & ~ms] = 1            # both at once -> conflicting, stay out
    s[ms & ~ml] = -1
    s[~np.isfinite(d.atr)] = 0
    return s


def trade_plan(d, g, i, direction):
    """Entry/SL/TP/leverage for a signal at bar i (entry is next bar's open, estimated by close)."""
    ref = d.c[i]
    dist = g["sl"] * d.atr[i]
    sl = ref - direction * dist
    tp = ref + direction * g["tp"] * d.atr[i]
    sl_pct = dist / ref
    lev = float(min(C.MAX_LEVERAGE, max(1.0, C.RISK_PER_TRADE / sl_pct)))
    return sl, tp, sl_pct, lev


def backtest(d, g, rng, sig=None):
    """Simulate trades for bars in rng=(a, b). Returns list of trade dicts (light)."""
    a, b = rng
    if sig is None:
        sig = signals(d, g)
    idx = np.flatnonzero(sig[a:b - 1]) + a
    cost = 2 * (C.FEE + C.SLIPPAGE)
    hold = int(g["hold"])
    out_i, out_dir, out_ret, out_slp, out_k, out_exit = [], [], [], [], [], []
    free = a
    n = len(idx)
    o, h, l, c, atr = d.o, d.h, d.l, d.c, d.atr
    while True:
        p = int(np.searchsorted(idx, free, side="left"))
        if p >= n:
            break
        s = int(idx[p])
        e = s + 1
        if e >= b:
            break
        direc = int(sig[s])
        entry = o[e]
        dist = g["sl"] * atr[s]
        sl = entry - direc * dist
        tp = entry + direc * g["tp"] * atr[s]
        end = min(e + hold, b)
        hi, lo = h[e:end], l[e:end]
        if direc == 1:
            hs, ht = lo <= sl, hi >= tp
        else:
            hs, ht = hi >= sl, lo <= tp
        ks = int(np.argmax(hs)) if hs.any() else 10**9
        kt = int(np.argmax(ht)) if ht.any() else 10**9
        if ks <= kt and ks < 10**9:
            k = ks
            px = sl
            op_ = o[e + k]
            if k > 0 and ((direc == 1 and op_ < sl) or (direc == -1 and op_ > sl)):
                px = op_   # gapped through the stop
            reason = "sl"
        elif kt < 10**9:
            k, px, reason = kt, tp, "tp"
        else:
            k, px, reason = end - 1 - e, c[end - 1], "time"
        hours = (k + 1) * d.bar_hours
        r = direc * (px / entry - 1) - cost - hours * C.FUNDING_PER_HOUR
        out_i.append(s); out_dir.append(direc); out_ret.append(r)
        out_slp.append(dist / entry); out_k.append(e + k); out_exit.append(reason)
        free = e + k          # a new signal may come at the close of the exit candle
    return {
        "sig_bar": np.array(out_i, int), "dir": np.array(out_dir, int),
        "ret": np.array(out_ret, float), "sl_pct": np.array(out_slp, float),
        "exit_bar": np.array(out_k, int), "reason": out_exit,
    }


# ---------------------------------------------------------------- metrics

def leverage_of(sl_pct):
    return np.clip(C.RISK_PER_TRADE / np.maximum(sl_pct, 1e-9), 1.0, C.MAX_LEVERAGE)


def metrics(d, t, rng):
    a, b = rng
    years = max((b - a) * d.bar_hours / 8760, 1e-9)
    r = t["ret"]
    rl = np.maximum(r * leverage_of(t["sl_pct"]), -1.0)
    res = {"trades": int(len(r)), "years": round(years, 2),
           "trades_per_year": round(len(r) / years, 1)}
    for tag, x in (("x1", r), ("lev", rl)):
        eq = np.cumprod(1 + x) if len(x) else np.array([1.0])
        peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
        dd = float(1 - (np.concatenate([[1.0], eq]) / peak).min())
        final = float(eq[-1])
        cagr = final ** (1 / years) - 1 if final > 0 else -1.0
        res[tag] = {"total": round(final - 1, 4), "cagr": round(cagr, 4), "max_dd": round(dd, 4)}
    if len(r):
        wins = r[r > 0]
        loss = -r[r <= 0].sum()
        res["win_rate"] = round(float((r > 0).mean()), 3)
        res["profit_factor"] = round(float(wins.sum() / loss), 2) if loss > 0 else 99.0
        res["avg_trade"] = round(float(r.mean()), 4)
    else:
        res.update(win_rate=0.0, profit_factor=0.0, avg_trade=0.0)
    bh = d.c[b - 1] / d.o[a]
    res["buy_hold"] = {"total": round(float(bh - 1), 4),
                       "cagr": round(float(bh ** (1 / years) - 1), 4)}
    return res


def _period_score(rl, years):
    """Risk-sized log growth per year minus drawdown, penalised for too few trades."""
    if len(rl) == 0:
        return -1.0
    logeq = np.concatenate([[0.0], np.cumsum(np.log1p(rl))])
    growth = logeq[-1] / years
    dd = float((np.maximum.accumulate(logeq) - logeq).max())
    s = growth - 0.5 * dd
    per_year = len(rl) / years
    if per_year < C.MIN_TRADES_PER_YEAR:
        s -= 0.5 * (1 - per_year / C.MIN_TRADES_PER_YEAR)
    return float(s)


def score(d, t, rng):
    """One number per coin, built to resist luck:
      1. the biggest winning trades (top DROP_TOP_WINNERS) are removed first
      2. the period is cut into ~1-year chunks and each chunk is scored on its own
      3. the coin's score is the average of its WORST chunks, so a strategy has to
         hold up in every kind of market, not ride one lucky rally."""
    a, b = rng
    r = t["ret"]
    if len(r) == 0:
        return -1.0
    rl = np.maximum(r * leverage_of(t["sl_pct"]), -0.999)
    bars = t["sig_bar"]
    k = int(np.ceil(len(rl) * C.DROP_TOP_WINNERS))
    if k and len(rl) >= 10:
        keep = np.ones(len(rl), bool)
        keep[np.argsort(rl)[-k:]] = False
        rl, bars = rl[keep], bars[keep]
    year_bars = int(8760 / d.bar_hours)
    edges = list(range(a, b, year_bars)) + [b]
    if len(edges) > 2 and edges[-1] - edges[-2] < year_bars // 2:
        edges.pop(-2)                      # merge a short last piece into the previous year
    chunks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (bars >= lo) & (bars < hi)
        chunks.append(_period_score(rl[m], (hi - lo) * d.bar_hours / 8760))
    chunks.sort()
    worst = chunks[:C.WORST_YEARS] if len(chunks) > C.WORST_YEARS else chunks[:1]
    return float(np.mean(worst))


def fitness(datasets, g, part):
    """part: 'train' | 'valid' | 'test'. Mean minus half the spread across coins,
    so a strategy must work on every coin, not just one."""
    vals = []
    for d in datasets:
        rng = getattr(d, part)
        if rng[1] - rng[0] < 50:
            continue
        vals.append(score(d, backtest(d, g, rng), rng))
    if not vals:
        return -9.0
    v = np.array(vals)
    complexity = C.COMPLEXITY_PENALTY * (len(g["long"]) + len(g["short"]))
    return float(v.mean() - 0.5 * v.std() - complexity)
