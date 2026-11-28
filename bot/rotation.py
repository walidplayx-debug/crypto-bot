"""Coin rotation (cross-sectional momentum).

Every day (or every few days) rank the coins by how much they rose over the last
N days. Buy the strongest, optionally short the weakest, and sit in cash when the
market filter says the whole market is weak.

No lookahead: positions are decided with daily CLOSES up to day t and are held
from the OPEN of day t+1 to the OPEN of day t+2. Rotation trades are MARKET orders
(taker fee + slippage) on every change of position.

The parameter grid is small on purpose (240 combinations), the winner is picked on
training + validation only, and the hidden test is just reported.
"""
import itertools
import json

import numpy as np
import pandas as pd

from . import config as C
from .fetch import load

LOOKBACKS = [3, 7, 14, 30, 60]
LONGS = [1, 2]
SHORTS = [0, 1]
FILTERS = ["none", "btc_ema50", "btc_ema200", "breadth"]
REBALANCE = [1, 3, 7]
FILTER_AR = {"none": "بدون فلتر", "btc_ema50": "BTC فوق EMA 50 يوم",
             "btc_ema200": "BTC فوق EMA 200 يوم", "breadth": "متوسط زخم السوق إيجابي"}


def daily_panel():
    opens, closes = {}, {}
    for s in C.SYMBOLS:
        df = load(s, "1h")
        g = df.resample("1D")
        full = g["close"].count() == 24
        opens[s] = g["open"].first()[full]
        closes[s] = g["close"].last()[full]
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    idx = o.index.union(c.index)
    o, c = o.reindex(idx), c.reindex(idx)
    return o, c


def weights_for(params, c):
    L, k, short, filt, R = params
    mom = c / c.shift(L) - 1
    listed = c.notna().cumsum() >= 30                      # at least 30 days of history
    mom = mom.where(listed)
    btc = c["BTCUSDT"]
    ok = pd.Series(True, index=c.index)
    if filt == "btc_ema50":
        ok = btc > btc.ewm(span=50, adjust=False, min_periods=50).mean()
    elif filt == "btc_ema200":
        ok = btc > btc.ewm(span=200, adjust=False, min_periods=200).mean()
    elif filt == "breadth":
        ok = mom.mean(axis=1) > 0
    W = np.zeros(c.shape)
    M = mom.to_numpy()
    okv = ok.to_numpy()
    prev = np.zeros(c.shape[1])
    for t in range(len(c)):
        if t % R:
            W[t] = prev
            continue
        row = M[t]
        avail = np.flatnonzero(np.isfinite(row))
        w = np.zeros(c.shape[1])
        if len(avail) >= 3:
            order = avail[np.argsort(row[avail])]          # weakest ... strongest
            longs = [i for i in order[::-1][:k] if row[i] > 0] if okv[t] else []
            shorts = [i for i in order[:1] if row[i] < 0 and i not in longs] if short else []
            n = len(longs) + len(shorts)
            for i in longs:
                w[i] = 1 / n
            for i in shorts:
                w[i] = -1 / n
        W[t] = prev = w
    return W


def simulate(params, o, c):
    """Daily portfolio returns, indexed by the day the return is earned."""
    W = weights_for(params, c)
    ret = (o.shift(-2) / o.shift(-1) - 1).to_numpy()       # open(t+1) -> open(t+2)
    ret = np.nan_to_num(ret)
    gross = (W * ret).sum(axis=1)
    turnover = np.abs(np.diff(np.vstack([np.zeros(W.shape[1]), W]), axis=0)).sum(axis=1)
    cost = turnover * (C.TAKER_FEE + C.SLIPPAGE) + np.abs(W).sum(axis=1) * 24 * C.FUNDING_PER_HOUR
    r = pd.Series(gross - cost, index=c.index + pd.Timedelta(days=1))
    changes = pd.Series(turnover > 1e-9, index=r.index)
    return r, changes, W


def position_trades(W, o, c):
    """Every continuous holding of one coin in one direction = one trade.
    Returns DataFrame: end day (as return index), return after costs."""
    ret = np.nan_to_num((o.shift(-2) / o.shift(-1) - 1).to_numpy())
    cost = 2 * (C.TAKER_FEE + C.SLIPPAGE)
    out = []
    for j in range(W.shape[1]):
        sign = np.sign(W[:, j])
        t = 0
        while t < len(sign):
            if sign[t] == 0:
                t += 1
                continue
            u = t
            while u + 1 < len(sign) and sign[u + 1] == sign[t]:
                u += 1
            r = float(np.prod(1 + sign[t] * ret[t:u + 1]) - 1 - cost)
            out.append((c.index[u] + pd.Timedelta(days=1), r, u - t + 1))
            t = u + 1
    df = pd.DataFrame(out, columns=["day", "ret", "days"]).set_index("day").sort_index()
    return df


def trade_stats(tr, which):
    t0, t1 = pd.Timestamp(C.TRAIN_END, tz="UTC"), pd.Timestamp(C.VALID_END, tz="UTC")
    if which == "train":
        x = tr[tr.index < t0]
    elif which == "valid":
        x = tr[(tr.index >= t0) & (tr.index < t1)]
    else:
        x = tr[tr.index >= t1]
    if not len(x):
        return {"trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "avg_days": 0.0}
    w, l = x["ret"][x["ret"] > 0], x["ret"][x["ret"] <= 0]
    return {"trades": int(len(x)), "win_rate": round(float((x["ret"] > 0).mean()), 3),
            # medians, so a few huge winners (e.g. a meme-coin rally) don't distort the picture
            "avg_win": round(float(w.median()), 4) if len(w) else 0.0,
            "avg_loss": round(float(l.median()), 4) if len(l) else 0.0,
            "avg_days": round(float(x["days"].mean()), 1)}


def _score(r):
    if len(r) < 30:
        return -9.0
    lg = np.log1p(r.clip(lower=-0.99).to_numpy())
    eq = np.concatenate([[0], np.cumsum(lg)])
    years = len(r) / 365
    return float(eq[-1] / years - 0.5 * (np.maximum.accumulate(eq) - eq).max())


def part(r, which):
    t0, t1 = pd.Timestamp(C.TRAIN_END, tz="UTC"), pd.Timestamp(C.VALID_END, tz="UTC")
    start = r.index[0] + pd.Timedelta(days=60)            # warm-up for the longest lookback
    if which == "train":
        return r[(r.index >= start) & (r.index < t0)]
    if which == "valid":
        return r[(r.index >= t0) & (r.index < t1)]
    return r[r.index >= t1].iloc[:-2]                      # last 2 days have no full return yet


def train_score(r):
    """Average of the 2 worst training years, like the main bot."""
    tr = part(r, "train")
    chunks = [tr.iloc[i:i + 365] for i in range(0, len(tr), 365)]
    if len(chunks) > 1 and len(chunks[-1]) < 180:
        chunks[-2] = pd.concat([chunks[-2], chunks.pop()])
    s = sorted(_score(x) for x in chunks)
    return float(np.mean(s[:2]))


def metrics(r, changes):
    years = max(len(r) / 365, 1e-9)
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    dd = float(1 - (eq / eq.cummax()).min()) if len(eq) else 0.0
    return {"total": round(total, 4), "cagr": round((1 + total) ** (1 / years) - 1, 4) if total > -1 else -1.0,
            "max_dd": round(dd, 4), "days": int(len(r)), "changes_per_year": round(float(changes.sum()) / years, 1),
            "in_market": round(float((r != 0).mean()), 3) if len(r) else 0.0}


def describe(params):
    L, k, short, filt, R = params
    every = "كل يوم" if R == 1 else f"كل {R} أيام"
    txt = [f"{every}: رتّب العملات حسب صعودها آخر {L} يوم",
           f"اشترِ أقوى {'عملة' if k == 1 else 'عملتين'} (بس إذا زخمها إيجابي)"]
    if short:
        txt.append("وبيع (Short) أضعف عملة (بس إذا زخمها سلبي)")
    if filt != "none":
        txt.append(f"الشراء بس إذا {FILTER_AR[filt]}، وإلا كاش")
    return txt


def run():
    o, c = daily_panel()
    rows = []
    for params in itertools.product(LOOKBACKS, LONGS, SHORTS, FILTERS, REBALANCE):
        r, ch, W = simulate(params, o, c)
        ts, vs = train_score(r), _score(part(r, "valid"))
        rows.append({"params": params, "train": ts, "valid": vs, "robust": min(ts, vs), "r": r, "ch": ch, "W": W})
    rows.sort(key=lambda x: x["robust"], reverse=True)
    best = rows[0]
    live_path = C.DOCS_DIR / "live.json"
    if live_path.exists():   # once live, always show the frozen live settings
        frozen = tuple(json.loads(live_path.read_text())["params"])
        best = next((x for x in rows if x["params"] == frozen), best)
    ew = (o.shift(-2) / o.shift(-1) - 1)
    bench = pd.Series(ew.mean(axis=1).to_numpy(), index=c.index + pd.Timedelta(days=1)).fillna(0)

    def block(row):
        out = {"params": dict(zip(["lookback", "longs", "short", "filter", "rebalance"], row["params"])),
               "rules": describe(row["params"]), "robust": round(row["robust"], 4),
               "train_score": round(row["train"], 4), "valid_score": round(row["valid"], 4)}
        trades = position_trades(row["W"], o, c)
        for p in ("train", "valid", "test"):
            rp = part(row["r"], p)
            out[p] = metrics(rp, row["ch"].reindex(rp.index).fillna(False))
            out[p].update(trade_stats(trades, p))
            out[p]["bench"] = metrics(part(bench, p), pd.Series(False, index=part(bench, p).index))["total"]
        return out

    W = best["W"][-1]
    today = [{"symbol": s, "side": "شراء (Long)" if w > 0 else "بيع (Short)", "weight": round(abs(float(w)), 2)}
             for s, w in zip(c.columns, W) if abs(w) > 1e-9]
    res = {"best": block(best), "top": [block(x) for x in rows[:5]],
           "today": today, "today_date": str(c.index[-1].date()),
           "positive_test_in_top5": sum(1 for x in rows[:5] if part(x["r"], "test").add(1).prod() > 1)}
    res["live"] = live_update(best["params"], o, c)
    return res, best["r"], bench


def live_update(best_params, o, c):
    """Paper trading from the go-live day on: the settings are frozen on the first day,
    every change of positions is logged, and performance is measured only on days
    that came AFTER the go-live day."""
    path = C.DOCS_DIR / "live.json"
    st = json.loads(path.read_text()) if path.exists() else None
    if not st:
        st = {"start": str(c.index[-1].date()), "params": list(best_params), "log": []}
    params = tuple(st["params"])
    r, _, W = simulate(params, o, c)
    day = c.index[-1]
    pos = [{"symbol": sym, "side": "long" if w > 0 else "short", "weight": round(abs(float(w)), 2),
            "price": round(float(c[sym].iloc[-1]), 6)}
           for sym, w in zip(c.columns, W[-1]) if abs(w) > 1e-9]
    key = sorted([p_["symbol"], p_["side"], p_["weight"]] for p_ in pos)
    last = st["log"][-1] if st["log"] else None
    if last is None or (last["key"] != key and str(day.date()) > last["date"]):
        st["log"].append({"date": str(day.date()), "positions": pos, "key": key})
    start = pd.Timestamp(st["start"], tz="UTC")
    done = r.iloc[:-2]                                   # last two days have no full return yet
    rl = done[done.index >= start + pd.Timedelta(days=1)]
    eq = (1 + rl).cumprod()
    perf = {"days": int(len(rl)), "total": round(float(eq.iloc[-1] - 1), 4) if len(rl) else 0.0,
            "max_dd": round(float(1 - (eq / eq.cummax()).min()), 4) if len(rl) else 0.0}
    R = params[4]
    t_last = len(c) - 1
    k = (R - t_last % R) % R or R
    next_signal = (day + pd.Timedelta(days=k + 1)).strftime("%Y-%m-%d")
    path.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return {"start": st["start"], "params": dict(zip(["lookback", "longs", "short", "filter", "rebalance"], params)),
            "rules": describe(params), "positions": pos, "as_of": str(day.date()),
            "next_signal": next_signal, "perf": perf, "log": st["log"][-15:][::-1]}


def plot(r, bench, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID, AXIS, BLUE = "#191919", "#efefef", "#9c9c9c", "#303030", "#474747", "#338ced"
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "legend.frameon": False, "legend.labelcolor": INK2})
    start = part(r, "train").index[0]
    r, bench = r[r.index >= start].iloc[:-2], bench[bench.index >= start].iloc[:-2]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(bench.index, (1 + bench).cumprod(), color=INK2, lw=1.2, ls="--", label="Equal-weight buy & hold")
    ax.plot(r.index, (1 + r).cumprod(), color=BLUE, lw=2, label="Rotation strategy")
    ax.set_yscale("log")
    t0, t1 = pd.Timestamp(C.TRAIN_END, tz="UTC"), pd.Timestamp(C.VALID_END, tz="UTC")
    ax.axvspan(t0, t1, color="#ffffff", alpha=0.04, lw=0)
    ax.axvspan(t1, r.index[-1], color=BLUE, alpha=0.10, lw=0)
    ax.set_title("shaded light: validation  ·  shaded blue: hidden test", loc="left", color=INK2, fontsize=8.5)
    ax.set_ylabel("Equity (x)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
