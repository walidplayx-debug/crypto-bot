"""Indicator features. Every value at bar i uses only data up to the CLOSE of bar i,
so a decision made at that close can never peek into the future."""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rma(s, n):  # Wilder smoothing
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _rsi(c, n):
    d = c.diff()
    up, dn = _rma(d.clip(lower=0), n), _rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df, n):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, n)


def features(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    f["rsi14"] = _rsi(c, 14)
    f["rsi7"] = _rsi(c, 7)

    macd = _ema(c, 12) - _ema(c, 26)
    sig = _ema(macd, 9)
    f["macd"] = macd / c * 100
    f["macd_hist"] = (macd - sig) / c * 100

    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20)
    f["bb_width"] = 4 * s20 / m20

    for n in (20, 50, 200):
        f[f"dist_ema{n}"] = c / _ema(c, n) - 1
    f["ema20_50"] = _ema(c, 20) / _ema(c, 50) - 1
    f["ema50_200"] = _ema(c, 50) / _ema(c, 200) - 1

    atr = _atr(df, 14)
    f["atr_pct"] = atr / c

    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr14 = _atr(df, 14)
    pdi, ndi = 100 * _rma(pdm, 14) / tr14, 100 * _rma(ndm, 14) / tr14
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    f["adx"] = _rma(dx, 14)
    f["di_diff"] = pdi - ndi

    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()

    tp = (h + l + c) / 3
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    f["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * md)

    for n in (1, 6, 24):
        f[f"roc{n}"] = c.pct_change(n)

    f["vol_z"] = (v - v.rolling(50).mean()) / v.rolling(50).std()

    for n in (20, 55):
        lo, hi = l.rolling(n).min(), h.rolling(n).max()
        f[f"donch{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)

    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    f["mfi"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_z"] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()

    rv = c.pct_change().rolling(24).std()
    f["vol_regime"] = rv / rv.rolling(200).mean()

    return f.replace([np.inf, -np.inf], np.nan), atr


def add_higher_tf(f1h, df4h, prefix="htf_"):
    """Attach 4h features to 1h rows using only 4h candles that are already CLOSED
    at the moment the 1h candle closes."""
    f4, _ = features(df4h)
    f4 = f4.add_prefix(prefix).sort_index()
    close_4h = f4.index + pd.Timedelta(hours=4)           # when each 4h candle is closed
    close_1h = f1h.index + pd.Timedelta(hours=1)          # when each 1h candle is closed
    pos = close_4h.searchsorted(close_1h, side="right") - 1
    out = np.full((len(f1h), f4.shape[1]), np.nan)
    ok = pos >= 0
    out[ok] = f4.to_numpy()[pos[ok]]
    stale = np.zeros(len(f1h), bool)
    stale[ok] = (close_1h[ok] - close_4h[pos[ok]]) > pd.Timedelta(hours=8)  # data gap
    out[stale] = np.nan
    return pd.concat([f1h, pd.DataFrame(out, index=f1h.index, columns=f4.columns)], axis=1)
