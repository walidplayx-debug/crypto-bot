"""Why does a strategy lose? Breaks its trades down by market condition.

Only TRAINING + VALIDATION trades are analysed. The hidden test is never looked at,
so changes made after reading this analysis cannot leak the test into the bot.
"""
import numpy as np

from research.engine import backtest, leverage_of

MIN_N = 15   # a bucket needs this many trades before we say anything about it


def _trades(d, g):
    rows = []
    for rng in (d.train, d.valid):
        if rng[1] - rng[0] < 50:
            continue
        t = backtest(d, g, rng)
        if not len(t["ret"]):
            continue
        rl = np.maximum(t["ret"] * leverage_of(t["sl_pct"]), -1)
        X = d.X
        dist200 = X[t["sig_bar"], d.col("dist_ema200")]
        volreg = X[t["sig_bar"], d.col("vol_regime")]
        for i in range(len(rl)):
            e, x, direc = t["entry_bar"][i], t["exit_bar"][i], t["dir"][i]
            px, risk = t["entry_px"][i], t["sl_pct"][i] * t["entry_px"][i]
            if direc == 1:
                mfe = (d.h[e:x + 1].max() - px) / risk
            else:
                mfe = (px - d.l[e:x + 1].min()) / risk
            ts = d.index[t["sig_bar"][i]]
            rows.append({
                "ret": float(rl[i]), "win": rl[i] > 0, "reason": t["reason"][i],
                "side": "long" if direc == 1 else "short",
                "trend": "up" if dist200[i] > 0 else "down",
                "vol": "high" if volreg[i] > 1.25 else ("low" if volreg[i] < 0.8 else "normal"),
                "session": f"{(ts.hour // 6) * 6:02d}-{(ts.hour // 6) * 6 + 6:02d}",
                "weekend": ts.dayofweek >= 5, "mfe": float(mfe),
            })
    return rows


def _bucket(rows, key):
    out = {}
    for r in rows:
        b = out.setdefault(str(r[key]), [])
        b.append(r)
    return {k: {"n": len(v), "win_rate": round(float(np.mean([x["win"] for x in v])), 3),
                "avg": round(float(np.mean([x["ret"] for x in v])), 4)} for k, v in sorted(out.items())}


AR = {
    "trend": ("الترند", {"up": "صاعد (فوق EMA 200)", "down": "هابط (تحت EMA 200)"}),
    "vol": ("التذبذب", {"high": "عالي", "normal": "عادي", "low": "هادي"}),
    "side": ("الاتجاه", {"long": "شراء", "short": "بيع"}),
    "session": ("التوقيت (UTC)", {}),
    "weekend": ("الأيام", {"True": "نهاية الأسبوع", "False": "أيام الأسبوع"}),
    "reason": ("طريقة الإغلاق", {"sl": "ضرب الوقف", "tp": "ضرب الهدف", "time": "خلص الوقت"}),
}


def analyse(datasets, g):
    rows = [r for d in datasets for r in _trades(d, g)]
    if not rows:
        return None
    total_avg = float(np.mean([r["ret"] for r in rows]))
    res = {"n": len(rows), "win_rate": round(float(np.mean([r["win"] for r in rows])), 3),
           "avg": round(total_avg, 4), "buckets": {k: _bucket(rows, k) for k in AR}, "insights": []}
    ins = res["insights"]
    # worst and best market conditions
    for k in ("trend", "vol", "side", "session", "weekend"):
        b = {kk: v for kk, v in res["buckets"][k].items() if v["n"] >= MIN_N}
        if len(b) < 2:
            continue
        worst = min(b, key=lambda x: b[x]["avg"])
        best = max(b, key=lambda x: b[x]["avg"])
        if b[worst]["avg"] < 0 and b[best]["avg"] - b[worst]["avg"] > 0.005:
            name, labels = AR[k]
            ins.append(f"{name}: بتخسر أكتر لما يكون {labels.get(worst, worst)} "
                       f"(نجاح {b[worst]['win_rate'] * 100:.0f}% · متوسط {b[worst]['avg'] * 100:+.2f}% بالصفقة) "
                       f"مقابل {labels.get(best, best)} (نجاح {b[best]['win_rate'] * 100:.0f}% · {b[best]['avg'] * 100:+.2f}%)")
    # how trades end
    rs = res["buckets"]["reason"]
    sl_share = rs.get("sl", {}).get("n", 0) / len(rows)
    if sl_share > 0.5:
        ins.append(f"{sl_share * 100:.0f}% من الصفقات بتسكّر على الوقف: على الأغلب الدخول متأخر أو الوقف قريب كتير")
    if rs.get("time", {}).get("n", 0) / len(rows) > 0.3:
        ins.append(f"{rs['time']['n'] / len(rows) * 100:.0f}% من الصفقات بتخلص وقتها بدون وقف ولا هدف: الهدف بعيد أو المدة قصيرة")
    losers = [r for r in rows if r["reason"] == "sl"]
    if len(losers) >= MIN_N:
        gave_back = np.mean([r["mfe"] >= 1.0 for r in losers])
        res["losers_were_up_1R"] = round(float(gave_back), 3)
        if gave_back > 0.25:
            ins.append(f"{gave_back * 100:.0f}% من الصفقات اللي ضربت الوقف كانت رابحة قدّ المخاطرة أو أكتر "
                       "قبل ما ترجع: نقل الوقف لنقطة الدخول أو هدف أقرب ممكن يساعد")
    if not ins:
        ins.append("ما في نمط واضح للخسائر: الخسارة موزعة على كل الظروف")
    return res
