"""Builds docs/data.json, docs/img/*.png and RESULTS.md from the saved champions."""
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .engine import backtest, build, leverage_of, metrics, signals, trade_plan  # noqa: E402

# ---- chart style (dark surface, thin marks, recessive grid)
SURF, INK, INK2, GRID, AXIS = "#191919", "#efefef", "#9c9c9c", "#303030", "#474747"
BLUE, ORANGE = "#338ced", "#e76624"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False, "legend.labelcolor": INK2,
})

LABELS = {
    "rsi14": "RSI 14", "rsi7": "RSI 7", "macd": "MACD", "macd_hist": "هيستوغرام MACD",
    "bb_pctb": "موقع السعر داخل بولينجر", "bb_width": "عرض بولينجر",
    "dist_ema20": "بُعد السعر عن EMA 20", "dist_ema50": "بُعد السعر عن EMA 50",
    "dist_ema200": "بُعد السعر عن EMA 200", "ema20_50": "EMA 20 مقابل EMA 50",
    "ema50_200": "EMA 50 مقابل EMA 200", "atr_pct": "التذبذب ATR%", "adx": "قوة الترند ADX",
    "di_diff": "اتجاه DI+ ناقص DI−", "stoch_k": "ستوكاستك K", "stoch_d": "ستوكاستك D",
    "cci": "CCI", "roc1": "تغيّر آخر شمعة", "roc6": "تغيّر آخر 6 شمعات", "roc24": "تغيّر آخر 24 شمعة",
    "vol_z": "حجم التداول مقارنة بالمعتاد", "donch20": "موقع السعر في قناة 20",
    "donch55": "موقع السعر في قناة 55", "mfi": "MFI", "obv_z": "OBV مقارنة بالمعتاد",
    "vol_regime": "التذبذب مقارنة بالمعتاد",
}


def label(name):
    if name.startswith("htf_"):
        return LABELS.get(name[4:], name[4:]) + " (فريم 4 ساعات)"
    return LABELS.get(name, name)


def fmt_val(x):
    return f"{x:.3g}" if abs(x) < 1 else f"{x:.2f}"


def describe(g, d):
    def side(rules):
        out = []
        for name, op, p in rules:
            thr = d.qtable[d.col(name), int(p)]
            word = "أقل من" if op == "<" else "أعلى من"
            out.append(f"{label(name)} {word} {fmt_val(thr)} (مستوى {int(p)}% تاريخياً)")
        return out
    unit = "ساعات" if d.bar_hours == 4 else "ساعة"
    return {
        "long": side(g["long"]), "short": side(g["short"]),
        "sl": f"وقف الخسارة = {g['sl']} × ATR",
        "tp": f"الهدف = {g['tp']} × ATR (العائد/المخاطرة ≈ {g['tp'] / g['sl']:.1f})",
        "hold": f"أقصى مدة للصفقة = {g['hold']} شمعة ({g['hold'] * d.bar_hours} {unit})",
    }


def verdict(m):
    lev, bh = m["lev"]["total"], m["buy_hold"]["total"]
    small = m["trades"] < 10
    if lev > 0 and lev > bh:
        v = ("good", "ربح وتغلّب على اشترِ واحتفظ")
    elif lev > 0:
        v = ("ok", "ربح لكن أقل من اشترِ واحتفظ")
    elif lev > bh:
        v = ("soft", "خسارة أقل من السوق")
    else:
        v = ("bad", "خسارة")
    return {"level": v[0], "text": v[1] + (" — عينة صغيرة" if small else "")}


def equity_points(d, g):
    """Leveraged & 1x equity across train→valid→test, plus buy & hold, sampled for plotting."""
    sig = signals(d, g)
    xs, e1, el = [d.index[d.train[0]]], [1.0], [1.0]
    for rng in (d.train, d.valid, d.test):
        if rng[1] - rng[0] < 2:
            continue
        t = backtest(d, g, rng, sig)
        lev = leverage_of(t["sl_pct"])
        for k, r, lv in zip(t["exit_bar"], t["ret"], lev):
            xs.append(d.index[k]); e1.append(e1[-1] * (1 + r)); el.append(el[-1] * max(1 + r * lv, 0.0))
    a = d.train[0]
    step = max(1, (len(d.c) - a) // 600)
    bx = d.index[a::step]
    by = d.c[a::step] / d.o[a]
    return xs, e1, el, bx, by


def plot_equity(datasets, g, path, title):
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n + 0.4), squeeze=False)
    for ax, d in zip(axes[:, 0], datasets):
        xs, e1, el, bx, by = equity_points(d, g)
        ax.plot(bx, by, color=INK2, lw=1.2, ls="--", label="Buy & hold")
        ax.step(xs, e1, where="post", color=BLUE, lw=2, label="Strategy 1x")
        ax.step(xs, el, where="post", color=ORANGE, lw=2, label="Strategy, suggested leverage")
        ax.set_yscale("log")
        ax.axvspan(d.index[d.valid[0]], d.index[d.valid[1] - 1], color="#ffffff", alpha=0.04, lw=0)
        ax.axvspan(d.index[d.test[0]], d.index[-1], color=BLUE, alpha=0.10, lw=0)
        ax.set_title(f"{d.symbol}  ·  shaded light: validation  ·  shaded blue: hidden test",
                     loc="left", color=INK2, fontsize=8.5)
        ax.set_ylabel("Equity (x)")
    axes[0, 0].legend(loc="upper left", ncol=3, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_progress(states, path):
    fig, ax = plt.subplots(figsize=(8, 3))
    for st in states:
        h = [x for x in st["history"] if x[2] is not None]
        if not h:
            continue
        ax.plot([x[0] for x in h], [x[2] for x in h], lw=2,
                label=st["scope"], color={"ALL": BLUE}.get(st["scope"]))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Champion score\n(weaker of train & validation)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(ncol=6, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def current_signal(d, g):
    """What the strategy says at the latest CLOSED candle (preview for the Telegram stage)."""
    s = signals(d, g)
    i = len(d.c) - 1
    if s[i] == 0:
        return None
    sl, tp, sl_pct, lev = trade_plan(d, g, i, int(s[i]))
    return {"time": d.index[i].isoformat(), "side": "شراء (Long)" if s[i] == 1 else "بيع (Short)",
            "entry": float(d.c[i]), "sl": float(sl), "tp": float(tp), "leverage": round(lev, 1)}


def main():
    out = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "split": {"train_end": C.TRAIN_END, "valid_end": C.VALID_END}, "variants": []}
    img = C.DOCS_DIR / "img"
    img.mkdir(parents=True, exist_ok=True)
    for variant in C.VARIANTS:
        vdir = C.STATE_DIR / variant
        if not vdir.exists():
            continue
        datasets = {s: build(s, variant) for s in C.SYMBOLS}
        states = [json.loads(p.read_text()) for p in sorted(vdir.glob("*.json"))]
        states.sort(key=lambda s: (s["scope"] != "ALL", s["scope"]))
        vslug = variant.replace("+", "_")
        plot_progress(states, img / f"progress_{vslug}.png")
        entries = []
        for st in states:
            ch = st.get("champion")
            e = {"scope": st["scope"], "generation": st["generation"],
                 "since_improve": st["since_improve"], "converged": st["since_improve"] >= 400}
            if not ch:
                entries.append(e); continue
            g = ch["genome"]
            ds = list(datasets.values()) if st["scope"] == "ALL" else [datasets[st["scope"]]]
            coins = []
            for d in ds:
                parts = {p: metrics(d, backtest(d, g, getattr(d, p)), getattr(d, p))
                         for p in ("train", "valid", "test")}
                parts["verdict"] = verdict(parts["test"])
                parts["symbol"] = d.symbol
                parts["signal_now"] = current_signal(d, g)
                coins.append(parts)
            fname = f"eq_{vslug}_{st['scope']}.png"
            plot_equity(ds, g, img / fname, f"{variant} · {st['scope']}")
            e.update(weak=ch.get("robust", 0) <= 0, genome=g, rules=describe(g, ds[0]), train_fitness=ch["train_fitness"],
                     valid_fitness=ch["valid_fitness"], found_generation=ch["found_generation"],
                     coins=coins, chart=f"img/{fname}")
            entries.append(e)
        out["variants"].append({"variant": variant, "progress": f"img/progress_{vslug}.png", "scopes": entries})
    (C.DOCS_DIR / "data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    write_markdown(out)
    print("report written")


def pct(x):
    return f"{x * 100:+.1f}%"


def write_markdown(out):
    L = ["# نتائج البوت", "",
         f"آخر تحديث: **{out['updated']}**", "",
         f"التدريب: حتى {out['split']['train_end']} · التحقق: حتى {out['split']['valid_end']} · "
         "**الاختبار المخفي: بعدها حتى اليوم** (البوت ما يشوفه أبداً وقت التطوير)", "",
         "الأرقام أدناه من **الاختبار المخفي** فقط، مع الرافعة المقترحة (مخاطرة 2% لكل صفقة).", ""]
    for v in out["variants"]:
        L += [f"## فريم {v['variant']}", "",
              "| الاستراتيجية | العملة | صفقات | صفقات/سنة | نسبة النجاح | الربح | أقصى تراجع | اشترِ واحتفظ | الحكم |",
              "|---|---|---|---|---|---|---|---|---|"]
        for e in v["scopes"]:
            name = "موحّدة لكل العملات" if e["scope"] == "ALL" else e["scope"]
            if e.get("weak"):
                name += " ⚠️ (أفضل الموجود، لسا مش رابحة بالتدريب والتحقق)"
            if "coins" not in e:
                L.append(f"| {name} | — | | | | | | | لسا ما لقى استراتيجية |"); continue
            for c in e["coins"]:
                t = c["test"]
                icon = {"good": "✅", "ok": "🟡", "soft": "🟠", "bad": "❌"}[c["verdict"]["level"]]
                L.append(f"| {name} | {c['symbol']} | {t['trades']} | {t['trades_per_year']} | {t['win_rate'] * 100:.0f}% | "
                         f"{pct(t['lev']['total'])} | {t['lev']['max_dd'] * 100:.0f}% | "
                         f"{pct(t['buy_hold']['total'])} | {icon} {c['verdict']['text']} |")
        L += ["", f"![progress]({'docs/' + v['progress']})", ""]
    (C.ROOT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
