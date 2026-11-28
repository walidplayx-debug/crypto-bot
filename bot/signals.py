"""Signal bot: runs the coin-rotation strategy and publishes the live signal.

Usage:  python -m bot.signals
Writes docs/data.json, docs/live.json, docs/img/rotation.png and RESULTS.md.
"""
import json
from datetime import datetime, timezone

from . import config as C
from . import rotation


def pct(x):
    return f"{x * 100:+.1f}%"


def main():
    res, r, bench = rotation.run()
    img = C.DOCS_DIR / "img"
    img.mkdir(parents=True, exist_ok=True)
    rotation.plot(r, bench, img / "rotation.png")
    out = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "split": {"train_end": C.TRAIN_END, "valid_end": C.VALID_END},
           "rotation": res, "chart": "img/rotation.png"}
    (C.DOCS_DIR / "data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    L, b = res["live"], res["best"]
    side = {"long": "🟢 شراء", "short": "🔴 بيع"}
    lines = ["# 📡 إشارات التناوب", "", f"آخر تحديث: **{out['updated']}**", "",
             f"## الإشارة الحالية (إغلاق {L['as_of']})", ""]
    if L["positions"]:
        lines += ["| | العملة | الحجم | السعر وقت الإشارة |", "|---|---|---|---|"]
        lines += [f"| {side[x['side']]} | {x['symbol']} | {round(x['weight'] * 100)}% | {x['price']} |" for x in L["positions"]]
    else:
        lines.append("💤 كاش، ما في صفقات مفتوحة")
    p = L["perf"]
    lines += ["", f"المراجعة الجاية: **{L['next_signal']}**، بعد الساعة 00:00 UTC", "",
              f"🧪 الاختبار الحي من {L['start']}: " + (f"**{pct(p['total'])}** خلال {p['days']} يوم، "
              f"وأقصى تراجع {p['max_dd'] * 100:.0f}%" if p["days"] else "لسا بلّش"), "",
              "## القواعد", "", *[f"- {x}" for x in L["rules"]], "",
              "## الباكتست", "",
              "| الفترة | الربح | أقصى تراجع | صفقات | رابحة من 100 | اشترِ واحتفظ (الخمسة) |", "|---|---|---|---|---|---|"]
    for k, n in (("train", "تدريب"), ("valid", "تحقق"), ("test", "**مخفي**")):
        m = b[k]
        lines.append(f"| {n} | {pct(m['total'])} | {m['max_dd'] * 100:.0f}% | {m.get('trades', 0)} | "
                     f"{m.get('win_rate', 0) * 100:.0f} | {pct(m['bench'])} |")
    if res.get("compare"):
        lines += ["", "## شو عمل كل تحسين", "", "الاختيار صار على التقييم (تدريب + تحقق) بس.", "",
                  "| | التقييم | ربح المخفي | أقصى تراجع |", "|---|---|---|---|"]
        lines += [f"| {'✅ ' if x['chosen'] else ''}{x['name']} | {x['robust']:.3f} | {pct(x['test_total'])} | {x['test_dd'] * 100:.0f}% |"
                  for x in res["compare"]]
    lines += ["", "![equity](docs/img/rotation.png)", "",
              "> ⚠️ اختبار على بيانات تاريخية وحيّة على الورق. مش ضمان للمستقبل، وما هو نصيحة مالية."]
    (C.ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("signal:", [(x["symbol"], x["side"]) for x in L["positions"]] or "cash", "| next:", L["next_signal"])


if __name__ == "__main__":
    main()
