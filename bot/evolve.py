"""Genetic algorithm: invents strategies, tests them, keeps the best, mixes and mutates them.

Usage:  python -m bot.evolve --variant 4h --minutes 40
It resumes from state/<variant>/<scope>.json, so every run continues where the last stopped.
Scopes: ALL (one strategy that must work on all five coins) + one per coin.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from . import config as C
from .engine import build, fitness

_DATA = {}   # variant -> list of datasets (shared with worker processes via fork)


# ---------------------------------------------------------------- genomes

def hold_range(variant):
    return (2, 60) if variant == "4h" else (4, 240)


def rand_rule(names):
    return [random.choice(names), random.choice("<>"), random.randint(5, 95)]


def normalize(g):
    """One condition per indicator per side, at most MAX_RULES, never an empty strategy."""
    for side in ("long", "short"):
        seen, keep = set(), []
        for r in g[side]:
            if r[0] not in seen:
                seen.add(r[0]); keep.append(r)
        g[side] = keep[:C.MAX_RULES]
    return g


def rand_genome(names, variant):
    kind = random.random()          # 50% both directions, 25% long only, 25% short only
    nl = random.randint(1, C.MAX_RULES) if kind < 0.75 else 0
    ns = random.randint(1, C.MAX_RULES) if kind >= 0.25 else 0
    lo, hi = hold_range(variant)
    return normalize({
        "long": [rand_rule(names) for _ in range(nl)],
        "short": [rand_rule(names) for _ in range(ns)],
        "sl": round(random.uniform(0.5, 5.0), 2),
        "tp": round(random.uniform(0.5, 10.0), 2),
        "hold": random.randint(lo, hi),
    })


def key(g):
    return json.dumps(g, sort_keys=True)


def crossover(a, b):
    child = {}
    for side in ("long", "short"):
        pool = a[side] + b[side]
        if not pool or (not a[side] and not b[side]):
            child[side] = []
            continue
        src = random.choice([a, b])[side]
        k = len(src) if src else 0
        child[side] = [list(r) for r in random.sample(pool, min(k, len(pool)))] if k else []
    for p in ("sl", "tp", "hold"):
        w = random.random()
        child[p] = a[p] * w + b[p] * (1 - w)
    child["sl"], child["tp"] = round(child["sl"], 2), round(child["tp"], 2)
    child["hold"] = int(round(child["hold"]))
    if not child["long"] and not child["short"]:
        child["long"] = [list(r) for r in (a["long"] or a["short"])][:1]
    return normalize(child)


def mutate(g, names, variant, rate=0.3):
    g = json.loads(json.dumps(g))
    lo, hi = hold_range(variant)
    for side in ("long", "short"):
        for r in g[side]:
            if random.random() < rate:
                r[2] = int(np.clip(r[2] + random.gauss(0, 8), 1, 99))
            if random.random() < rate * 0.2:
                r[1] = "<" if r[1] == ">" else ">"
            if random.random() < rate * 0.2:
                r[0] = random.choice(names)
        if g[side] and random.random() < rate * 0.2:
            g[side].pop(random.randrange(len(g[side])))
        if len(g[side]) < C.MAX_RULES and random.random() < rate * 0.2:
            g[side].append(rand_rule(names))
    if not g["long"] and not g["short"]:
        g["long"] = [rand_rule(names)]
    if random.random() < rate:
        g["sl"] = round(float(np.clip(g["sl"] * np.exp(random.gauss(0, 0.2)), 0.3, 6)), 2)
    if random.random() < rate:
        g["tp"] = round(float(np.clip(g["tp"] * np.exp(random.gauss(0, 0.2)), 0.3, 12)), 2)
    if random.random() < rate:
        g["hold"] = int(np.clip(g["hold"] * np.exp(random.gauss(0, 0.25)), lo, hi))
    if random.random() < rate * 0.1:            # occasionally open the other direction
        other = "short" if not g["short"] else ("long" if not g["long"] else None)
        if other:
            g[other] = [rand_rule(names)]
    return normalize(g)


# ---------------------------------------------------------------- evaluation

def _eval(args):
    variant, scope, g, part = args
    ds = [d for d in _DATA[variant] if scope == "ALL" or d.symbol == scope]
    try:
        return fitness(ds, g, part)
    except Exception:
        return -9.0


def tournament(pop, fit, k=3):
    best = max(random.sample(range(len(pop)), k), key=lambda i: fit[i])
    return pop[best]


def run_scope(pool, variant, scope, seconds):
    path = C.STATE_DIR / variant / f"{scope}.json"
    names = _DATA[variant][0].names
    if path.exists():
        st = json.loads(path.read_text())
    else:
        st = {"variant": variant, "scope": scope, "generation": 0, "population": [],
              "champion": None, "history": [], "since_improve": 0}
    pop = [normalize(g) for g in st["population"] if all(r[0] in names for s in ("long", "short") for r in g[s])]
    while len(pop) < C.POPULATION:
        pop.append(rand_genome(names, variant))
    champ = st["champion"]
    seen = set()
    t0 = time.time()
    gens = 0
    while time.time() - t0 < seconds:
        fit = pool.map(_eval, [(variant, scope, g, "train") for g in pop])
        order = np.argsort(fit)[::-1]
        # validation check for the best few new ones
        cand = [pop[i] for i in order[:6] if fit[i] > 0 and key(pop[i]) not in seen]
        improved = False
        if cand:
            vfit = pool.map(_eval, [(variant, scope, g, "valid") for g in cand])
            for g, v in zip(cand, vfit):
                seen.add(key(g))
                tf = fit[pop.index(g)]
                robust = min(tf, v)   # must be good on BOTH training and validation
                if v > 0 and (champ is None or robust > champ["robust"] + C.CHAMPION_MARGIN):
                    champ = {"genome": g, "train_fitness": round(tf, 4), "valid_fitness": round(v, 4),
                             "robust": round(robust, 4), "found_generation": st["generation"]}
                    improved = True
        st["since_improve"] = 0 if improved else st["since_improve"] + 1
        st["generation"] += 1
        gens += 1
        st["history"].append([st["generation"], round(float(fit[order[0]]), 4),
                              champ["robust"] if champ else None])
        # next generation: elites + children + a few random newcomers for diversity
        elite = [pop[i] for i in order[:C.ELITE]]
        nxt = [json.loads(json.dumps(g)) for g in elite]
        if champ:
            nxt.append(json.loads(json.dumps(champ["genome"])))
        n_new = C.POPULATION // 10
        while len(nxt) < C.POPULATION - n_new:
            a, b = tournament(pop, fit), tournament(pop, fit)
            nxt.append(mutate(crossover(a, b), names, variant))
        while len(nxt) < C.POPULATION:
            nxt.append(rand_genome(names, variant))
        pop = nxt
    st["population"] = pop
    st["champion"] = champ
    st["history"] = st["history"][-3000:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st))
    cv = champ["valid_fitness"] if champ else None
    print(f"[{variant} {scope}] +{gens} generations (total {st['generation']}), champion valid={cv}, "
          f"no improvement for {st['since_improve']} gens", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=C.VARIANTS)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--scopes", default="ALL," + ",".join(C.SYMBOLS))
    args = ap.parse_args()
    random.seed()
    t = time.time()
    _DATA[args.variant] = [build(s, args.variant) for s in C.SYMBOLS]
    print(f"datasets ready in {time.time() - t:.0f}s", flush=True)
    scopes = args.scopes.split(",")
    budget = args.minutes * 60 - (time.time() - t)
    # the all-coins strategy is 5x more work, so it gets half of the time
    weights = [len(C.SYMBOLS) if s == "ALL" else 1 for s in scopes]
    ctx = mp.get_context("fork")
    with ctx.Pool(os.cpu_count()) as pool:
        for s, w in zip(scopes, weights):
            run_scope(pool, args.variant, s, budget * w / sum(weights))


if __name__ == "__main__":
    main()
