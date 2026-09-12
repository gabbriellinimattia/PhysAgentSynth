"""impulsive_batch_stress_test.py - batch di physical_agents_stress_test.py su
PIU' file reali per strike/pluck/shaker (non un file a caso per run), con
riepilogo aggregato: harmonicity/noisiness achieved vs target (il problema
appena corretto per strike, vedi StrikeExciter/FIX 2026-09-07) + i
descrittori mai in loss (band_crest x3/hfc/zcr/transientness/jitter/
n_partials_mean, come physical_agents_stress_test.py). Un solo comando per
vedere se restano problemi sugli eccitatori impulsivi su un campione ampio
di sample reali, non su un singolo file.

Uso:
  python3 impulsive_batch_stress_test.py
  python3 impulsive_batch_stress_test.py --types strike,pluck --n 8 --iters 300 --restarts 1 --seed 0
"""
import argparse
import glob
import os
import random

import numpy as np

from physical_agents_stress_test import eval_errors, _analyze_extra, EVAL_KEYS, EVAL_BAND_KEYS
from physical_agents_train import train_agent_best, render
from synth_torch_before import SR_DEFAULT, analyze_reference

IMPULSIVE_TYPES = ("strike", "pluck", "shaker")


def list_files(sample_root, exciter_type, n, seed):
    files = sorted(glob.glob(os.path.join(sample_root, exciter_type, "**", "*.wav"), recursive=True))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def run_one(path, exciter_type, seconds, restarts, iters):
    target, f0 = analyze_reference(path, seconds=seconds, sr=SR_DEFAULT)
    ref_extra = target.get("extra", {})
    exciter, modal, achieved, loss = train_agent_best(
        exciter_type, target, f0=f0, restarts=restarts, iters=iters, seconds=seconds, log=print)
    y = render(exciter, modal, seconds)
    agent_extra = _analyze_extra(y, sr=SR_DEFAULT)
    errs = eval_errors(ref_extra, agent_extra)
    return {
        "path": path, "loss": float(loss),
        "h_target": float(target["harmonicity"]), "h_achieved": float(achieved["harmonicity"]),
        "n_target": float(target["noisiness"]), "n_achieved": float(achieved["noisiness"]),
        "errs": errs,
    }


def summarize(etype, results):
    h_gap = np.array([abs(r["h_achieved"] - r["h_target"]) for r in results])
    n_gap = np.array([abs(r["n_achieved"] - r["n_target"]) for r in results])
    losses = np.array([r["loss"] for r in results])
    print(f"\n=== {etype}  (n={len(results)}) ===")
    print(f"  loss: mean={losses.mean():.3f} max={losses.max():.3f}")
    print(f"  |harmonicity achieved-target|: mean={h_gap.mean():.3f} max={h_gap.max():.3f}")
    print(f"  |noisiness   achieved-target|: mean={n_gap.mean():.3f} max={n_gap.max():.3f}")
    for k in EVAL_KEYS:
        vals = np.array([r["errs"][k] for r in results if k in r["errs"]])
        if len(vals):
            print(f"  {k:16s} rel_err: mean={vals.mean():.3f} max={vals.max():.3f}")
    for k in EVAL_BAND_KEYS:
        vals = np.array([r["errs"][k] for r in results if k in r["errs"]])
        if len(vals):
            print(f"  {k:24s} abs_err: mean={vals.mean():.3f} max={vals.max():.3f}")
    worst = max(results, key=lambda r: abs(r["h_achieved"] - r["h_target"]))
    print(f"  peggiore su harmonicity: {os.path.basename(worst['path'])}"
          f"  target={worst['h_target']:.3f} achieved={worst['h_achieved']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default=",".join(IMPULSIVE_TYPES))
    ap.add_argument("--sample-root", default="sample")
    ap.add_argument("--n", type=int, default=6, help="file reali per tipo")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    types = args.types.split(",")
    all_results = {}
    for etype in types:
        files = list_files(args.sample_root, etype, args.n, args.seed)
        if not files:
            print(f"[{etype}] nessun file trovato sotto {args.sample_root}/{etype}, salto")
            continue
        results = []
        for i, path in enumerate(files, 1):
            print(f"[{etype}] ({i}/{len(files)}) avvio {os.path.basename(path)}", flush=True)
            r = run_one(path, etype, args.seconds, args.restarts, args.iters)
            results.append(r)
            print(f"[{etype}] {os.path.basename(path):40s} loss={r['loss']:.3f}  "
                  f"h target={r['h_target']:.3f} achieved={r['h_achieved']:.3f}  "
                  f"n target={r['n_target']:.3f} achieved={r['n_achieved']:.3f}")
        all_results[etype] = results
        summarize(etype, results)

    print("\n=== RIEPILOGO ===")
    for etype, results in all_results.items():
        h_gap = np.mean([abs(r["h_achieved"] - r["h_target"]) for r in results])
        print(f"  {etype:8s} |harmonicity gap| medio = {h_gap:.3f}  (n={len(results)})")


if __name__ == "__main__":
    main()
