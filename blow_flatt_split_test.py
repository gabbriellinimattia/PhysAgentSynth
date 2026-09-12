"""blow_flatt_split_test.py - conferma su campione piu' ampio l'effect size
flutter-tongue su blow: batch separati flatt vs non-flatt (stessa
convenzione sorted-glob+shuffle seed=0 di impulsive_batch_stress_test.py),
confronto gap medio harmonicity/noisiness. Segue la riaggregazione manuale
su n=40 (flatt 0.200 n=13 vs non-flatt 0.116 n=27, blow_roughness_
correlate.py) con due gruppi bilanciati e piu' ampi, prima di progettare
qualunque parametro di modulazione dedicato al flutter-tongue (rullata di
lingua/ugola, ~20-30Hz, mai modellata in BlowExciter - vedi analisi:
BlowExciter non ha equivalente di mod_rate/mod_depth di BowExciter).

Uso: python3 blow_flatt_split_test.py [--n 20] [--seed 0] [--iters 300] [--restarts 1]
"""
import argparse
import glob
import os
import random

import numpy as np

from physical_agents_stress_test import eval_errors, _analyze_extra, EVAL_KEYS, EVAL_BAND_KEYS
from physical_agents_train import train_agent_best, render
from synth_torch_before import SR_DEFAULT, analyze_reference


def list_files(sample_root, exciter_type, seed, predicate=None):
    files = sorted(glob.glob(os.path.join(sample_root, exciter_type, "**", "*.wav"), recursive=True))
    if predicate is not None:
        files = [f for f in files if predicate(os.path.basename(f))]
    rng = random.Random(seed)
    rng.shuffle(files)
    return files


def run_one(path, exciter_type, seconds, restarts, iters):
    target, f0 = analyze_reference(path, seconds=seconds, sr=SR_DEFAULT)
    ref_extra = target.get("extra", {})
    exciter, modal, achieved, loss = train_agent_best(
        exciter_type, target, f0=f0, restarts=restarts, iters=iters, seconds=seconds, log=None)
    y = render(exciter, modal, seconds)
    agent_extra = _analyze_extra(y, sr=SR_DEFAULT)
    errs = eval_errors(ref_extra, agent_extra)
    return {
        "path": path, "loss": float(loss),
        "h_target": float(target["harmonicity"]), "h_achieved": float(achieved["harmonicity"]),
        "n_target": float(target["noisiness"]), "n_achieved": float(achieved["noisiness"]),
        "errs": errs,
    }


def summarize(label, results):
    h_gap = np.array([abs(r["h_achieved"] - r["h_target"]) for r in results])
    losses = np.array([r["loss"] for r in results])
    print(f"\n=== {label} (n={len(results)}) ===")
    print(f"  loss: mean={losses.mean():.3f} max={losses.max():.3f}")
    print(f"  |harmonicity gap|: mean={h_gap.mean():.3f} max={h_gap.max():.3f} std={h_gap.std():.3f}")
    for k in EVAL_BAND_KEYS:
        vals = np.array([r["errs"][k] for r in results if k in r["errs"]])
        if len(vals):
            print(f"  {k:24s} abs_err: mean={vals.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", default="sample")
    ap.add_argument("--n", type=int, default=20, help="file per gruppo (flatt e non-flatt)")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flatt_all = list_files(args.sample_root, "blow", args.seed, predicate=lambda b: "flatt" in b.lower())
    nonflatt_all = list_files(args.sample_root, "blow", args.seed, predicate=lambda b: "flatt" not in b.lower())
    print(f"flatt disponibili: {len(flatt_all)}  non-flatt disponibili: {len(nonflatt_all)}")

    flatt_files = flatt_all[:args.n]
    nonflatt_files = nonflatt_all[:args.n]

    groups = {}
    for label, files in [("flatt", flatt_files), ("non-flatt", nonflatt_files)]:
        results = []
        for i, path in enumerate(files, 1):
            r = run_one(path, "blow", args.seconds, args.restarts, args.iters)
            results.append(r)
            gap = abs(r["h_achieved"] - r["h_target"])
            print(f"[{label}] ({i}/{len(files)}) {os.path.basename(path):40s} "
                  f"loss={r['loss']:.3f} h_target={r['h_target']:.3f} h_achieved={r['h_achieved']:.3f} gap={gap:.3f}", flush=True)
        groups[label] = results
        summarize(label, results)

    print("\n=== RIEPILOGO ===")
    for label, results in groups.items():
        h_gap = np.mean([abs(r["h_achieved"] - r["h_target"]) for r in results])
        print(f"  {label:10s} |harmonicity gap| medio = {h_gap:.3f}  (n={len(results)})")


if __name__ == "__main__":
    main()
