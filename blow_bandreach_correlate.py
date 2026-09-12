"""blow_bandreach_correlate.py - verifica se il gap harmonicity/noisiness su
blow e' spiegato dalla "reachability" delle bande in _tone_amp_profile: per
note molto gravi le armoniche (k*f0, k=1..n_harm) possono non raggiungere
mai la banda 3000-20000Hz (e_high target "perso" e ridistribuito sulle
bande raggiungibili, vedi codice), per note molto acute puo' essere la
banda 20-300Hz a non avere armoniche (gia' il fondamentale f0 sta sopra
300Hz). In entrambi i casi l'ampiezza per-armonica imposta al tono non
rispecchia piu' il target reale - ipotesi strutturale sul meccanismo
proprio di blow (_tone_amp_profile), non presa in prestito da bow.

Segue: floor/cliff su roughness (escluso), gradient routing bow (peggiora),
flutter-tongue (escluso), varianza da restart (escluso) - blow_roughness_
correlate.py, blow_flatt_split_test.py, impulsive_batch_stress_test.py
--restarts 3.

Uso: python3 blow_bandreach_correlate.py [--n 40] [--seed 0] [--iters 300] [--restarts 1]
"""
import argparse
import glob
import os
import random

import numpy as np
import torch

from physical_agents_train import train_agent_best, _n_harm_feedback
from synth_torch_before import SR_DEFAULT, analyze_reference

EPS = 1e-9


def list_files(sample_root, exciter_type, n, seed):
    files = sorted(glob.glob(os.path.join(sample_root, exciter_type, "**", "*.wav"), recursive=True))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def band_reach(f0, n_harm, target):
    """Replica la logica di reachability di _tone_amp_profile senza
    duplicarne il calcolo dell'ampiezza: quali bande hanno almeno
    un'armonica, e quale frazione di energia target cade in bande NON
    raggiungibili (persa e ridistribuita)."""
    k = torch.arange(1, n_harm + 1, dtype=torch.float32)
    freqs_hz = k * float(f0)
    band_of = torch.bucketize(freqs_hz, torch.tensor([300.0, 3000.0]))
    counts = [int((band_of == b).sum()) for b in range(3)]
    e_band = [float(target.get("e_low", 0.0)), float(target.get("e_mid", 0.0)), float(target.get("e_high", 0.0))]
    total = sum(e_band) + EPS
    lost = sum(e for c, e in zip(counts, e_band) if c == 0)
    return counts, lost / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", default="sample")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=1)
    args = ap.parse_args()

    files = list_files(args.sample_root, "blow", args.n, args.seed)
    rows = []
    for i, path in enumerate(files, 1):
        target, f0 = analyze_reference(path, seconds=args.seconds, sr=SR_DEFAULT)
        n_harm = _n_harm_feedback(f0, target)
        counts, lost_frac = band_reach(f0, n_harm, target)

        exciter, modal, achieved, loss = train_agent_best(
            "blow", target, f0=f0, restarts=args.restarts, iters=args.iters,
            seconds=args.seconds, log=None)
        h_t, h_a = float(target["harmonicity"]), float(achieved["harmonicity"])
        gap = abs(h_a - h_t)
        rows.append((os.path.basename(path), f0, n_harm, counts, lost_frac, h_t, h_a, gap))
        print(f"[{i}/{len(files)}] {os.path.basename(path):40s} f0={f0:6.1f} n_harm={n_harm:3d} "
              f"counts(low/mid/high)={counts} lost_frac={lost_frac:.3f} "
              f"h_target={h_t:.3f} h_achieved={h_a:.3f} gap={gap:.3f}", flush=True)

    gaps = np.array([r[7] for r in rows])
    lost = np.array([r[4] for r in rows])
    f0s = np.array([r[1] for r in rows])
    zero_high = np.array([1.0 if r[3][2] == 0 else 0.0 for r in rows])
    zero_low = np.array([1.0 if r[3][0] == 0 else 0.0 for r in rows])

    print("\n=== RIEPILOGO ===")
    print(f"gap medio={gaps.mean():.3f}")
    print(f"corr(lost_frac, gap)={np.corrcoef(lost, gaps)[0,1]:.3f}")
    print(f"corr(f0, gap)={np.corrcoef(f0s, gaps)[0,1]:.3f}")
    print(f"gap medio con banda ALTA irraggiungibile (n={int(zero_high.sum())}): "
          f"{gaps[zero_high==1].mean() if zero_high.sum() else float('nan'):.3f}  "
          f"vs raggiungibile (n={int((1-zero_high).sum())}): {gaps[zero_high==0].mean():.3f}")
    print(f"gap medio con banda BASSA irraggiungibile (n={int(zero_low.sum())}): "
          f"{gaps[zero_low==1].mean() if zero_low.sum() else float('nan'):.3f}  "
          f"vs raggiungibile (n={int((1-zero_low).sum())}): {gaps[zero_low==0].mean():.3f}")
    worst = sorted(rows, key=lambda r: -r[7])[:8]
    print("peggiori 8 per gap (file, f0, counts, lost_frac, gap):")
    for r in worst:
        print(f"  {r[0]:40s} f0={r[1]:6.1f} counts={r[3]} lost_frac={r[4]:.3f} gap={r[7]:.3f}")


if __name__ == "__main__":
    main()
