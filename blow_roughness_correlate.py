"""blow_roughness_correlate.py - diagnosi collasso harmonicity/noisiness su
blow: per ciascuno degli n file (stessa convenzione sorted-glob+shuffle
seed=0 di impulsive_batch_stress_test.py), stampa target/achieved h/n, gap,
e i parametri finali dell'exciter (roughness, focus_bw, pressure) - per
verificare se il gap e' dovuto a roughness_raw che satura indipendentemente
dal target (stesso sospetto "cliff" gia' diagnosticato su bow prima del fix
tremolo, mai ancora verificato su blow: BlowExciter non ha ne' hf_roughness
ne' noise_floor).

Uso: python3 blow_roughness_correlate.py [--n 40] [--seed 0] [--iters 300] [--restarts 1]
"""
import argparse
import glob
import os
import random

import numpy as np

from physical_agents_train import train_agent_best
from synth_torch_before import SR_DEFAULT, analyze_reference


def list_files(sample_root, exciter_type, n, seed):
    files = sorted(glob.glob(os.path.join(sample_root, exciter_type, "**", "*.wav"), recursive=True))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


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
        exciter, modal, achieved, loss = train_agent_best(
            "blow", target, f0=f0, restarts=args.restarts, iters=args.iters,
            seconds=args.seconds, log=None)
        p = exciter.params()
        roughness = float(p["roughness"])
        focus_bw = float(p["focus_bw"])
        pressure = float(p["pressure"])
        h_t, h_a = float(target["harmonicity"]), float(achieved["harmonicity"])
        gap = abs(h_a - h_t)
        rows.append((os.path.basename(path), h_t, h_a, gap, roughness, focus_bw, pressure))
        print(f"[{i}/{len(files)}] {os.path.basename(path):40s} "
              f"h_target={h_t:.3f} h_achieved={h_a:.3f} gap={gap:.3f} "
              f"roughness={roughness:.4f} focus_bw={focus_bw:.1f} pressure={pressure:.3f}", flush=True)

    gaps = np.array([r[3] for r in rows])
    roughs = np.array([r[4] for r in rows])
    h_t = np.array([r[1] for r in rows])
    h_a = np.array([r[2] for r in rows])

    print("\n=== RIEPILOGO ===")
    print(f"gap medio={gaps.mean():.3f}")
    print(f"roughness finale: mean={roughs.mean():.4f} std={roughs.std():.4f}  (std piccola rispetto a mean -> indizio di floor/attrattore)")
    print(f"corr(roughness, gap)={np.corrcoef(roughs, gaps)[0,1]:.3f}")
    print(f"corr(h_target, h_achieved)={np.corrcoef(h_t, h_a)[0,1]:.3f}  (vicino a 0 -> achieved quasi indipendente dal target)")
    worst = sorted(rows, key=lambda r: -r[3])[:8]
    print("peggiori 8 per gap:")
    for r in worst:
        print(f"  {r[0]:40s} h_target={r[1]:.3f} h_achieved={r[2]:.3f} gap={r[3]:.3f} roughness={r[4]:.4f}")


if __name__ == "__main__":
    main()
