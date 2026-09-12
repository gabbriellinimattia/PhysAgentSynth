"""physical_agents_stress_test.py - allena un agente fisico (physical_agents_
train.py) su un file REALE campionato da sample/ e confronta l'agente
allenato col riferimento SOLO sui descrittori che non entrano mai nella
loss di training: band_crest (3 bande), zcr, hfc, transientness, jitter,
n_partials_mean (da AnalyzerV3, tramite aggregate_energy_weighted -
analyzer.py). Non sono differenziabili rispetto ai parametri del modello
(richiedono peak-picking/partial-tracking discreti su spettro/segnale
reso), quindi restano fuori dal training per costruzione (vedi
sample_calibration.py, sezione EXTRA_KEYS) - qui servono a rispondere a
una domanda diversa dalla loss: un agente che converge bene sui
descrittori-target (centroid/rolloff/bande/harmonicity/noisiness/flux/
t_centroid/spread, +inharmonicity/roughness per strike/pluck) risulta
anche timbricamente plausibile su assi che non ha mai visto in training?

Uso: python3 physical_agents_stress_test.py [path/al/file.wav]
Senza argomenti, pesca un file random da sample/ (vedi pick_random_file
sotto) e ne deduce l'exciter_type dalla cartella di primo livello (deve
combaciare con una chiave di EXCITERS).
"""
import glob
import os
import random
import sys

import numpy as np

import torch

from analyzer import AnalyzerV3, aggregate_energy_weighted
from physical_agents_train import EXCITERS, train_agent_best, render, _diagnose_run
from synth_torch_before import SR_DEFAULT, N_FFT_DEFAULT, analyze_reference

EPS = 1e-9


def pick_random_file(sample_root="sample", seed=None):
    """Pesca una cartella random di primo livello dentro sample_root, poi
    un file .wav random al suo interno (ricorsivo su eventuali sottocartelle).
    Inlined da sample_stress_test.py (rimosso: pipeline SNT legacy
    mixer/sineSynth/noiseSynth/transientSynth/resynth_loop, superata dal
    synth fisico-modale - questa era l'unica funzione ancora usata da li')."""
    rng = random.Random(seed)
    dirs = [d for d in os.listdir(sample_root)
            if os.path.isdir(os.path.join(sample_root, d))]
    if not dirs:
        raise FileNotFoundError(f"nessuna sottocartella trovata in {sample_root}")
    rng.shuffle(dirs)
    for d in dirs:
        files = glob.glob(os.path.join(sample_root, d, "**", "*.wav"), recursive=True)
        if files:
            return rng.choice(files)
    raise FileNotFoundError(f"nessun file .wav trovato sotto {sample_root}")

EVAL_KEYS = ("zcr", "hfc", "transientness", "jitter", "n_partials_mean")
EVAL_BAND_KEYS = ("band_crest_20-300", "band_crest_300-3000", "band_crest_3000-20000")


def _analyze_extra(y, sr=SR_DEFAULT, n_fft=N_FFT_DEFAULT):
    """Stessa pipeline di analyze_reference (synth_torch_before.py) ma su
    un buffer gia' in memoria (il render dell'agente allenato), non un
    file - qui interessa solo il sotto-dict "extra" (i descrittori di
    valutazione), non i target usati per il training."""
    analyzer = AnalyzerV3(sample_rate=sr, blocksize=n_fft)
    frames = [analyzer.process_frame(c, include_spectrum=True)
              for c in analyzer.ring_buffer.push(y.astype(np.float32))]
    if not frames:
        return {}
    agg = aggregate_energy_weighted(frames, hop=analyzer.hop, sr=sr, n_fft=analyzer.n_fft)
    return agg.get("extra", {})


def eval_errors(ref_extra, agent_extra):
    def rel_err(a, b):
        return float(abs(a - b) / (abs(a) + EPS))

    def abs_err(a, b):
        return float(abs(a - b))

    out = {}
    for k in EVAL_KEYS:
        if k in ref_extra and k in agent_extra:
            out[k] = rel_err(ref_extra[k], agent_extra[k])
    for k in EVAL_BAND_KEYS:
        if k in ref_extra and k in agent_extra:
            out[k] = abs_err(ref_extra[k], agent_extra[k])
    return out


def main(path=None, sample_root="sample", seed=None, seconds=1.0, restarts=3, iters=300):
    if path is None:
        path = pick_random_file(sample_root, seed=seed)
    rel = os.path.relpath(path, sample_root)
    exciter_type = rel.split(os.sep)[0]
    if exciter_type not in EXCITERS:
        raise ValueError(f"cartella '{exciter_type}' non combacia con nessun EXCITERS ({list(EXCITERS)}): {path}")

    print(f"[{path}] exciter_type={exciter_type}\n")
    target, f0 = analyze_reference(path, seconds=seconds, sr=SR_DEFAULT)
    ref_extra = target.get("extra", {})

    exciter, modal, achieved, loss = train_agent_best(
        exciter_type, target, f0=f0, restarts=restarts, iters=iters, seconds=seconds, log=print)
    print(f"\n  loss finale training={loss:.4f}")

    y = render(exciter, modal, seconds)
    agent_extra = _analyze_extra(y, sr=SR_DEFAULT)

    errs = eval_errors(ref_extra, agent_extra)
    print("\n-- VALUTAZIONE POST-TRAINING (mai in loss) --")
    for k in EVAL_KEYS:
        if k in errs:
            print(f"   {k:16s} rel_err={errs[k]:.4f}  (ref={ref_extra[k]:.4f} agente={agent_extra[k]:.4f})")
    for k in EVAL_BAND_KEYS:
        if k in errs:
            print(f"   {k:24s} abs_err={errs[k]:.4f}  (ref={ref_extra[k]:.4f} agente={agent_extra[k]:.4f})")

    # diagnosi Q/gain/tail_mix per modo - vedi indagine harmonicity overshoot/
    # undershoot tt-tt_edge/tt-tymp_edge (2026-09-08): serve a distinguere se
    # un h fuori target e' dovuto a tail_noise_raw (leva nuova) o a Q/gain
    # (leva preesistente) senza dover rilanciare tutto sotto single_train.py.
    _diagnose_run(exciter_type, achieved, target, modal, f0, log=print)
    if getattr(modal, "use_tail_noise", False):
        tail_mix_raw = float(torch.sigmoid(modal.tail_noise_raw))
        print(f"  tail_mix (sigmoid grezzo)={tail_mix_raw:.3f}  effettivo (cap 0.4)={0.4 * tail_mix_raw:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
