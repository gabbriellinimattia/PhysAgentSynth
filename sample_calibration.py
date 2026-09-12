"""sample_calibration.py - estrae, per famiglia di exciter, la distribuzione
REALE dei descrittori (stessa metrica di descriptor_vector_torch, quella
usata da physical_agents_train.py per il training) sui campioni audio in
sample/. Serve a calibrare random_target sul punto 10 (allargare i target
oltre la comfort zone) usando range/correlazioni osservate davvero, invece
che Dirichlet/uniform indipendenti tra i descrittori come oggi.

Uso: python3 sample_calibration.py
Scrive sample_calibration.json nella cartella corrente (non lo eseguo qui,
lo fa l'utente).

Usa analyze_reference (synth_torch_before.py), ora allineata a
descriptor_vector_torch (centroid/rolloff/e_bands pesati per energia
sull'intera registrazione via aggregate_energy_weighted in analyzer.py,
non piu' media semplice frame-per-frame) - un'unica pipeline di analisi
condivisa con AnalyzerV3, invece delle due implementazioni parallele di
prima (che rischiavano di disallinearsi a ogni modifica).

Ogni riga porta anche "extra": i descrittori di AnalyzerV3 (skewness/
kurtosis/slope/hfc/roughness/zcr/transientness/inharmonicity/jitter/
band_crest). Di questi, inharmonicity/roughness alimentano ora i target
dei proxy differenziabili in physical_agents_train.py (strike/pluck);
band_crest/zcr/hfc/transientness/jitter/n_partials_mean restano solo
metriche di valutazione POST-training (physical_agents_stress_test.py),
mai una loss - non sono differenziabili rispetto ai parametri del modello
(richiedono peak-picking/partial-tracking discreti).

flux e spread sono usati in training per TUTTE le famiglie (spread e'
stato promosso da EXTRA_KEYS a DESCRIPTOR_KEYS: vedi sopra).

NOTA su Orchidea/tam-tam: le tecniche a frizione (superball/brushes/
scraping) usano un risonatore a piastra inarmonico, non compatibile con
l'assunzione di pettine armonico di bow/blow (freq_raw congelato su k*f0,
vedi ModalBank.warm_start in physical_agents_train.py) - escluse per ora
da questa calibrazione (FAMILY_DIRS sotto non include "Orchidea").
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from synth_torch_before import analyze_reference, SR_DEFAULT

SAMPLE_ROOT = "sample"
# finestra di analisi: stessa durata di default di train_agent(seconds=1.0)
# in physical_agents_train.py, cosi' t_centroid/flux (sensibili alla durata
# del buffer) restano comparabili con quello che il training osserva.
SECONDS = 1.0

# cartella sample/<nome> -> tipo di exciter in EXCITERS (physical_agents_train.py).
FAMILY_DIRS = {
    "strike": "strike",
    "pluck": "pluck",
    "bow": "bow",
    "blow": "blow",
    "shaker": "shaker",
    "noise": "noise",
}

DESCRIPTOR_KEYS = ("harmonicity", "noisiness", "centroid", "rolloff",
                    "flatness", "flux", "t_centroid", "e_low", "e_mid", "e_high",
                    "spread")   # spread promosso da EXTRA_KEYS: ora un target di training
                                # vero (descriptor_vector_torch/random_target), non solo osservato.

# inharmonicity/roughness restano qui (extra): NON alimentano achieved{}
# direttamente in training (li' sono proxy differenziabili dai parametri
# del banco - vedi _inharmonicity_proxy/_roughness_proxy in
# physical_agents_train.py, non da questa misura audio). Servono pero'
# come range/calibrazione REALE per i target che quei proxy inseguono
# (strike/pluck, vedi random_target) - l'unico uso di summary "extra" che
# arriva davvero in training. band_crest/zcr/hfc/transientness/jitter/
# n_partials_mean restano solo metriche di valutazione post-training (vedi
# physical_agents_stress_test.py), mai un target.
EXTRA_KEYS = ("skewness", "kurtosis", "slope", "hfc", "roughness", "zcr",
              "transientness", "inharmonicity", "jitter", "n_partials_mean",
              "band_crest_20-300", "band_crest_300-3000", "band_crest_3000-20000")


def analyze_file(path, sr=SR_DEFAULT, seconds=SECONDS):
    """Wrapper sottile su analyze_reference (synth_torch_before.py, ora
    allineata a descriptor_vector_torch - vedi nota in testa al file):
    prima questo script duplicava la stessa logica chiamando
    descriptor_vector_torch direttamente, perche' analyze_reference
    passava da un'altra pipeline (AnalyzerV3) con formule non allineate -
    ora e' un'unica fonte di verita' invece di due implementazioni
    parallele da tenere sincronizzate a mano."""
    target, f0 = analyze_reference(path, seconds=seconds, sr=sr)
    out = dict(target)
    out["f0"] = f0
    return out


def collect_family(name, folder, sample_root=SAMPLE_ROOT):
    files = sorted(glob.glob(os.path.join(sample_root, folder, "**", "*.wav"), recursive=True))
    rows = []
    for path in files:
        try:
            d = analyze_file(path)
        except Exception as e:
            print(f"  [{name}] SALTATO {path}: {e}")
            continue
        d["path"] = path
        rows.append(d)
        f0_str = f"{d['f0']:.1f}Hz" if d.get("f0") is not None else "n/d"
        print(f"  [{name}] {os.path.basename(path)}: "
              f"harmonicity={d['harmonicity']:.3f} noisiness={d['noisiness']:.3f} "
              f"centroid={d.get('centroid', float('nan')):.0f}Hz f0={f0_str}")
    return rows


def summarize(rows, keys):
    """Media/std/min/max per descrittore PIU' la matrice di correlazione tra
    i descrittori: random_target oggi campiona harmonicity/noisiness/bande
    in modo indipendente (Dirichlet/uniform), ignorando correlazioni reali
    (es. alta noisiness che tipicamente coincide con e_high piu' alto) -
    questo e' il motivo per cui serve questo file, non solo i range."""
    if not rows:
        return None
    valid = [r for r in rows if all(r.get(k) is not None for k in keys)]
    if not valid:
        return None
    arr = np.array([[r[k] for k in keys] for r in valid])
    summary = {
        "n_files": int(arr.shape[0]),
        "mean": {k: float(v) for k, v in zip(keys, arr.mean(0))},
        "std": {k: float(v) for k, v in zip(keys, arr.std(0))},
        "min": {k: float(v) for k, v in zip(keys, arr.min(0))},
        "max": {k: float(v) for k, v in zip(keys, arr.max(0))},
    }
    corr = np.corrcoef(arr, rowvar=False)
    summary["correlation"] = {
        keys[i]: {keys[j]: float(corr[i, j]) for j in range(len(keys))}
        for i in range(len(keys))
    }
    return summary


def summarize_extra(rows, keys):
    """Come summarize ma solo media/std (niente correlazione): per i
    descrittori non ancora usati in training basta sapere il range tipico
    per famiglia, per ora - non servono correlazioni fini finche' non
    entrano davvero in un target/loss (vedi raccomandazione)."""
    valid = [r for r in rows if "extra" in r and all(r["extra"].get(k) is not None for k in keys)]
    if not valid:
        return None
    arr = np.array([[r["extra"][k] for k in keys] for r in valid])
    return {
        "n_files": int(arr.shape[0]),
        "mean": {k: float(v) for k, v in zip(keys, arr.mean(0))},
        "std": {k: float(v) for k, v in zip(keys, arr.std(0))},
    }


def main():
    calibration = {}
    for name, folder in FAMILY_DIRS.items():
        folder_path = os.path.join(SAMPLE_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f"[{name}] cartella non trovata: {folder_path}, salto")
            continue
        print(f"=== {name} ({folder_path}) ===")
        rows = collect_family(name, folder)
        summary = summarize(rows, DESCRIPTOR_KEYS)
        extra_summary = summarize_extra(rows, EXTRA_KEYS)
        f0_vals = [r["f0"] for r in rows if r.get("f0") is not None]
        calibration[name] = {
            "summary": summary,
            "extra_summary": extra_summary,
            "f0_range": [float(min(f0_vals)), float(max(f0_vals))] if f0_vals else None,
            # target per-file grezzi (con "extra" incluso): utili per
            # campionare direttamente dalla distribuzione empirica
            # (bootstrap), non solo dalla gaussiana fit su mean/std sopra -
            # piu' fedele se la distribuzione reale e' multimodale o
            # asimmetrica (verosimile per famiglie con tecniche esecutive
            # diverse nello stesso file .wav).
            "raw": rows,
        }
        print()

    out_path = "sample_calibration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    print(f"Scritto: {out_path}")


if __name__ == "__main__":
    main()
