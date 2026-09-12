"""predictor_dataset_gen.py - genera il dataset di distillazione per
resonator_predictor.py: per ogni file .wav selezionato sotto sample/
allena il modello fisico (train_agent_best, la
pipeline gia' in uso) e salva la coppia (target, f0, parametri
convergenti). Il predittore imparera' a predire QUESTI parametri da
QUESTI target - distillazione, non spectral matching diretto sull'audio.

Usiamo anche le tecniche fortemente inarmoniche (tam-tam graffiato,
multifonici, ecc): il predittore deve imparare anche quella distribuzione
reale, non solo i target sintetici Dirichlet di random_target.

SELEZIONE FILE - nessun file viene MAI saltato per tipo (anche tecniche
percussive su strumenti a fiato come "kiss": vengono comunque allenate,
solo con l'architettura piu' vicina disponibile), ma ogni CARTELLA FINALE
(quella che contiene direttamente i .wav, a qualunque profondita') e'
SOTTOCAMPIONATA a --per-folder file casuali (default 4, seed fisso per
riproducibilita') - SEMPRE, incluse le cartelle famiglia (strike/pluck/
bow/blow/shaker/noise/inharmonic): dopo la riorganizzazione manuale (ex
Orchidea SOL redistribuita nei tipi exciter, non piu' una cartella a se')
anche loro contengono ora tante sottocartelle/tecniche diverse - stessa
logica di prima: tante note della stessa tecnica, 4 campioni bastano.
Nessuna eccezione piu' per cartella "curata" vs "esterna".

Il tipo exciter e' inferito dal primo livello di sample/ (match diretto
con strike|pluck|bow|blow|shaker|noise, "inharmonic" instradato al nuovo
agente "chaotic" - ChaoticExciter + ModalBank.use_coupling,
physical_agents_train.py, accoppiamento non lineare tra modi, l'unica
architettura pensata per tam-tam graffiato/death whistle/scream/
multifonici) o da parole chiave nel path per tutto cio' che non e' sotto
una di queste 7 cartelle (KEYWORD_TYPE_MAP), default "strike" (banco
modale libero, il piu' adatto a materiale complesso/inarmonico non altrimenti
classificato).

MENO CORREZIONI per file (restarts/iters ridotti rispetto al default di
train_agent_best): qui serve solo un esempio approssimativo per il
predittore, non un risultato finale di produzione.

RIPRENDIBILE, POCA MEMORIA: ogni risultato e' scritto SUBITO in
predictor_dataset.jsonl (una riga per file, append + flush su disco - mai
tenuto tutto in RAM, mai un JSON unico scritto solo a fine corsa:
un'interruzione a meta' non perde nulla). --resume salta i file gia'
presenti nel .jsonl. predictor_dataset.json (il formato che
resonator_predictor.py si aspetta) viene rigenerato dal .jsonl ogni
--flush-every file (default 10, alzato per chi ha poca RAM disponibile
sulla macchina - la rigenerazione rilegge il .jsonl e riscrive il .json,
non accumula nulla di persistente in piu' nel processo) E alla fine.

Uso: python3 predictor_dataset_gen.py [--iters 100] [--restarts 1]
     [--per-folder 4] [--resume]
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from physical_agents_train import EXCITERS, train_agent_best
from resonator_predictor import flatten_params
from synth_torch_before import analyze_reference, SR_DEFAULT

SAMPLE_ROOT = "sample"
JSONL_PATH = "predictor_dataset.jsonl"

# le cartelle famiglia (physical_agents_train.EXCITERS): usate SOLO per
# instradare al tipo exciter giusto (infer_exciter_type sotto) - non piu'
# per decidere se sottocampionare, vedi select_files (--per-folder si
# applica sempre, a ogni cartella finale, famiglia o no).
FAMILY_TYPES = tuple(EXCITERS.keys())

# cartelle il cui nome non corrisponde a una chiave di EXCITERS: mappa nome
# cartella -> exciter type. Oggi solo "inharmonic" -> "chaotic" (vedi
# ChaoticExciter, physical_agents_train.py).
FOLDER_TYPE_OVERRIDE = {"inharmonic": "chaotic"}

# parole chiave (case-insensitive, cercate ovunque nel path relativo) per
# instradare file fuori dalle cartelle famiglia (es. Orchidea SOL/
# inharmonic: nomi di strumento/tecnica, non di famiglia fisica) - lista
# breve e conservativa di proposito: il default "strike" (banco modale
# libero) copre gia' bene i casi ambigui/percussivi/a frizione/inarmonici.
KEYWORD_TYPE_MAP = [
    (("pluck", "pizz"), "pluck"),
    (("shake", "marac", "rattle", "tambourine", "cabasa"), "shaker"),
    (("_noise", "breath", "aeolian"), "noise"),
]


def infer_exciter_type(rel_path):
    """Mai None: ogni file viene sempre etichettato con QUALCHE tipo, anche
    se non e' la piu' fisicamente accurata (es. una tecnica percussiva su
    strumento a fiato non spostata in una cartella famiglia finisce
    comunque su "strike" via KEYWORD_TYPE_MAP/default, mai scartata)."""
    top = rel_path.split(os.sep)[0].lower()
    if top in FOLDER_TYPE_OVERRIDE:
        return FOLDER_TYPE_OVERRIDE[top]
    if top in FAMILY_TYPES:
        return top
    low = rel_path.lower()
    for keywords, etype in KEYWORD_TYPE_MAP:
        if any(k in low for k in keywords):
            return etype
    return "strike"


def select_files(sample_root, per_folder, seed):
    """Ogni cartella FINALE (quella che contiene direttamente i .wav, a
    qualunque profondita' sotto sample/ - anche una cartella famiglia
    intera se i file sono li' diretti senza sottocartelle): max per_folder
    file casuali (seed fisso -> stessa selezione ad ogni run, utile con
    --resume). Nessuna eccezione per cartella: --per-folder si applica
    sempre, uniforme."""
    all_files = sorted(glob.glob(os.path.join(sample_root, "**", "*.wav"), recursive=True))
    rng = random.Random(seed)
    by_leaf = defaultdict(list)
    for path in all_files:
        rel = os.path.relpath(path, sample_root)
        by_leaf[os.path.dirname(rel)].append(rel)
    selected = []
    for leaf, files in by_leaf.items():
        files = sorted(files)
        rng.shuffle(files)
        selected.extend(files[:per_folder])   # slice oltre la lunghezza = nessun errore: cartelle con <per_folder file vengono usate per intero
    return sorted(selected)


def _load_done(jsonl_path):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["path"])
    return done


def _flush_json(jsonl_path, out_path):
    """Rigenera predictor_dataset.json (il formato {tipo: [...]} atteso da
    resonator_predictor.train_predictor) dal .jsonl accumulato finora -
    cosi' il dataset e' sempre pronto all'uso anche a corsa non finita.
    Un dict per ogni tipo TROVATO nel .jsonl (non solo i 4 supportati dal
    predittore - bow/blow inclusi, restano nel file anche se
    resonator_predictor.py oggi non li usa, vedi ambito in quel file)."""
    dataset = defaultdict(list)
    if not os.path.exists(jsonl_path):
        return dataset
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dataset[row["type"]].append({k: row[k] for k in ("path", "target", "f0", "params", "loss")})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    return dataset


def main(sample_root=SAMPLE_ROOT, restarts=1, iters=100, per_folder=4, seed=0, seconds=1.0,
         out_path="predictor_dataset.json", jsonl_path=JSONL_PATH, resume=False, flush_every=10):
    selected = select_files(sample_root, per_folder, seed)
    done = _load_done(jsonl_path) if resume else set()
    todo = [rel for rel in selected if rel not in done]
    print(f"file selezionati: {len(selected)}  (di cui gia' fatti: {len(done)}, da fare: {len(todo)})")

    jf = open(jsonl_path, "a", encoding="utf-8")
    n_done, n_skipped = len(done), 0
    for i, rel in enumerate(todo, 1):
        path = os.path.join(sample_root, rel)
        etype = infer_exciter_type(rel)
        try:
            target, f0 = analyze_reference(path, seconds=seconds, sr=SR_DEFAULT)
        except Exception as e:
            print(f"  SALTATO {rel}: analisi fallita ({e})")
            n_skipped += 1
            continue
        target = {k: v for k, v in target.items() if k != "extra"}   # extra non e' un target di training (vedi random_target)
        try:
            exciter, modal, achieved, loss = train_agent_best(
                etype, target, f0=f0, restarts=restarts, iters=iters, seconds=seconds,
                use_predictor=False, log=None)   # use_predictor=False: le etichette devono venire SOLO dal warm_start analitico,
                                                  # mai da un predittore parzialmente allenato in un run precedente (niente feedback loop)
        except Exception as e:
            print(f"  SALTATO {rel}: training fallito ({e})")
            n_skipped += 1
            continue
        params = flatten_params(exciter, modal).tolist()

        # Fix 2026-09-10: sanitizzazione SOLO qui, dopo train_agent_best - target["decay_slope"]
        # (synth_torch_before.analyze_reference, anchor Q/decay-slope) e' un tensor, serve
        # tale e quale dentro train_agent (achieved_slope - target["decay_slope"], entrambi
        # tensor) - convertirlo prima rompeva quella sottrazione (tensor - list) per OGNI
        # file strike/pluck/shaker. Qui invece serve solo json-serializzabile, e non e'
        # comunque riletto da resonator_predictor.py (non e' in TARGET_KEYS).
        target_json = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in target.items()}
        row = {"type": etype, "path": rel, "target": target_json, "f0": f0, "params": params, "loss": loss}
        jf.write(json.dumps(row) + "\n")
        jf.flush()
        n_done += 1
        print(f"[{etype:8s}] {rel}: loss={loss:.4f}  ({i}/{len(todo)}, totale {n_done})")

        if i % flush_every == 0:
            _flush_json(jsonl_path, out_path)

    jf.close()
    dataset = _flush_json(jsonl_path, out_path)
    print()
    for t in sorted(dataset):
        print(f"{t}: {len(dataset[t])} esempi")
    print(f"saltati questa corsa: {n_skipped}")
    print(f"Scritto: {out_path} (e {jsonl_path}, riprendibile con --resume)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--per-folder", type=int, default=4, help="max file casuali per cartella finale, sempre (nessuna eccezione per cartella)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="predictor_dataset.json")
    p.add_argument("--jsonl", default=JSONL_PATH)
    p.add_argument("--resume", action="store_true", help="salta i file gia' presenti nel .jsonl di una corsa precedente")
    p.add_argument("--flush-every", type=int, default=10)
    a = p.parse_args()
    main(restarts=a.restarts, iters=a.iters, per_folder=a.per_folder, seed=a.seed,
         out_path=a.out, jsonl_path=a.jsonl, resume=a.resume, flush_every=a.flush_every)
