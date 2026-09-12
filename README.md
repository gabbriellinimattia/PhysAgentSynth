# PhysAgentSynth

Sintetizzatore basato su agenti fisici (exciter + banco modale) fittati per-descrittore via gradient descent, con una rete (resonator_predictor) che distilla il fitting in un predittore diretto target->parametri.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

La GUI real-time usa `tkinter` (incluso di default in Python su macOS/Windows; su Linux: `apt install python3-tk`).

I file audio di riferimento (`sample/`, ~12GB) NON sono inclusi nel repo. Servono solo per `predictor_dataset_gen.py`, `physical_agents_stress_test.py` e `sample_calibration.py`.

## Struttura

- `physical_agents_train.py` — core: training offline degli agenti fisici (N exciter iper-specializzati + 1 banco modale condiviso) per fit di un vettore di descrittori target via discesa del gradiente. Eseguito direttamente (`python physical_agents_train.py`) lancia una demo su tutti i tipi di exciter con target casuali.
- `single_train.py` — allena UN solo tipo di exciter (`python single_train.py <tipo>`, `--list` per i tipi disponibili).
- `resonator_predictor.py` — rete che predice i parametri exciter+banco modale direttamente da un vettore target, distillando `physical_agents_train.py` (no fitting per-istanza a runtime).
- `train_all_predictors.py` — allena `resonator_predictor.py` per tutti i tipi supportati (tranne "chaotic", instabile) usando `predictor_dataset.json`. Un checkpoint `.pt` per tipo.
- `predictor_dataset_gen.py` — genera `predictor_dataset.json`/`.jsonl`: per ogni .wav in `sample/` allena l'agente fisico e salva (target, f0, parametri convergenti).
- `sample_calibration.py` — estrae dai campioni in `sample/` la distribuzione reale dei descrittori per famiglia di exciter, per calibrare i target random.
- `analyzer.py` — analisi/estrazione descrittori audio (AnalyzerV3).
- `modal_bank_rt.py` — banco modale ridisegnato per uso realtime (interpolazione tra campionamenti a intervalli, non FFT sull'intero buffer).
- `synth_gui_rt.py` — GUI realtime (ModalBankRT + resonator_predictor) con slider sui descrittori.
- `synth_torch_before.py` — motore di sintesi differenziabile (sines+noise+transient) per fit per-discesa-gradiente, alternativa a CMA-ES/resynth 1:1.
- `physical_agents_stress_test.py` — allena un agente su un file reale da `sample/` e confronta col riferimento su descrittori esclusi dalla loss di training (validazione fuori-target).
- `impulsive_batch_stress_test.py` — batch dello stress test sopra su più file.
- `probe_pluck_freeze.py`, `probe_pluck_steps.py` — diagnosi mirate su un bug di freeze nel fitting dell'exciter pluck.
- `blow_*.py` — analisi/verifica su collasso harmonicity/noisiness per l'exciter blow.

## Training

```
# 1) (opzionale, richiede sample/) genera il dataset di distillazione
python predictor_dataset_gen.py --iters 100 --per-folder 4

# 2) training/demo degli agenti fisici (tutti i tipi)
python physical_agents_train.py
# oppure un solo tipo:
python single_train.py pluck --list   # elenca i tipi
python single_train.py pluck

# 3) allena i predittori NN (richiede predictor_dataset.json, incluso nel repo)
python train_all_predictors.py --epochs 300
# genera resonator_predictor_<tipo>.pt (non versionati, da rigenerare)
```

## Test

```
python physical_agents_stress_test.py [file.wav]   # round-trip vs campione reale, richiede sample/
python impulsive_batch_stress_test.py              # stesso test in batch
python sample_calibration.py                        # calibrazione target su distribuzione reale
```

## GUI real-time

```
python synth_gui_rt.py
```

Richiede i checkpoint `resonator_predictor_<tipo>.pt` generati da `train_all_predictors.py`.

## Note

- Checkpoint `.pt`, log di training e cartella `sample/` non sono versionati: rigenerabili con gli script sopra.
- Vedi `Claude outputs/riassunto_bow_debug.md` per note di debug su un problema noto (exciter bow).
