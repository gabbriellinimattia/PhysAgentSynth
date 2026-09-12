"""synth_gui_rt.py - GUI per pilotare il motore real-time (ModalBankRT +
resonator_predictor.py) via slider sui descrittori, con ascolto immediato
del risultato di ogni modifica.

Prerequisiti:
  - resonator_predictor_<tipo>.pt allenati (train_all_predictors.py) per
    strike/pluck/shaker/noise/chaotic. Senza, i parametri dell'exciter
    restano ai default fissi e solo Q/rolloff del banco modale rispondono
    ai descrittori (harmonicity/rolloff/e_low/e_mid/e_high/flux/t_centroid
    non hanno effetto - vedi nota in ModalBank.warm_start: l'unico posto
    che li applica davvero e' il delta del predittore). bow/blow non hanno
    predittore per costruzione (vedi SUPPORTED_TYPES in
    resonator_predictor.py): solo warm start analitico, sempre.
  - pip install sounddevice   (bindings PortAudio per l'audio in uscita)

"noise"/bow/blow sono continui (sliders cambiano il timbro mentre
suonano); strike/pluck/shaker/chaotic sono a impulso: premi Trigger (o
barra spazio) per un colpo, il decadimento che senti dopo e' il RING
naturale del banco modale (memoria del filtro, non un inviluppo aggiunto)
- puoi anche muovere gli slider mentre sta ancora suonando e sentirne
l'effetto sul ring in corso. chaotic: la non-linearita' di coupling tra
modi non e' implementata nel motore realtime (solo nel fit offline via
FFT) - suona col preset freq/gain/Q imparato dal predittore, non col
comportamento dinamico caotico vero. bow/blow: harmonicity/noisiness
NON hanno effetto per design (Q fissata alta per il tono, non derivata
dal noisiness target - vedi warm_start); flux/t_centroid neanche, senza
tremolo (mod_rate/mod_depth, solo bow) ne' rampa d'attacco (solo blow) -
nessun inviluppo che varia nel tempo in questo preview, solo tono
continuo+rumore base. centroid/rolloff/spread/e_low/e_mid/e_high invece
rispondono (bank_feedback dedicato, fino a 24 armoniche - vedi
_rebuild_feedback).

Uso: python3 synth_gui_rt.py
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
import torch.nn.functional as F

import physical_agents_train as P
from resonator_predictor import load_predictor, apply_delta
from modal_bank_rt import ModalBankRT

try:
    import sounddevice as sd
except ImportError as e:
    raise SystemExit("serve sounddevice: pip install sounddevice") from e

SR = P.SR_DEFAULT
BLOCK = 256
N_MODES = 12
GLIDE_MS = 60          # finestra di interpolazione ad ogni cambio slider (control-rate GUI, non l'X-time di un agente esterno)
BURST_SECONDS = 1.0    # durata del render dell'exciter ad un Trigger (stessa convenzione del training)
TYPES = ["strike", "pluck", "shaker", "noise", "chaotic", "bow", "blow"]
FEEDBACK_TYPES = ("bow", "blow")   # tono armonico continuo + rumore che bypassa il banco (vedi _rebuild_feedback/audio_callback)

lock = threading.Lock()
state = {
    "type": "strike",
    "f0": 220.0,
    "target": {"harmonicity": 0.3, "noisiness": 0.7, "centroid": 2000.0,
               "rolloff": 4000.0, "e_low": 0.2, "e_mid": 0.5, "e_high": 0.3,
               "flux": 0.15, "spread": 800.0, "t_centroid": 0.05},
}
bank = ModalBankRT(SR, n_modes=N_MODES)
_limiter_env = 1e-6   # stato del limiter RT, vedi audio_callback
LOOP_SECONDS_TARGET = 2.0   # durata approssimativa del loop bow/blow, vedi _rebuild_feedback
_feedback_pos = 0     # cursore di lettura nel loop bow/blow, vedi audio_callback
burst_queue = queue.Queue()
current_exciter = {"obj": None, "type": None, "feedback": None}


def _rebuild_feedback(target, f0, exciter_type):
    """bow/blow: dopo due round falliti a reimplementare tono+banco a mano
    in streaming (biquad RT, sintesi additiva) - bug propri impossibili da
    stanare alla cieca senza poter ascoltare l'audio da qui, culminati in
    un'onda triangolare fissa - questa versione RIUSA DIRETTAMENTE P.render
    (physical_agents_train.py), la stessa identica pipeline offline gia'
    validata (bow/blow suonano correttamente li', loss 0.38/0.27 nel report
    di sessione). Costo: non e' piu' sample-accurate in tempo reale come gli
    altri exciter - ogni cambio slider ri-renderizza un loop e lo sostituisce
    (possibile piccolo click al cambio, non un artefatto continuo).

    Durata del loop = un numero intero di periodi di f0: la componente
    tonale (dominante) chiude senza click al giro del loop; la componente
    di rumore (minoritaria) puo' avere una piccola discontinuita' al bordo,
    non risolta qui."""
    exciter = P.EXCITERS[exciter_type](SR)
    n_harm = P._n_harm_feedback(f0, target)
    modal = P.ModalBank(SR, n_modes=n_harm, f0_init=f0, use_decay_envelope=False,
                        use_coupling=False, use_legacy_filter=False)
    tone_amp_exponent = float(exciter.params()["p"]) if exciter_type != "blow" else None
    modal.warm_start(target, f0=f0, is_feedback=True, exciter_type=exciter_type,
                      tone_amp_exponent=tone_amp_exponent)
    n_periods = max(1, round(f0 * LOOP_SECONDS_TARGET))
    seconds = n_periods / f0
    buf = P.render(exciter, modal, seconds, sr=SR, target=target)
    with lock:
        current_exciter["obj"] = exciter
        current_exciter["type"] = exciter_type
        current_exciter["feedback"] = {"buffer": buf}

def rebuild(target, f0, exciter_type):
    """Ricalcola parametri exciter+modal (warm_start analitico +
    correzione del predittore, se il checkpoint esiste) e li applica al
    motore RT. Chiamato ad ogni cambio slider: e' lavoro a control-rate
    (una piccola MLP + qualche formula), non deve essere lock-free."""
    if exciter_type in FEEDBACK_TYPES:
        _rebuild_feedback(target, f0, exciter_type)
        return
    exciter = P.EXCITERS[exciter_type](SR)
    # use_legacy_filter/use_coupling allineati a train_agent (physical_agents_
    # train.py): chaotic riattivato in questa GUI 2026-09-12, serve
    # use_coupling=True altrimenti la non-linearita' che lo caratterizza
    # (_nonlinear_coupling) semplicemente non esiste nel modal creato qui, e
    # il predittore chaotic (che si aspetta anche coupling_raw/
    # coupling_threshold_raw nel suo delta, vedi _new_modules in
    # resonator_predictor.py) applicherebbe un delta piu' corto di quanto il
    # modal si aspetti in silenzio. use_decay_envelope resta False per
    # tutti: a runtime il decadimento e' il ring naturale del biquad
    # (ModalBankRT), non serve l'inviluppo esplicito usato nel fit offline.
    modal = P.ModalBank(SR, n_modes=N_MODES, f0_init=f0, use_decay_envelope=False,
                        use_coupling=(exciter_type == "chaotic"),
                        use_legacy_filter=(exciter_type in ("noise", "chaotic")))
    modal.warm_start(target, f0=f0, is_feedback=False, exciter_type=exciter_type)
    model = load_predictor(exciter_type)
    if model is not None:
        with torch.no_grad():
            delta = model(target, f0, model._x_mean, model._x_std) * model._y_std + model._y_mean
        apply_delta(exciter, modal, delta)
    with torch.no_grad():
        freq = torch.exp(modal.freq_raw).numpy()
        gain = F.softplus(modal.gain_raw).numpy()
        q = torch.clamp(F.softplus(modal.q_raw) + 0.4, max=P.Q_MAX).numpy()
    glide_samples = int(GLIDE_MS * 1e-3 * SR)
    with lock:
        bank.set_target(freq, gain, q, glide_samples)
        current_exciter["obj"] = exciter
        current_exciter["type"] = exciter_type
        current_exciter["feedback"] = None


def on_change(*_):
    with lock:
        target = dict(state["target"])
        f0 = state["f0"]
        etype = state["type"]
    tot = target["e_low"] + target["e_mid"] + target["e_high"]
    if tot > 1e-6:   # normalizza a somma 1, come i target reali (random_target/Dirichlet)
        for k in ("e_low", "e_mid", "e_high"):
            target[k] /= tot
    rebuild(target, f0, etype)


def trigger():
    with lock:
        etype, exciter = current_exciter["type"], current_exciter["obj"]
    if exciter is None or etype not in ("strike", "pluck", "shaker", "chaotic"):
        return   # "noise" e' gia' continuo, nessun impulso da accodare
    with torch.no_grad():
        burst = exciter(int(BURST_SECONDS * SR)).numpy().astype(np.float64)
    for i in range(0, len(burst), BLOCK):
        burst_queue.put(burst[i:i + BLOCK])


def audio_callback(outdata, frames, time_info, status):
    global _feedback_pos
    with lock:
        etype, exciter = current_exciter["type"], current_exciter["obj"]
        fb = current_exciter["feedback"]
    if etype in FEEDBACK_TYPES and fb is not None:
        # loop del buffer pre-renderizzato (P.render, vedi _rebuild_feedback)
        # invece di sintesi a blocco - vedi il docstring li' per il perche'.
        buf = fb["buffer"]
        n = len(buf)
        idx = (np.arange(frames) + _feedback_pos) % n
        y = buf[idx].astype(np.float64)
        _feedback_pos = (_feedback_pos + frames) % n
    elif etype == "noise" and exciter is not None:
        with torch.no_grad():
            excitation = exciter(frames).numpy().astype(np.float64)
        y = bank.process_block(excitation)
    else:
        try:
            excitation = burst_queue.get_nowait()
            if len(excitation) < frames:
                excitation = np.pad(excitation, (0, frames - len(excitation)))
        except queue.Empty:
            # nessun impulso in coda: il banco continua a suonare SOLO per
            # via del proprio stato (ring naturale via il feedback y[n-1],
            # y[n-2] del biquad) - esattamente il comportamento fisico di
            # un risonatore che si spegne, non un inviluppo simulato.
            excitation = np.zeros(frames)
        y = bank.process_block(excitation)
    # Limiter con inviluppo persistente (attacco veloce, rilascio lento) al
    # posto del rescale istantaneo per-blocco: quest'ultimo (peak locale del
    # SOLO blocco corrente, ~256 campioni=5.8ms a 44.1kHz) cambia guadagno di
    # continuo blocco per blocco seguendo il ripple naturale di modi
    # risonanti che battono tra loro - fisiologico anche a Q moderate - e
    # quel guadagno che salta ogni 5.8ms senza alcuno smoothing e' udibile
    # come distorsione/pumping continuo su OGNI exciter (segnalato in
    # sessione), non un vero overs occasionale. Qui l'inviluppo di picco
    # sale quasi subito su un transiente (attacco ~1 blocco, previene comunque
    # gli overs) ma scende su ~30 blocchi (~170ms, rilascio lento): il
    # guadagno resta stabile sul ring naturale di un modo invece di
    # rincorrerne ogni micro-fluttuazione.
    global _limiter_env
    block_peak = float(np.abs(y).max())
    coef = 0.9 if block_peak > _limiter_env else 0.03
    _limiter_env += coef * (block_peak - _limiter_env)
    gain = min(1.0, 0.95 / max(_limiter_env, 1e-6))
    outdata[:, 0] = (y * gain).astype(np.float32)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
root = tk.Tk()
root.title("Synth RT - controllo a descrittori")

type_var = tk.StringVar(value="strike")


def on_type(*_):
    state["type"] = type_var.get()
    on_change()


ttk.Label(root, text="Exciter:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
ttk.OptionMenu(root, type_var, "strike", *TYPES, command=on_type).grid(row=0, column=1, sticky="w")
ttk.Button(root, text="Trigger (spazio)", command=trigger).grid(row=0, column=2, padx=10)
root.bind("<space>", lambda e: trigger())

SLIDERS = [
    ("f0", 60, 1200, 220),
    ("harmonicity", 0.0, 1.0, 0.3),
    ("noisiness", 0.0, 1.0, 0.7),
    ("centroid", 100, 10000, 2000),
    ("rolloff", 100, 15000, 4000),
    ("e_low", 0.0, 1.0, 0.2),
    ("e_mid", 0.0, 1.0, 0.5),
    ("e_high", 0.0, 1.0, 0.3),
    ("flux", 0.0, 0.5, 0.15),
    ("spread", 50, 5000, 800),
    ("t_centroid", 0.005, 0.4, 0.05),
]


def make_cb(name, var):
    def cb(_evt=None):
        if name == "f0":
            state["f0"] = var.get()
        else:
            state["target"][name] = var.get()
        on_change()
    return cb


for row, (name, lo, hi, default) in enumerate(SLIDERS, start=1):
    ttk.Label(root, text=name).grid(row=row, column=0, sticky="w", padx=4)
    var = tk.DoubleVar(value=default)
    cb = make_cb(name, var)
    ttk.Scale(root, from_=lo, to=hi, orient="horizontal", variable=var,
              command=lambda _v, cb=cb: cb()).grid(row=row, column=1, columnspan=2, sticky="ew", padx=4)

root.columnconfigure(1, weight=1)

stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=1,
                          dtype="float32", callback=audio_callback)
stream.start()
on_change()   # inizializza il motore con i valori di default degli slider

root.mainloop()
stream.stop()
