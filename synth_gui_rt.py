"""synth_gui_rt.py - GUI per pilotare il motore real-time (ModalBankRT +
resonator_predictor.py) via slider sui descrittori, con ascolto immediato
del risultato di ogni modifica.

Prerequisiti:
  - resonator_predictor_<tipo>.pt allenati (train_all_predictors.py) per
    strike/pluck/shaker/noise. Senza, i parametri dell'exciter restano ai
    default fissi e solo Q/rolloff del banco modale rispondono ai
    descrittori (harmonicity/rolloff/e_low/e_mid/e_high/flux/t_centroid
    non hanno effetto - vedi nota in ModalBank.warm_start: l'unico posto
    che li applica davvero e' il delta del predittore).
  - pip install sounddevice   (bindings PortAudio per l'audio in uscita)

"noise" e' continuo (sliders cambiano il timbro mentre suona); strike/
pluck/shaker sono a impulso: premi Trigger (o barra spazio) per un colpo,
il decadimento che senti dopo e' il RING naturale del banco modale
(memoria del filtro, non un inviluppo aggiunto) - puoi anche muovere gli
slider mentre sta ancora suonando e sentirne l'effetto sul ring in corso.

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
TYPES = ["strike", "pluck", "shaker", "noise"]

lock = threading.Lock()
state = {
    "type": "strike",
    "f0": 220.0,
    "target": {"harmonicity": 0.3, "noisiness": 0.7, "centroid": 2000.0,
               "rolloff": 4000.0, "e_low": 0.2, "e_mid": 0.5, "e_high": 0.3,
               "flux": 0.15, "spread": 800.0, "t_centroid": 0.05},
}
bank = ModalBankRT(SR, n_modes=N_MODES)
burst_queue = queue.Queue()
current_exciter = {"obj": None, "type": None}


def rebuild(target, f0, exciter_type):
    """Ricalcola parametri exciter+modal (warm_start analitico +
    correzione del predittore, se il checkpoint esiste) e li applica al
    motore RT. Chiamato ad ogni cambio slider: e' lavoro a control-rate
    (una piccola MLP + qualche formula), non deve essere lock-free."""
    exciter = P.EXCITERS[exciter_type](SR)
    # use_legacy_filter allineato a train_agent: solo "noise" (niente chaotic
    # in questa GUI, vedi TYPES) resta sul vecchio filtro H(0)=1.
    modal = P.ModalBank(SR, n_modes=N_MODES, f0_init=f0, use_decay_envelope=False, use_coupling=False,
                        use_legacy_filter=(exciter_type == "noise"))
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
    if exciter is None or etype not in ("strike", "pluck", "shaker"):
        return   # "noise" e' gia' continuo, nessun impulso da accodare
    with torch.no_grad():
        burst = exciter(int(BURST_SECONDS * SR)).numpy().astype(np.float64)
    for i in range(0, len(burst), BLOCK):
        burst_queue.put(burst[i:i + BLOCK])


def audio_callback(outdata, frames, time_info, status):
    with lock:
        etype, exciter = current_exciter["type"], current_exciter["obj"]
    if etype == "noise" and exciter is not None:
        with torch.no_grad():
            excitation = exciter(frames).numpy().astype(np.float64)
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
    peak = np.abs(y).max()
    if peak > 0.95:
        y = y / peak * 0.95
    outdata[:, 0] = y.astype(np.float32)


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
