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
# bank dedicato a bow/blow: fino a P.N_HARM_FEEDBACK_MAX armoniche (24, vedi
# physical_agents_train.py), non i 12 modi di "bank" - a differenza degli
# altri exciter (dove il banco sceglie liberamente DOVE mettere i modi,
# quindi 12 bastano ovunque nello spettro), per bow/blow le frequenze dei
# modi sono FISSATE alle armoniche esatte di f0: il numero di armoniche
# determina direttamente quanto in alto puo' arrivare il tono, capare a 12
# rendeva centroid/rolloff/e_high inefficaci per f0 bassi (non c'era alcuna
# armonica sopra i 3000Hz da poter enfatizzare).
bank_feedback = ModalBankRT(SR, n_modes=P.N_HARM_FEEDBACK_MAX)
_limiter_env = 1e-6   # stato del limiter RT, vedi audio_callback
_tone_phase = 0.0     # fase persistente del fondamentale bow/blow tra blocchi, vedi audio_callback
burst_queue = queue.Queue()
current_exciter = {"obj": None, "type": None, "feedback": None}


def _rebuild_feedback(target, f0, exciter_type):
    """bow/blow: architettura diversa dagli altri exciter (vedi
    _feedback_synthesize in physical_agents_train.py) - tono armonico a
    n_harm righe esatte (6-24, scala con f0/target, vedi _n_harm_feedback)
    che il banco modale shape-a via warm_start(is_feedback=True), PIU'
    rumore colorato concentrato attorno a f0 che bypassa il banco e si
    somma dopo (altrimenti il filtro ad alto Q lo "ripulisce" in energia
    tonale, vedi nota in BowExciter). Nessun predittore esiste per questi
    due tipi (SUPPORTED_TYPES in resonator_predictor.py li esclude
    esplicitamente), quindi qui non c'e' correzione di delta, solo warm
    start analitico - stesso limite del training offline.

    Usa bank_feedback (fino a 24 modi, non i 12 di "bank"): qui il numero
    di armoniche determina la frequenza massima raggiungibile dal tono
    (fisse alle armoniche esatte di f0), a differenza degli altri exciter
    dove il banco sceglie liberamente dove mettere i modi."""
    exciter = P.EXCITERS[exciter_type](SR)
    n_harm = P._n_harm_feedback(f0, target)
    modal = P.ModalBank(SR, n_modes=n_harm, f0_init=f0, use_decay_envelope=False,
                        use_coupling=False, use_legacy_filter=False)
    params = exciter.params()
    # BUG FIX 2026-09-12: mancava, causava "onda triangolare" fissa immune a
    # centroid/rolloff. Per bow il tono in ingresso ha gia' un'attenuazione
    # 1/k^p (vedi sotto); senza dirlo a warm_start (tone_amp_exponent), il
    # gain-fix in _band_gain_profile calcola il profilo di banda come se
    # l'ingresso fosse piatto - il risultato finito e' dominato dal SOLO
    # 1/k^p del tono (che non dipende da nessun descrittore) invece che dal
    # target, mascherando l'effetto di centroid/rolloff/spread. blow resta
    # None: non usa 1/k^p (vedi _tone_amp_profile sotto), nulla da compensare
    # - stessa condizione esatta di train_agent (physical_agents_train.py).
    tone_amp_exponent = float(params["p"]) if exciter_type != "blow" else None
    modal.warm_start(target, f0=f0, is_feedback=True, exciter_type=exciter_type,
                      tone_amp_exponent=tone_amp_exponent)
    with torch.no_grad():
        freq = torch.exp(modal.freq_raw).numpy()
        gain = F.softplus(modal.gain_raw).numpy()
        q = torch.clamp(F.softplus(modal.q_raw) + 0.4, max=P.Q_MAX).numpy()
    pad = P.N_HARM_FEEDBACK_MAX - n_harm
    if pad > 0:   # modi oltre n_harm: gain 0 (silenti), freq/q un valore qualunque innocuo
        freq = np.concatenate([freq, np.full(pad, freq[-1])])
        gain = np.concatenate([gain, np.zeros(pad)])
        q = np.concatenate([q, np.full(pad, 1.0)])
    k = np.arange(1, n_harm + 1)
    if exciter_type == "blow":
        amp = P._tone_amp_profile(f0, n_harm, target).numpy()
    else:
        amp = 1.0 / (k ** float(params["p"]))
        amp = amp / amp.sum()
    feedback_state = {
        "f0": float(f0), "n_harm": n_harm, "amp": amp,
        "pressure": float(params["pressure"]), "roughness": float(params["roughness"]),
        "noise_slope_raw": params["noise_slope"], "focus_bw": float(params["focus_bw"]),
    }
    glide_samples = int(GLIDE_MS * 1e-3 * SR)
    with lock:
        bank_feedback.set_target(freq, gain, q, glide_samples)
        current_exciter["obj"] = exciter
        current_exciter["type"] = exciter_type
        current_exciter["feedback"] = feedback_state


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
    global _tone_phase
    with lock:
        etype, exciter = current_exciter["type"], current_exciter["obj"]
        fb = current_exciter["feedback"]
    if etype in FEEDBACK_TYPES and fb is not None:
        # tono a fase persistente (continuita' tra blocchi: ricalcolarla da
        # zero ogni blocco produrrebbe un salto di fase, udibile come click,
        # ad ogni chiamata del callback) passato nel banco per lo shaping
        # formantico, rumore sommato DOPO bypassando il banco - stessa
        # architettura additiva di _feedback_synthesize, vedi _rebuild_feedback.
        t = np.arange(frames)
        w = 2.0 * np.pi * fb["f0"] / SR
        phases = _tone_phase + w * t
        tone = np.zeros(frames)
        for k_idx in range(1, fb["n_harm"] + 1):
            tone += fb["amp"][k_idx - 1] * np.sin(k_idx * phases)
        _tone_phase = float((_tone_phase + w * frames) % (2.0 * np.pi))
        tone *= fb["pressure"]
        with torch.no_grad():
            noise = fb["roughness"] * P._colored_noise(
                frames, SR, fb["noise_slope_raw"], center_hz=fb["f0"], bw=fb["focus_bw"]).numpy()
        y = bank_feedback.process_block(tone) + fb["pressure"] * noise
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
