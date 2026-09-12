"""
physical_agents_train.py - training offline degli agenti di sintesi per
modelli fisici: N moduli exciter iper-specializzati (2-4 parametri fisici
ciascuno, selezione manuale nel synth) + 1 agente condiviso di banco
modale (gestione dei filtri risonanti). Nessun file audio in ingresso:
il target e' un vettore di descrittori, l'ottimizzazione lavora su buffer
in memoria (vedi descriptor_vector_torch), il file .wav viene scritto solo
a convergenza (save_wav, riusato da synth_torch_before.py).

Ogni exciter ha pochi parametri -> poche iterazioni bastano; il banco
modale viene inizializzato analiticamente dal target (warm start), non da
valori casuali, cosi' anche i suoi ~3*n_modes parametri convergono in
poche decine di step.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synth_torch_before import (
    SR_DEFAULT, N_FFT_DEFAULT, BANDS, EPS, RATIO_KEYS,
    descriptor_vector_torch, _descriptor_loss_terms, save_wav,
    decay_slope_db_per_band, NOISINESS_DB_LOW, NOISINESS_DB_HIGH,
)

LIMITER_DRIVE = 0.9
DECAY_SLOPE_TYPES = ("strike", "pluck", "shaker")   # exciter impulsivi: unici con un
# decadimento la cui slope (Schroeder EDC per banda, vedi synth_torch_before.py)
# ha senso come target - bow/blow sono a regime, non decadono.
DECAY_SLOPE_WEIGHT = 0.05   # PROVVISORIO - non ancora validato su probe sentinel
DECAY_SLOPE_NORM = 100.0   # normalizzazione (dB/s) prima del quadrato - ordine di
# grandezza tipico di uno scarto slope osservabile, non tarato con precisione
Q_MAX = 600.0   # Q massimo (mode piu' armonico/stretto) raggiungibile in warm start
DECAY_ENV_FLOOR = 2e-3   # floor sull'inviluppo di decadimento (strike/pluck/shaker,
# vedi ModalBank.forward) - CAUSA STRUTTURALE del gap di harmonicity/flux per
# 5 round di tuning su cap/pesi/LR (vedi storia sopra e in _pluck_comb_profile).
# FIX 2026-09-07 (punto 1 review): harmonicity/noisiness in
# descriptor_vector_torch erano una spectral flatness mediata SEMPLICEMENTE sui
# frame STFT, non pesata per energia - non piu': ora pesata per frame_mag_sum,
# come aggregate_energy_weighted/analyzer.py (aggiornato in coppia, stessa
# logica). Il floor sotto resta comunque utile di suo (fix di sintesi, non di
# metrica - vedi ultima riga del blocco), ma non e' piu' l'unica difesa contro
# il bug di aggregazione. Un pizzico reale continua a suonare, per
# quanto piano, con la STESSA struttura armonica (bassa flatness) ben oltre il
# decadimento percepito - t_centroid e' energy-weighted, quindi corto anche se
# la coda tonale dura a lungo a livello trascurabile. La nostra sintesi invece
# applica exp(-t/tau) SENZA floor: con tau tirato corto dal target t_centroid,
# a t=1s il floatta underfloa a ZERO ESATTO (non solo piccolo) molto prima
# della fine del buffer - i frame STFT del resto del buffer diventano EPS
# uniforme su ogni bin, la cui flatness e' 1.0 ESATTO (massima possibile per
# costruzione, non una stima) e domina la media semplice sui ~40 frame,
# schiacciando harmonicity indipendentemente da quanto sia pulito il ring nei
# 2-3 frame realmente attivi (osservato: harmonicity 0.30-0.42 anche con Q
# 300-600, mai spiegato da drift/Q/gain in nessun tentativo precedente).
# Floor a 2e-3 (~-54dB, energeticamente 4e-6 in potenza - trascurabile per
# centroid/rolloff/spread/t_centroid, tutti energy-weighted): oltre il punto
# in cui exp(-t/tau) scenderebbe sotto il floor, il segnale smette di essere
# ulteriormente schiacciato dall'inviluppo extra e torna a seguire il proprio
# decadimento naturale di Q (gia' presente in y_modes dalla risposta del
# filtro, scalato dal floor) - resta un residuo genuinamente tonale (bassa
# flatness) invece di collassare a EPS. Non tocca la formula di
# descriptor_vector_torch/aggregate_energy_weighted (nessuna ricalibrazione
# dei target necessaria): e' un fix di sintesi, non di metrica - rende la coda
# fisicamente piu' realistica (nessun vero risuonatore ammutolisce a zero
# esatto), non maschera il segnale ne' lo forza verso il target.
NYQUIST_MARGIN = 0.47   # tetto su f0s in ModalBank.forward, come frazione di sr: freq_raw
# non ha mai avuto un limite superiore, e in training puo' derivare sopra Nyquist
# (osservato: modi a 26129Hz/28289Hz/31556Hz con sr=44100, Nyquist=22050) - li'
# ratio=freqs/f0s resta sempre <1 su tutto lo spettro rappresentabile, quindi il
# modo non risuona mai a un vero picco stretto ma agisce da shelf ad ampio spettro
# che spinge energia verso l'estremo alto della banda udibile: spiega il collasso
# persistente di e_high su pluck, sopravvissuto a 5 tentativi di vincolare
# gain_raw/freq_raw via termini di loss (il problema non era li').
LOSS_WEIGHTS = {"harmonicity": 2.0, "noisiness": 2.0, "e_low": 2.0, "e_mid": 2.0, "e_high": 2.0,
                "harmonicity_low": 1.5, "harmonicity_mid": 1.5, "harmonicity_high": 1.5,
                "noisiness_low": 1.5, "noisiness_mid": 1.5, "noisiness_high": 1.5}
# harmonicity/noisiness_{low,mid,high} (2026-09-09): stessa flatness/sigmoid della
# versione globale ma per banda (synth_torch_before.py/descriptor_vector_torch) -
# unico canale rimasto sensibile a Q dopo che tau_raw ha scollegato il decadimento
# da Q (physical_agents_train.py, punto 8), ma diluito nel LogSumExp quando e' un
# solo scalare globale. Peso 1.5 (< 2.0 dei globali, che restano): PROVVISORIO, non
# ancora validato su probe sentinel.
# flux a peso 2.0 provato e RITIRATO: run seed 0/1 mostrano flux migliorato
# (0.004->0.032, 0.029->0.06) ma loss aggregata peggiore in entrambi i run
# (0.600->0.704/1.318, 0.462->0.508/0.629) - competeva con harmonicity/e_bands
# nello stesso minimax, costando piu' di quanto rendeva. La causa vera del
# gap di flux/harmonicity e' probabilmente a monte (vedi ModalBank.forward,
# DECAY_ENV_FLOOR) - risolvere li' invece di spostare pesi tra termini che
# competono per lo stesso budget di gradiente.
# t_centroid a 2.0 provato e RITIRATO: per shaker (dove t_centroid e' piu' lontano dal target,
# 0.109 vs 0.226) alzare il suo peso non risolve il conflitto con le bande - lo sposta, peggiorando
# la loss aggregata (1.1347->1.8489, e_low 0.122->0.259 lontano dal target 0.0). Decadimento e
# forma spettrale condividono le stesse leve (Q, ora anche tau_raw) per shaker: il trade-off e'
# strutturale, non risolvibile spostando peso tra i termini di loss.
FLUX_NORM_FLOOR = 0.05   # pavimento per l'errore relativo di flux (vedi _weighted_loss_terms).
# flux e' l'unico descrittore rimasto nel ramo 'assoluto' di _weighted_loss_terms con un range
# di valori piccolo (target 0.0-0.4, vedi random_target/_sample_calibrated): a scala assoluta
# (achieved-target)**2 resta ~1-2 ordini di grandezza sotto harmonicity/noisiness (pesati 2.0 su
# range 0-1) o ai termini log2-ratio di centroid/rolloff/spread - nella LogSumExp (MINIMAX_BETA=10)
# non riceve mai pressione di gradiente reale (osservato su ogni exciter type: flux sistematicamente
# sotto target, fino a -86% su blow, anche quando tutti gli altri descrittori convergono bene).
# NON portato in RATIO_KEYS (stesso trattamento log2 di centroid/rolloff/spread/t_centroid): li'
# achieved->0 (regime comune per flux a inizio training, e strutturale per pluck - vedi
# PluckExciter, decadimento piu' corto di un frame STFT) manderebbe log2(achieved/target) a
# -infinito, saturando tanh(raw/TERM_CLAMP) e azzerando il gradiente esattamente nel caso che
# piu' ha bisogno di spinta - la stessa patologia (region piatta vicino a zero) che TERM_CLAMP fu
# introdotto per evitare su altri descrittori. L'errore relativo con floor,
# ((achieved-target)/(target+floor))**2, resta finito ovunque (achieved=0 -> circa 1 invece di
# infinito) mantenendo comunque l'auto-scala col target che RATIO_KEYS da' agli altri.
GAIN_PROFILE_WEIGHT = 6.0   # peso del vincolo diretto su gain_raw (vedi _gain_profile_loss).
# Ripristinato da 2.0: la causa della regressione blow/shaker non era il peso ma la formula
# reach applicata al regime sbagliato (vedi _band_gain_profile, use_reach) - corretta quella,
# un'ancora debole toglie anche a pluck (che la richiede, storicamente fragile) la spinta che gli
# serve: a peso 2.0 pluck e' regredito (e_mid/e_high 0.84/0.16 contro target 0.535/0.465, la stessa
# forma di collasso vista prima del fix originale).
FREQ_DRIFT_WEIGHT = 3.0     # peso del vincolo su freq_raw rispetto al warm start (vedi _freq_drift_loss)
Q_ANCHOR_WEIGHT = 3.0      # peso del vincolo su q_raw verso il target analitico (vedi _q_anchor_loss).
# Fix 2026-09-09, PROVVISORIO (non ancora tarato su probe estesa oltre i 3 sentinel) - isolamento
# gradiente Q: q_raw era l'unico parametro allenabile senza alcun'ancora indipendente dal ramo
# achieved/_weighted_loss_terms/_aggregate_loss (che puo' saturare per errori raw grandi a warm
# start - vedi storia TERM_CLAMP/tanh sopra), a differenza di gain_raw (gain_profile_loss) e
# freq_raw (freq_drift_loss).
PLUCK_FREQ_LR_MULT = 0.35   # RITIRATO (vedi train_agent, freq_lr_mult) - non piu' applicato.
# Ipotesi originaria: freq_raw a LR pieno viene tirato da centroid/rolloff/spread lontano
# dalla serie armonica di warm start, e questo drift (17-31% osservato) spiegherebbe il gap
# di harmonicity (0.30-0.42 contro target fino a 0.748). Testato (seed 0/1): drift dimezzato
# (17%->11%) ma harmonicity INVARIATA - il drift non era la causa. Nel frattempo togliere
# mobilita' a freq_raw ha scaricato piu' lavoro sul gain (stesso parametro del leak DC, vedi
# _pluck_comb_profile), riaprendo il leak e peggiorando e_mid/e_high. Costante lasciata per
# riferimento, non collegata a nulla.
MIN_ACTIVE_MODES = 8.0      # soglia minima di 'modi effettivi' per noise (vedi _active_modes_loss)
ACTIVE_MODES_WEIGHT = 4.0   # peso del vincolo anti-collasso per noise
# 2026-09-08/09: TENTATO _active_modes_loss anche su strike (stessa firma di collasso di
# noise, harmonicity achieved SNAP a 0.000 su 29/40 file reali, restarts=1 - vedi
# impulsive_batch_stress_test.py). FALSIFICATO da misura diretta (eff_n instrumentato in
# log): eff_n resta 10-11.8 su 12 modi per l'INTERA run, su file collassati E su file buoni
# (Va-legno_batt-C5, Hn-slap-C#5, tt-tt_edge) - il gain non si concentra mai su pochi modi,
# quindi il vincolo (soglia 3.0) non si attiva mai (active_modes=0.0000 sempre). Rimosso.
# Nuovo indizio dalla stessa misura: su Va-legno_batt-C5 (il file peggiore) q_min/q_max e il
# termine di loss harmonicity restano IDENTICI bit per bit da it0 a it299 - il gradiente
# sembra bloccato su Q per questo file, non un problema di concentrazione di gain. Prossimo
# passo: instrumentare q_raw.grad (vedi sotto, "diagnostica gradiente Q").
GAIN_PROFILE_REFRESH_EVERY = 20   # every N iters (post-warmup) il gain-fix per bow/blow
# ricalcola il profilo con p corrente invece di farlo una sola volta a fine warmup:
# se p continua a muoversi nella seconda meta' del training il profilo one-shot torna
# a scostarsi. Ricalcolo economico (solo tensor ops, no backward).
# INPUT_AMP_COMP_EXP (0.6/0.8/1.0, compensazione 1/input_amp**p su bow/blow):
# storia di tuning 1.0->0.6->0.8 mai chiusa (0.8 restava in sottoshoot,
# e_high 0.06) - RIMOSSA, sostituita dal test Q-only sopra
# (Q_ONLY_COMP_EXP) in _band_gain_profile: se il fattore mancante e' Q, non
# serve piu' ritarare input_amp; se non lo e', si riparte da qui.
Q_ONLY_COMP_EXP = 1.0   # esponente su 1/q**p per bow/blow, in SOSTITUZIONE (non in
# aggiunta) della compensazione input_amp in _band_gain_profile - test isolato
# richiesto dopo che input_amp da solo (fino a **0.8) e input_amp+Q impilati
# avevano dato esiti opposti e incoerenti col modello 'gain*Q*input_amp'
# (vedi _band_gain_profile). Isola l'effetto di Q puro: se non basta a
# chiudere il sottoshoot di e_high, il fattore mancante non e' Q. Valore
# di partenza, da ritarare sul prossimo run (range Q qui ~1.7-2x, molto piu'
# stretto del range di input_amp: puo' richiedere un esponente ben piu' alto
# di 1.0 per un effetto paragonabile).
HARMONICITY_WEIGHT_BOW_MULT = 3.0   # FIX 2026-09-11: harmonicity/noisiness (bow) restano
# indietro nel minimax pesato rispetto a e_low/e_mid/e_high (che tirano roughness
# verso l'alto per riempire le bande di energia, vedi diagnosi in BowExciter.__init__)
# - stesso pattern gia' in uso per GAIN_PROFILE_WEIGHT_PLUCK_MULT/_STRIKE_MULT sotto,
# qui su harmonicity/noisiness invece che su gain_profile, solo per bow.
GAIN_PROFILE_WEIGHT_PLUCK_MULT = 1.5   # moltiplicatore extra su GAIN_PROFILE_WEIGHT
# solo per pluck (vedi train_agent/_gain_profile_loss): a peso comune 6.0
# pluck resta il piu' fragile, con residuo overshoot e_high (~0.55-0.65 contro
# target 0.465) anche dopo che 6.0 (ripristinato da 2.0) aveva gia' risolto il
# collasso peggiore (0.84/0.16). Un'ancora piu' forte SOLO per pluck evita di
# alzare il peso globale (rischio gia' documentato: a 2.0 regredisce, valori
# piu' alti per tutti non testati sugli altri exciter type).
GAIN_PROFILE_WEIGHT_STRIKE_MULT = 2.0   # ripristinato a 2.0 (2026-09-07, stesso giorno):
# abbassato brevemente a 1.0 con la motivazione che gp_loss~0 (anchor soddisfatta) mentre
# e_mid/e_high reali restavano lontani dal target - ma quell'analisi era PRIMA del fix
# della flatness energy-weighted (punto 1 review, stesso giorno). A 1.0, su 2 seed diversi
# (--seed 1, --seed 2) e' riemerso ESATTAMENTE il collasso bimodale gia' diagnosticato qui
# sotto (pochi modi altissimi isolati, es. un modo a 15013Hz o due modi a gain 1.8+ contro
# ~0.8 dei vicini) che il moltiplicatore 2.0 era stato introdotto per prevenire - centroid/
# spread ne escono trascinati fuori target. Il fix della flatness resta comunque valido ed
# e' indipendente da questo peso (harmonicity/noisiness sono migliorate su entrambi i run
# a 1.0); il problema qui non era la flatness, era aver rimosso il freno anti-collasso.
# dopo il fix del warm start (_mode_ratios, blend inharmonicity) il gain-fit strike restava
# libero di collassare su un profilo BIMODALE (pochi modi altissimi, g=0.5-0.8, su un fondo
# quasi muto g=0.06-0.15) per chiudere l'overshoot di e_high - numericamente centra la banda
# ma non e' il decadimento graduale gain-vs-frequenza di una membrana reale (quello gia'
# imposto dal warm start fisico). Un'ancora piu' forte SOLO per strike (valore di partenza,
# da ritarare sul prossimo run) mantiene gain_raw piu' vicino a quel profilo graduale anche
# quando il target spinge per piu' energia in alto - si accetta un piccolo residuo numerico
# su e_high in cambio di un timbro ancora credibile come corpo colpito.
MODE_REPULSION_WEIGHT = 2.0   # peso del vincolo anti-collisione tra modi, strike e pluck.
# (vedi _mode_repulsion_loss). Esteso a pluck: stessa soglia, stesso peso - osservata una coppia
# di modi alti (7714Hz/7904Hz, 0.035 ottave) gia' sotto MODE_MIN_SPACING_OCT, stesso rischio di
# collisione visto su strike. Introdotto dopo aver osservato, SOLO su strike e SOLO dopo il
# fix del peso di flux (physical_agents_train.py/_weighted_loss_terms), due modi convergere
# quasi esattamente sulla stessa frequenza (3884Hz/3885Hz, 0.0004 ottave di distanza): un
# artefatto del gradiente, non una soluzione fisica (una membrana reale non ha risonanze
# coincidenti - stesso problema percettivo del comb-filtering) che oltretutto spreca un grado
# di liberta' utile al gain-fit per coprire le bande del target.
MODE_MIN_SPACING_OCT = 0.05   # spaziatura minima tra coppie di modi in ottave (log2): sotto
# questa soglia due modi sono indistinguibili. La spaziatura piu' stretta osservata tra modi
# genuinamente target-driven (non collisi) e' ~0.15 ottave, quindi 0.05 penalizza solo la
# collisione senza vincolare le configurazioni normali. Fisso, non Q-dipendente: resta un
# vincolo semplice e stabile invece di legarlo alla larghezza di banda (Q) di ciascun modo.
MODE_MIN_SPACING = MODE_MIN_SPACING_OCT * float(np.log(2.0))   # soglia in unita' log-naturali
# di freq_raw (freq_raw e' log naturale, vedi ModalBank.forward/warm_start - non log2)
REBAND_MIN_MODES = 3   # modi minimi spostati da _reband_ratios (strike/pluck) dentro
# una banda a target non-zero altrimenti scoperta - fisso, non proporzionale a
# e_band*n_modes (vedi _reband_ratios: la versione proporzionale doppiava la
# correzione gia' fatta dal gain-fix, causando overshoot).
MIN_HIGH_HARM_FEEDBACK = 3   # armoniche minime garantite sopra 3000Hz per bow/blow
# quando il target chiede energia alta li' (vedi _n_harm_feedback): il reach
# scalato sopra puo' comunque non bastare in un caso limite dove rolloff/
# centroid del target coincidono quasi esattamente col vecchio tetto fisso
# (osservato: f0=713Hz, rolloff=4280≈6*f0, reach scalato dava comunque solo
# 6 armoniche/2 sopra 3000Hz per un target e_high=0.551).
GAIN_FREQ_WARMUP_FRAC = 1.0 / 6.0   # frazione iniziale di iters con gain_raw/freq_raw
# congelati (vedi train_agent): tre tentativi diversi di vincolare gain_raw/freq_raw
# con termini di loss aggiuntivi (gain_profile, freq_drift, poi il tetto morbido sui
# descrittori) non hanno spostato il collasso di pluck (e_high restato 0.96-1.0 in
# tutti) - segno che il gradiente enorme dei primissimi step (es. spread=13.97 a it0
# su pluck) strappa gain/freq dal warm start PRIMA che Q/tau/exciter abbiano ridotto
# la parte facile della loss, e nessun regolarizzatore aggiuntivo riesce a competere
# in tempo. Congelare gain/freq per una finestra iniziale (grad=None, salta l'update
# Adam per quei soli parametri) lascia agli altri il primo assestamento prima di
# lasciarli muovere.
TERM_CLAMP = 6.0   # tetto (morbido) per singolo termine in _weighted_loss_terms:
# RITIRATI 2026-09-09 (falsificati, non applicati): (1) clamp razionale al posto
# di tanh - rompe il clustering-verso-tetto che LogSumExp/MINIMAX_BETA=10
# assume (vedi sotto), misurato anche su tt-tt_edge (bloccato). (2) alzare
# TERM_CLAMP a 24 tenendo beta=10 - stesso effetto, beta non piu' calibrata
# sulla nuova scala dei termini. Isolato invece (senza fix ancora applicato):
# q_raw non ha alcuna anchor loss indipendente (a differenza di gain_raw/
# freq_raw, vedi gain_profile/freq_drift) - dipende SOLO da questo percorso,
# e riceve gradiente reale ma minuscolo (~1e-9/1e-11, misurato via
# .grad.norm() strumentato) su file dove nessun'altra loss riesce a smuovere
# raw sotto soglia in tempo utile - non necessariamente un bug numerico,
# puo' essere semplicemente una catena di Jacobiani lunga e poco sensibile.
# un solo descrittore malscalato a inizio training (osservato: spread=13.97 su
# pluck it0, tutto il resto <2.3) fa collassare la LSE quasi interamente su di se'
# (MINIMAX_BETA=10 -> gradiente ~zero al resto). torch.clamp() duro provato e
# RITIRATO: oltre la soglia e' piatto -> gradiente ESATTAMENTE zero, con piu'
# termini saturati insieme il training si blocca del tutto (misurato: pluck
# fermo a loss=6.0000 identico per 300 iterazioni). tanh() satura in valore ma
# resta ovunque differenziabile (derivata mai nulla), stesso tetto senza il
# plateau.
# harmonicity/noisiness: i due termini piu' lenti a convergere.
# t_centroid: ex-voce qui con peso 200 (errore assoluto, tarato sul range
# stretto strike/pluck 0.015-0.065s) - RIMOSSA: t_centroid e' ora in
# RATIO_KEYS (synth_torch_before.py, errore log2-ratio, si auto-scala col
# target come centroid/rolloff/spread), il peso fisso non serve piu' e
# rompeva shaker (range piu' ampio 0.03-0.25s, vedi random_target) facendo
# esplodere il termine assoluto*200 fino a dominare l'intera loss.
MINIMAX_BETA = 10.0  # temperatura del softmax nella LogSumExp: vedi _aggregate_loss

FEEDBACK_TYPES = {"bow", "blow"}    # exciter a tono periodico (stick-slip approssimato), non rumore filtrato

# N_HARM_FEEDBACK fisso=8 tagliava lo spettro raggiungibile a 8*f0: su f0 bassi
# (es. 240Hz) il tetto cadeva a ~1.9kHz, ben sotto il range reale di un arco
# (moto di Helmholtz: decadimento armonico lento ~1/k, energia densa fino a
# diversi kHz anche su note gravi - vedi euphonics.org/BowedStringReview.pdf).
# Scalare con f0 invece di un conteggio fisso copre sempre la stessa banda
# assoluta, indipendentemente dalla nota.
N_HARM_FEEDBACK_MIN = 6
N_HARM_FEEDBACK_MAX = 24
FEEDBACK_REACH_HZ = 4000.0


def _n_harm_feedback(f0, target=None):
    """Numero di armoniche per bow/blow: scala con f0 per coprire sempre
    ~FEEDBACK_REACH_HZ invece di affamare le note gravi (troppo poche
    armoniche) o sprecare calcolo su quelle acute (troppe, oltre Nyquist).

    target: se presente, la reach si alza fino al rolloff/centroid richiesto
    (mai abbassa sotto FEEDBACK_REACH_HZ) - un reach fisso su f0 alti (es.
    713Hz) lasciava solo 1-2 armoniche sopra i 3000Hz, troppo poche per un
    target e_high alto (~55%: gain fix costretto su concentrazioni estreme
    su 1-2 modi). target=None nella generazione del target stesso (chicken-
    egg: qui il target non esiste ancora, vedi random_target)."""
    reach_hz = FEEDBACK_REACH_HZ
    if target is not None:
        reach_hz = max(reach_hz, float(target.get("rolloff", reach_hz)),
                        float(target.get("centroid", reach_hz)))
    n = int(np.clip(np.ceil(reach_hz / max(float(f0), 1.0)),
                     N_HARM_FEEDBACK_MIN, N_HARM_FEEDBACK_MAX))
    if target is not None and float(target.get("e_high", 0.0)) > 0.3:
        k0 = int(np.ceil(3000.0 / max(float(f0), 1.0))) + 1   # prima armonica sopra 3000Hz
        n = max(n, min(k0 + MIN_HIGH_HARM_FEEDBACK - 1, N_HARM_FEEDBACK_MAX))
    return n


_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_calibration.json")
_calibration_cache = None


def _load_calibration():
    """Carica sample_calibration.json (prodotto da sample_calibration.py
    sui campioni reali in sample/) una sola volta per processo. File
    assente o non ancora generato -> {} (random_target ricade sui range
    euristici, nessun errore): la calibrazione e' un affinamento opzionale,
    non una dipendenza obbligatoria."""
    global _calibration_cache
    if _calibration_cache is None:
        _calibration_cache = {}
        if os.path.exists(_CALIBRATION_PATH):
            try:
                with open(_CALIBRATION_PATH, "r", encoding="utf-8") as f:
                    _calibration_cache = json.load(f)
            except Exception:
                _calibration_cache = {}
    return _calibration_cache


def _sample_calibrated(rng, exciter_type, key, fallback_range, extra=False):
    """Campiona key dalla distribuzione REALE (summary/extra_summary di
    sample_calibration.json, famiglia exciter_type) se disponibile,
    altrimenti uniforme su fallback_range: sostituisce il guess sintetico
    senza richiedere codice diverso a seconda che l'utente abbia gia'
    generato la calibrazione o no. Gaussiana su mean/std (poi clampata al
    fallback_range, mai fuori dal fisicamente plausibile), non bootstrap
    dai raw: piu' semplice e sufficiente per un target scalare per run."""
    cal = _load_calibration().get(exciter_type)
    summ = cal.get("extra_summary" if extra else "summary") if cal else None
    if summ and summ.get("mean", {}).get(key) is not None:
        mean = summ["mean"][key]
        std = summ.get("std", {}).get(key, 0.0)
        lo, hi = fallback_range
        std = std if std > 1e-6 else (hi - lo) * 0.1
        return float(np.clip(rng.normal(mean, std), lo, hi))
    return float(rng.uniform(*fallback_range))


# rapporti f_i/f0 dei modi di una membrana circolare ideale (zeri delle
# funzioni di Bessel, modi (0,1),(1,1),(2,1),(0,2),(3,1),(1,2),(4,1),
# (2,2),(0,3),(5,1),(3,2),(6,1) in ordine di frequenza crescente,
# normalizzati al primo modo) - usati per il warm start di strike, invece
# della serie armonica pura: una membrana/corpo colpito e' naturalmente
# inarmonico, partire dall'armonico costringe il gradiente a scoprire da
# solo l'inarmonicita' (sprecando iterazioni, e spesso restando vicino
# all'armonico - vedi analisi 'migliore qualita' acustica').
MEMBRANE_RATIOS = (1.000, 1.594, 2.136, 2.296, 2.653, 2.918,
                    3.156, 3.501, 3.600, 3.652, 4.059, 4.154)
STRING_INHARM_B = 0.0004   # coefficiente di inarmonicita' di corda rigida tipico (chitarra/arpa): f_k = f0*k*sqrt(1+B*k^2)


def _mode_ratios(exciter_type, n_modes, target=None):
    """Rapporti f_i/f0 per l'inizializzazione dei modi, per famiglia fisica
    dell'exciter invece di k=1,2,3... sempre uguale.

    target: SOLO strike (vedi blend sotto, dopo la tiling) - per gli altri
    branch resta None senza effetto, comportamento identico a prima."""
    if exciter_type == "strike":
        base = list(MEMBRANE_RATIOS)
    elif exciter_type == "pluck":
        base = [k * (1.0 + STRING_INHARM_B * k * k) ** 0.5 for k in range(1, n_modes + 1)]
    elif exciter_type == "chaotic":
        # tam-tam graffiato / death whistle / scream / multifonici: ne' la
        # serie armonica ne' un modello fisico singolo (membrana/corda)
        # sono adatti - serve un pettine DENSO e NON periodico. Successione
        # di Weyl con l'angolo aureo (k*golden mod 1, -0.5..0.5): resta
        # deterministica (warm start riproducibile da target/f0, vedi nota
        # in train_agent_best) ma a bassa discrepanza/mai periodica, la
        # stessa proprieta' "senza pattern" dei modi reali di gong/piastre,
        # invece di clusterizzarsi su pochi rapporti ripetuti come farebbe
        # un base-pattern corto ripetuto (gli altri rami sotto).
        golden = 0.6180339887498949
        k = torch.arange(1, n_modes + 1, dtype=torch.float32)
        jitter = torch.remainder(k * golden, 1.0) - 0.5
        return k * (1.0 + 0.4 * jitter)
    else:
        # bow/blow: armoniche esatte per costruzione (vedi _feedback_synthesize).
        # shaker/noise: nessuna struttura tonale attesa, la serie armonica
        # resta un default neutro.
        base = list(range(1, n_modes + 1))
    ratios = (base * (n_modes // len(base) + 1))[:n_modes]
    if exciter_type == "strike" and target is not None and "inharmonicity" in target:
        # Il warm start Bessel pieno (mix=1) e' fisso indipendentemente dal
        # target, ma _freq_drift_loss (peso 3.0, train_agent) ancora freq_raw
        # PROPRIO li' mentre _inharmonicity_proxy (peso 1.0, solo
        # strike/chaotic) spinge verso la serie armonica quando il target la
        # richiede (es. inharmonicity=0) - la stessa ancora che serve a non
        # perdere la banda del target (vedi _reband_ratios) diventa un
        # avversario piu' forte dell'obiettivo quando il target e' quasi
        # armonico (osservato: inharmonicity achieved 0.40-0.51 contro target
        # 0.0, invariato indipendentemente dalle iterazioni - il drift vince
        # sempre per peso). Fix a monte invece che nel bilanciamento dei pesi
        # (which rischia di far derivare le bande per gli altri target):
        # miscela il warm start verso la serie armonica in proporzione a
        # quanto il target e' vicino ad armonico, cosi' ancora e obiettivo
        # puntano nella stessa direzione invece di scontrarsi. mix=1
        # riproduce ESATTAMENTE il pattern Bessel pieno (nessun cambiamento
        # per target gia' vicini alla nativa inarmonicita' della membrana),
        # mix=0 parte dalla serie armonica pura per target inharmonicity~0.
        k_idx = list(range(1, n_modes + 1))
        native = (sum(((ratios[i] - k_idx[i]) / k_idx[i]) ** 2 for i in range(n_modes)) / n_modes) ** 0.5
        mix = max(0.0, min(1.0, float(target["inharmonicity"]) / max(native, 1e-6)))
        ratios = [k_idx[i] + mix * (ratios[i] - k_idx[i]) for i in range(n_modes)]
    return torch.tensor(ratios, dtype=torch.float32)


def _reband_ratios(ratios, f0, target):
    """strike/pluck: sposta i modi piu' alti (o piu' bassi) del ladder
    fisico dentro una banda che il target chiede ma dove il warm start
    fisico (_mode_ratios) non piazza NESSUN modo - es. strike, membrana:
    rapporti Bessel fino a ~4.15, con f0 alti 3000Hz cade gia' a ratio~9,
    quindi ZERO modi partono sopra 3000Hz anche con target e_high=0.5+. Il
    gradiente deve allora scoprire da solo lo spostamento (osservato:
    mode-collapse su 1 sola riga isolatissima, es. 3045Hz contro 11 modi
    <1300Hz, invece di piu' modi a coprire la banda con continuita').
    Sposta solo il minimo indispensabile - un MINIMO FISSO (REBAND_MIN_MODES),
    non una quota proporzionale al target - e SOLO se il warm start fisico ne
    ha davvero 0/troppo pochi. Quota proporzionale provata e RITIRATA: sposta
    troppi modi (es. 6/12 per e_high=0.529) proprio nella banda dove
    _band_gain_profile (use_reach=True) ricalcola Q sui NUOVI ratio - Q basso
    li' -> reach piccolo -> gain-fix li amplifica ANCORA, le due leve
    (posizione + ampiezza) si sommano e overshoot (osservato: e_high
    0.715 contro 0.529, loss peggiore di prima del reband). Un minimo fisso
    rende solo la banda raggiungibile, lasciando al gain-fix il compito
    (che gia' aveva) di riprodurre la quota esatta."""
    if not all(k in target for k in ("e_low", "e_mid", "e_high")):
        return ratios
    r = ratios.clone()
    n = r.numel()
    r_lo, r_hi = 300.0 / max(f0, 1.0), 3000.0 / max(f0, 1.0)
    e_low, e_high = float(target["e_low"]), float(target["e_high"])
    n_high_needed = REBAND_MIN_MODES if e_high > 0.05 else 0
    n_high_have = int((r > r_hi).sum())
    if n_high_needed > n_high_have:
        k = min(n_high_needed - n_high_have, n)
        top_idx = torch.topk(r, k).indices
        hi_ceil = max(r_hi * 3.0, float(r.max()))
        r[top_idx] = torch.tensor(np.geomspace(r_hi * 1.1, hi_ceil, k), dtype=torch.float32).sort().values
    if r_lo > 1.0:   # sotto 1.0 non esiste una vera "banda bassa" (ratio parte da 1 = f0)
        n_low_needed = REBAND_MIN_MODES if e_low > 0.05 else 0
        n_low_have = int((r < r_lo).sum())
        if n_low_needed > n_low_have:
            k = min(n_low_needed - n_low_have, n)
            bot_idx = torch.topk(-r, k).indices
            r[bot_idx] = torch.tensor(np.geomspace(1.0, r_lo * 0.9, k), dtype=torch.float32).sort().values
    return torch.sort(r).values


# ---------------------------------------------------------------------------
# AGENTE CONDIVISO: banco modale (risonatori a 2 poli, skirt ~12dB/oct)
# ---------------------------------------------------------------------------
def _band_gain_profile(freqs_hz, q, target, n_modes, use_reach, input_amp=None, tau=None):
    """Ripartizione e_low/e_mid/e_high per modo -> gain relativo NON
    normalizzato (il chiamante normalizza) - fattorizzata da warm_start
    per essere richiamabile ANCHE a fine warmup (train_agent) con Q/freq
    aggiornati invece che congelati al warm start: GAIN_FREQ_WARMUP_FRAC
    lascia muovere Q (e per i non-feedback anche freq_raw) PRIMA che
    gain_raw si sblocchi, quindi il profilo Q-dipendente calcolato QUI
    pre-warmup e' gia' stale nel momento esatto in cui _gain_profile_loss
    inizia a mordere (gain_raw ancora congelato durante il warmup, il suo
    gradiente e' scartato) - train_agent lo ricalcola con q/freq correnti
    a fine warmup per restare coerente.

    use_reach=True (strike/pluck/shaker, use_decay_envelope): un modo alto
    ha Q piu' basso (ratio**0.25) quindi ring piu' corto, a parita' di gain
    accumula MENO energia nel buffer di un fix puramente spettrale -
    dividere l'energia-target per reach~Q/f PRIMA della sqrt compensa.
    use_reach=False (bow/blow/noise/chaotic, nessun decay envelope): il
    modo e' pilotato in continuo, l'ampiezza di regime dipende da gain*Q
    direttamente, non da un'energia integrata su un decadimento - la legge
    reach (energia integrata) non si applica li' (misurato: applicata a
    blow, e_mid/e_high invertiti 0.217/0.783 contro target 0.449/0.551,
    sovra-pesando i modi alta frequenza/basso Q oltre il dovuto).
    Divisione per Q PURO (non Q/freq) qui sotto per bow/blow, in AGGIUNTA a
    input_amp**exp, provata e RITIRATA: sommata alla compensazione
    input_amp gia' presente ha reso il training instabile (loss piatta
    1.44-1.47 da it120 a it299, e_mid/e_high ribaltati in overshoot 0.152/
    0.848 contro target 0.449/0.551 - peggio del sottoshoot che doveva
    correggere). Le due correzioni impilate si amplificano a vicenda oltre
    il dovuto; il modello 'gain*Q*input_amp' che le giustificava non regge
    empiricamente (un run precedente con input_amp**1 pieno e NESSUNA
    compensazione Q aveva gia' dato overshoot nella stessa direzione, non
    il sottoshoot che quel modello prevede - segno che manca un fattore,
    forse leakage tra modi a Q basso/skirt larghi).
    Test isolato ORA ATTIVO: Q-only, SENZA input_amp (Q_ONLY_COMP_EXP,
    sostituisce la divisione per input_amp anziche' sommarcisi) - isola se Q
    da solo spiega/corregge il sottoshoot prima di ritentare una
    combinazione."""
    if not all(k in target for k in ("e_low", "e_mid", "e_high")):
        return None
    band_of = torch.bucketize(freqs_hz, torch.tensor([300.0, 3000.0]))
    e_band = torch.tensor([float(target["e_low"]), float(target["e_mid"]), float(target["e_high"])])
    counts = torch.tensor([float((band_of == b).sum()) for b in range(3)])
    reachable = counts > 0
    if not bool(reachable.all()):
        lost = e_band[~reachable].sum()
        e_band = e_band * reachable
        if bool(reachable.any()) and float(e_band[reachable].sum()) > 0:
            e_band[reachable] += lost * e_band[reachable] / e_band[reachable].sum()
    energy_per_mode = torch.zeros(n_modes)
    for b in range(3):
        if counts[b] > 0:
            energy_per_mode[band_of == b] = e_band[b] / counts[b]
    if use_reach:
        # NOTA: reach**3 (invece di reach) provato e RITIRATO - ipotesi che
        # l'energia integrata scalasse come tau^3 per il polo doppio di H^2
        # (vedi sotto), ma il decadimento qui viene da tau_raw (inviluppo
        # esplicito applicato DOPO il filtro in forward(), indipendente da
        # Q, init a 1000s = no-op al warm start) - NON dalla risonanza del
        # filtro stesso. reach=Q/f non era mai una vera legge energetica,
        # solo una costante tarata empiricamente: "correggerla" analiticamente
        # peggiora strike/pluck monotonicamente su entrambi i tentativi
        # (H^2 da solo, poi H^2+reach^3) - centroid pluck 4021->5428->5846.
        # Tornato a reach=Q/f.
        reach = (q / freqs_hz.clamp(min=1.0)).clamp(min=1e-6)
        if tau is not None:
            # il decadimento reale per strike/pluck viene dall'inviluppo
            # esplicito tau_raw (indipendente da Q, vedi forward()), non
            # dalla sola risonanza del filtro: reach=Q/f ignorava questo
            # secondo canale di decadimento, gia' un parametro libero e
            # allenato fin dall'inizio (nessun freeze su tau_raw durante il
            # warmup, a differenza di gain_raw/freq_raw). Il ring effettivo
            # e' il piu' corto dei due meccanismi (chi taglia per primo
            # l'energia) -> min(), non sostituzione: a warm_start tau=1000s
            # e' no-op (>> Q/f per qualunque Q/freq plausibile), comportamento
            # invariato li'.
            reach = torch.minimum(reach, tau.clamp(min=1e-6))
        band_gain = torch.sqrt(energy_per_mode.clamp(min=1e-6) / reach)
    else:
        band_gain = torch.sqrt(energy_per_mode.clamp(min=1e-6))
    if input_amp is not None:
        # is_feedback (bow/blow): un modo alto ha Q piu' basso (vedi
        # warm_start, q=Q_MAX/ratio**0.25) quindi skirt piu' largo/ring piu'
        # corto - test isolato Q-only (SOSTITUISCE la compensazione
        # input_amp, non si somma: vedi docstring sopra per la storia
        # input_amp da sola e input_amp+Q impilate).
        band_gain = band_gain / q.clamp(min=1e-3) ** Q_ONLY_COMP_EXP
    return band_gain


def _tone_amp_profile(f0, n_harm, target):
    """SOLO blow: ampiezza per-armonica del tono di _feedback_synthesize
    derivata da e_low/e_mid/e_high del target, in SOSTITUZIONE della legge
    fissa 1/k^p - stessa logica di _band_gain_profile (ramo use_reach=False,
    energia diretta senza integrazione su decadimento: e' lo stesso regime
    'pilotato in continuo' di bow/blow), applicata pero' all'INGRESSO del
    banco invece che al suo gain.

    Motivo: 1/k^p e' monotona per costruzione, quindi l'ingresso non puo'
    MAI avere piu' energia in alto che in mezzo (blow lo richiede spesso,
    es. e_high=0.551>e_mid=0.449) - un tetto strutturale a monte che nessun
    gain-fix a valle (input_amp**exp, poi Q-only, entrambi provati e mai
    risolutivi: e_high restato ~0.06 in entrambi) puo' correggere, perche'
    dovrebbe invertire un ordine gia' scritto nell'ingresso. Con l'ingresso
    gia' nella forma di banda giusta, la compensazione a valle diventa
    superflua: is_feedback and exciter_type!='blow' in train_agent lascia
    tone_amp_exponent=None per blow, quindi input_amp=None ovunque a valle
    (_band_gain_profile, warm_start) - nessuna doppia correzione.

    Bow NON tocca questa funzione (mai chiamata per bow, vedi call site in
    _feedback_synthesize/train_agent): resta sulla legge 1/k^p originale,
    comportamento identico a prima."""
    k = torch.arange(1, n_harm + 1, dtype=torch.float32)
    freqs_hz = k * float(f0)
    if not all(key in target for key in ("e_low", "e_mid", "e_high")):
        return 1.0 / k   # fallback: harmonicity ancora garantita, decadimento generico se il target non ha bande
    band_of = torch.bucketize(freqs_hz, torch.tensor([300.0, 3000.0]))
    e_band = torch.tensor([float(target["e_low"]), float(target["e_mid"]), float(target["e_high"])])
    counts = torch.tensor([float((band_of == b).sum()) for b in range(3)])
    reachable = counts > 0
    if not bool(reachable.all()):
        lost = e_band[~reachable].sum()
        e_band = e_band * reachable
        if bool(reachable.any()) and float(e_band[reachable].sum()) > 0:
            e_band[reachable] += lost * e_band[reachable] / e_band[reachable].sum()
    energy_per_harm = torch.zeros(n_harm)
    for b in range(3):
        if counts[b] > 0:
            energy_per_harm[band_of == b] = e_band[b] / counts[b]
    amp = torch.sqrt(energy_per_harm.clamp(min=1e-6))
    return amp / amp.sum()


def _pluck_comb_profile(f0, ratios, q, target, n_modes):
    """Ampiezza per-modo di un pizzico ideale (Fletcher & Rossing): una
    corda rilasciata da ferma con spostamento triangolare a frazione p
    della sua lunghezza (p=0 ponte, p=0.5 centro) produce armoniche con
    ampiezza |sin(k*pi*p)|/k^2 - p piccolo (vicino al ponte) sposta il
    primo nodo del comb piu' in alto, dando energia acuta SENZA dover
    forzare gain specifici come fa _band_gain_profile (quota di banda +
    reach, cieca alla forma fisica reale). H(k,kappa)=1/(1+(k*kappa)^2)
    aggiunge il rolloff di un contatto plettro/dito piu' lungo (piu'
    morbido) oltre al comb puro.

    p e kappa sono scelti QUI con una ricerca a griglia chiusa (nessun
    gradiente) invece di essere parametri allenati: una versione
    precedente con posizione+tilt allenati dal gradiente e' stata
    provata e RITIRATA (vedi nota in ModalBank.__init__) - la derivata
    di sin(k*pi*p) cresce con k, instabile su 300 iterazioni di
    ottimizzazione stocastica. Qui serve solo a scegliere un MIGLIOR
    PUNTO DI PARTENZA per gain_raw (che resta libero come prima, 12
    gradi di liberta' intatti) - zero parametri nel grafo di training,
    zero rischio di instabilita'.

    q: serve per la STESSA compensazione reach di _band_gain_profile
    (use_reach=True) - amp^2 e' l'energia FISICA desiderata per modo, ma
    un modo con Q/f piu' alto (reach=q/f piu' lungo) accumula PIU' energia
    integrata nel buffer a parita' di gain. Senza dividere per reach prima
    della sqrt, i modi bassi (gia' favoriti da un reach molto piu' lungo,
    vedi warm_start) restano ANCHE favoriti dal gain assoluto - risultato
    peggiore del warm start precedente (misurato: harmonicity/rolloff/
    e_high tutti peggiorati), non migliore, perche' l'energia desiderata
    dal comb fisico non si traduceva nell'energia poi REALMENTE misurata."""
    if not all(key in target for key in ("e_low", "e_mid", "e_high")):
        return None, None
    k = np.arange(1, n_modes + 1, dtype=np.float64)
    freqs_hz = float(f0) * np.array([float(r) for r in ratios], dtype=np.float64)
    band_of = np.digitize(freqs_hz, [300.0, 3000.0])   # 0=low,1=mid,2=high, come _band_gain_profile
    target_e = np.array([float(target["e_low"]), float(target["e_mid"]), float(target["e_high"])])

    best = None
    for p in np.linspace(0.03, 0.5, 16):
        for kappa in (0.0, 0.05, 0.1, 0.2, 0.4):
            amp = np.abs(np.sin(k * np.pi * p)) / k ** 2 / (1.0 + (k * kappa) ** 2)
            energy = amp ** 2   # energia FISICA desiderata per modo (comb puro, prima del reach)
            e_band = np.array([energy[band_of == b].sum() for b in range(3)])
            tot = e_band.sum()
            if tot <= 1e-12:
                continue
            dist = float(((e_band / tot - target_e) ** 2).sum())
            if best is None or dist < best[0]:
                best = (dist, energy.copy())
    if best is None:
        return None, None
    energy = best[1]
    # Rescale ogni banda del pettine fisico sul target ESATTO: il best-fit
    # (p, kappa) sopra e' solo il MIGLIOR APPROSSIMANTE su una famiglia a 2
    # parametri, quasi sempre con un residuo strutturale (osservato: bias
    # sistematico e_mid overshoot/e_high undershoot identico sia a target
    # armonico che rumoroso, segno di un limite del modello fisico, non di
    # rumore di fit). Quel residuo finiva congelato in gain_profile_init e
    # imposto a peso 9.0 (GAIN_PROFILE_WEIGHT*PLUCK_MULT), dominando il
    # termine diretto e_mid/e_high (peso 2.0) e bloccando gain_raw vicino
    # allo split sbagliato. Riscalare il TOTALE per banda sul target esatto
    # mantiene la forma relativa fisica tra i modi della stessa banda
    # (stesso principio di _band_gain_profile) ma rimuove il tetto di
    # precisione del pettine a 2 parametri.
    e_band_fit = np.array([energy[band_of == b].sum() for b in range(3)])
    # Correzione CAPPATA, non match esatto: un rescale esatto (provato e
    # RITIRATO) amplifica troppo i modi alti (naturalmente deboli nel
    # pettine, energia ~1/k^2) fino a un regime dove il leak strutturale in
    # continua del risonatore a 2 poli (H(0)=1 per OGNI modo: sotto la
    # propria risonanza non e' uno zero come un vero passabanda, lascia
    # passare l'eccitazione a banda larga quasi inalterata) diventa udibile -
    # misurato: e_low 0.0 -> 0.054/0.156 e probe "modi alti soli" da
    # e_low=0.000 a 0.063/0.087, loss aggregata peggiorata su entrambi i run
    # (0.876->0.956, 0.434->0.824). Cap 2.5x testato: migliora run1 (loss
    # -33%, drift 31.2%->17.4%) ma inverte e_mid/e_high nella direzione
    # opposta (0.45/0.55 contro target 0.535/0.465) e regredisce leggermente
    # run2 (loss 0.434->0.484). Ritarato a 2.0x per centrare meglio senza
    # riaprire il leak.
    # Split gain/Q (factor^(1/3) su entrambi) provato e RITIRATO: run seed 0/1
    # mostrano e_low/e_mid/e_high peggiorati rispetto al cap semplice sul solo
    # gain, in ENTRAMBE le varianti di freq LR provate - la leva Q non aiutava
    # quanto sperato e aggiungeva solo un grado di liberta' in piu' da
    # sbagliare. Tornato al cap 2.0x sul solo gain (vedi nota sopra).
    q_np = q.detach().cpu().numpy() if torch.is_tensor(q) else np.asarray(q, dtype=np.float64)
    for b in range(3):
        if e_band_fit[b] > 1e-12:
            factor = np.clip(target_e[b] / e_band_fit[b], 0.0, 2.0)
            energy[band_of == b] *= factor
    # reach=Q/f lineare - vedi nota di reversione in _band_gain_profile.
    reach = np.clip(q_np / np.clip(freqs_hz, 1.0, None), 1e-6, None)
    gain = np.sqrt(np.clip(energy, 1e-12, None) / reach)   # stessa trasformazione di _band_gain_profile
    return torch.tensor(gain, dtype=torch.float32), None


class ModalBank(nn.Module):
    def __init__(self, sr, n_modes=12, f0_init=220.0, use_decay_envelope=False, use_coupling=False,
                 use_legacy_filter=False, use_tail_noise=False):
        super().__init__()
        # use_tail_noise: SOLO strike/pluck (vedi train_agent) - non shaker, che
        # non aveva il gap di harmonicity che ha motivato tail_noise_raw
        # (2026-09-08) e che nel primo test empirico e' REGREDITO includendolo
        # (gap medio harmonicity osservato: 0.009->0.061 su un batch di 6 file) -
        # probabile capacita' inutile che il gradiente ha comunque sfruttato un
        # po', destabilizzando un caso gia' a posto. Disaccoppiato da
        # use_decay_envelope (che shaker mantiene per tau_raw) invece di
        # riusarlo, cosi' shaker resta INVARIATO rispetto a prima di questo fix.
        self.use_tail_noise = use_tail_noise
        self.sr = sr
        self.n_modes = n_modes
        self.use_decay_envelope = use_decay_envelope
        self.use_coupling = use_coupling
        self.use_legacy_filter = use_legacy_filter
        # use_legacy_filter: SOLO noise/chaotic (vedi train_agent) - tengono il
        # vecchio H_i(f)=1/((1-ratio^2)+j*ratio/Q) (picco=Q, H(0)=1). Per
        # entrambi il leak a banda larga sotto risonanza (la causa del gap di
        # harmonicity/e_low analizzato per pluck) non e' un difetto da
        # correggere: noise vuole gia' alta noisiness (il leak la aiuta, non
        # la ostacola) e usa _active_modes_loss apposta per restare a banda
        # larga; chaotic vuole inarmonicita'/roughness deliberate con
        # frequenze libere non armoniche, gia' ben servito cosi' com'e' -
        # nessuna evidenza che il leak gli nuoccia, nessun motivo di
        # ricalibrare gain/Q li' senza necessita'. Tutti gli altri (strike/
        # pluck/shaker/bow/blow) passano al nuovo filtro sotto.
        k = torch.arange(1, n_modes + 1, dtype=torch.float32)
        self.freq_raw = nn.Parameter(torch.log(f0_init * k))          # log-freq: positivo per costruzione
        self.q_raw = nn.Parameter(torch.full((n_modes,), _inv_softplus_t(torch.tensor(8.0)).item()))  # softplus -> Q
        self.gain_raw = nn.Parameter(-1.5 * torch.log(k + 1.0))        # softplus -> gain, decade coi modi alti
        self.gain_profile_init = None   # profilo per-modo target (vedi warm_start/_gain_profile_loss), None = nessun vincolo
        self.freq_init = None   # log-freq di warm start (vedi warm_start/_freq_drift_loss), None = nessun vincolo
        self.q_anchor_target = None   # Fix 2026-09-09: target analitico per q_raw (vedi warm_start/_q_anchor_loss),
        # None = nessun vincolo - stesso pattern di gain_profile_init/freq_init, mai avuto un equivalente per Q
        # (vedi commento nel refresh periodico in train_agent: "Q resta un parametro libero... nessun
        # freq_drift-like anchor su q_raw" - qui il gap si chiude). Isolato via .grad.norm() strumentato:
        # su file dove il ramo achieved/_weighted_loss_terms/_aggregate_loss satura (TERM_CLAMP/tanh, warm
        # start lontano dal target), q_raw non riceve NESSUN altro gradiente per l'intera run.
        # NOTA: la versione a 3 parametri (posizione + tilt, ispirata a
        # Modalys) e' stata provata e RITIRATA - misurata regressione reale
        # (e_high crollato da 0.748 target a 0.005 achieved, pluck instabile
        # con loss che sale da 0.13 a 1.53): un solo tilt monotono non puo'
        # rappresentare spettri con energia concentrata su modi alti
        # specifici, e il comb |sin(k*pi*pos)| ha sensibilita' crescente con
        # k che destabilizza il gradiente. 12 parametri liberi confermati
        # piu' robusti nella pratica.
        if use_decay_envelope:
            # T60 esplicito per modo (punto 8), SOLO per eccitazioni
            # impulsive (strike/pluck/shaker) - vedi forward(). Per un'eccitazione
            # continua (bow/blow/noise) moltiplicare per un
            # inviluppo che parte da t=0 spegnerebbe artificialmente un
            # tono/rumore che dovrebbe restare sostenuto: la "decadenza"
            # non e' osservabile li', il banco filtra un ingresso sempre
            # attivo. tau inizializzato lungo (5s, >> qualunque buffer
            # usato) cosi' all'inizializzazione l'inviluppo e' un no-op e
            # il comportamento gia' convergente (guidato solo da Q) non
            # cambia finche' il gradiente non lo accorcia per inseguire un
            # target di decadimento (t_centroid).
            # tau=5s si e' rivelato NON abbastanza inerte: pluck (target
            # spesso e_high=0.0, ottimizzazione gia' fragile - vedi note
            # su capacita' sprecata) e' regredito nettamente (best-of-3
            # 0.2018->0.5010) solo per la perturbazione residua
            # dell'inviluppo (exp(-1/5)=0.82, non ~1) interagendo con una
            # traiettoria di training gia' delicata. tau=1000s rende
            # l'inviluppo indistinguibile da 1.0 su un buffer di 1s
            # (exp(-1/1000)=0.999) - no-op vero, non solo approssimato.
            self.tau_raw = nn.Parameter(
                torch.full((n_modes,), _inv_softplus_t(torch.tensor(1000.0)).item()))
            # tail_noise_raw: mix di rumore bianco NON filtrato dal banco,
            # inviluppato dallo stesso ring (vedi forward, ring_env) invece
            # di un floor costante - AGGIUNTO 2026-09-08 (indagine harmonicity
            # overshoot su exciter impulsivi molto risonanti: tt-tt_edge/
            # tt-tymp_edge/Hp-harm_fngr). Causa strutturale isolata: per
            # strike/pluck l'eccitazione rumorosa dura al massimo ~21ms
            # (StrikeExciter, fix 2026-09-07), DOPO la quale ModalBank.forward
            # produce SOLO la risposta libera (IFFT) del filtro lineare a
            # quell'impulso gia' esaurito - matematicamente un decadimento
            # sinusoidale puro per costruzione, qualunque siano Q/gain. Su un
            # buffer di 1s, un target che richiede un ring lungo (Q alto, per
            # t_centroid/e_high) passa quindi la maggioranza dei ~40 frame
            # STFT come contenuto puramente tonale: harmonicity/noisiness
            # (pesate per energia, fix 2026-09-07) finiscono strutturalmente
            # vicine al tetto per QUALSIASI target, indipendentemente da
            # quanto rumoroso fosse il transiente iniziale - a differenza di
            # DECAY_ENV_FLOOR (che evita la coda FINTA-silenziosa, gia'
            # risolto), qui non c'e' proprio alcuna leva per una coda
            # genuinamente rumorosa. Un vero tam-tam/timpani a bordo pelle
            # (target harmonicity 0.37/0.32, cioe' ben oltre "puro tono" pur
            # richiedendo un ring lungo) ha invece rumore fisico persistente
            # per tutta la durata del suono (irregolarita' di smorzamento,
            # jitter/beating tra modi quasi degeneri, turbolenza d'aria - non
            # modellato dal banco lineare). tail_noise_raw da' esattamente
            # questa leva mancante. Init quasi nullo (sigmoid(-4)~0.018): no-op
            # sul comportamento gia' tarato (2026-09-07) per la popolazione
            # generale, si attiva solo se il gradiente lo richiede per un
            # target che il solo Q/decadimento non puo' soddisfare.
        if use_tail_noise:
            self.tail_noise_raw = nn.Parameter(torch.tensor(-4.0))
        if use_coupling:
            # accoppiamento non lineare tra coppie di modi (vedi
            # _nonlinear_coupling/ChaoticExciter): due soli scalari
            # allenabili (guadagno + soglia), non una matrice n_modes^2 - il
            # fenomeno da modellare (soglia di ampiezza oltre cui si attiva
            # lo scambio di energia tra modi non armonici - Touze' &
            # Chaigne, "Nonlinear vibrations and chaos in gongs and
            # cymbals") e' globale al banco, non specifico per coppia.
            self.coupling_raw = nn.Parameter(torch.tensor(-2.0))            # softplus*0.3 -> guadagno, init debole
            self.coupling_threshold_raw = nn.Parameter(
                torch.tensor(_inv_softplus_t(torch.tensor(0.15)).item()))   # softplus+0.02 -> soglia ampiezza

    def warm_start(self, target, f0=220.0, is_feedback=False, exciter_type=None, tone_amp_exponent=None):
        """Inizializzazione analitica da descrittori, non casuale: elimina
        la parte piu' lenta della convergenza (trovare l'ordine di
        grandezza giusto di freq/Q/gain).

        is_feedback: per bow/blow il rumore (roughness) bypassa il banco
        (vedi _feedback_synthesize) - qui il banco shape-a SOLO il tono,
        quindi Q va scelto per la resa formantica del tono (alto=picco
        pulito), non abbassato in base al noisiness target come per gli
        altri exciter (dove serve invece lasciare passare rumore a banda
        larga attraverso lo stesso banco).

        exciter_type: seleziona i rapporti f_i/f0 dei modi per famiglia
        fisica (vedi _mode_ratios) invece della serie armonica pura -
        usato anche al posto dell'indice k nelle formule di Q/gain, cosi'
        il decadimento resta legato alla posizione IN FREQUENZA del modo
        e non al suo rango nella serie."""
        with torch.no_grad():
            ratios = _mode_ratios(exciter_type, self.n_modes, target)
            if exciter_type in ("strike", "pluck"):
                # freq_raw libero per questi due (a differenza di bow/blow): un
                # warm start che copre gia' le bande del target evita al
                # gradiente di scoprire da solo lo spostamento (vedi _reband_ratios).
                ratios = _reband_ratios(ratios, f0, target)
            self.freq_raw.copy_(torch.log(torch.tensor(float(f0)) * ratios))
            # ancora per _freq_drift_loss: senza, un modo puo' migrare di banda durante
            # il training (osservato: pluck, drift 29.5%, i modi "medi" finiscono tutti
            # sopra 4800Hz) e il profilo per-modo di _gain_profile_loss resta vincolato
            # a un indice di modo che non corrisponde piu' alla banda di frequenza reale.
            self.freq_init = self.freq_raw.detach().clone()
            noisiness = 0.0 if is_feedback else float(target.get("noisiness", 0.3))
            # Q alto = picco stretto/armonico, Q basso = picco largo/rumoroso.
            # Q_MAX tarato per skirt sufficiente a isolare il picco dal rumore
            # a banda larga (vedi nota su modo a 2 poli sotto). ratio**0.25,
            # non sqrt(ratio): con sqrt il decadimento coi modi alti era
            # troppo aggressivo e affossava harmonicity anche a noisiness
            # target~0.
            q = Q_MAX * (1.0 - 0.9 * noisiness) / (ratios ** 0.25)
            # floor piu' alto per shaker: skirt troppo larghi ad alta noisiness
            # (floor generico 0.5) lasciano trapelare energia sotto 300Hz anche
            # con target e_low=0.0 (osservato: e_low achieved ~0.15, stessa
            # causa strutturale gia' nota per noise ma mai trattata qui).
            q_floor = 3.0 if exciter_type == "shaker" else 0.5
            self.q_raw.copy_(_inv_softplus_t(q.clamp(min=q_floor)))
            self.q_anchor_target = q.clamp(min=q_floor).detach().clone()
            centroid = float(target.get("centroid", f0 * 3))
            rolloff_k = max(1.0, centroid / max(f0, 1.0))
            gain = torch.exp(-((ratios - rolloff_k / 3.0).clamp(min=0)) / max(rolloff_k, 1.0))
            freqs_hz = f0 * ratios
            input_amp = None
            if is_feedback and tone_amp_exponent is not None:
                k_idx = torch.arange(1, self.n_modes + 1, dtype=torch.float32)
                input_amp = 1.0 / (k_idx ** tone_amp_exponent)
            q_adjusted = None
            if exciter_type == "pluck":
                # comb fisico (posizione+durezza del pizzico) invece della quota di
                # banda con reach - vedi _pluck_comb_profile per il perche'. Ora
                # ritorna anche Q corretta (split gain/Q della correzione di banda,
                # vedi _pluck_comb_profile): sovrascrive q_raw sotto, DOPO il warm
                # start iniziale a Q_MAX*(1-0.9*noisiness) qui sopra.
                band_gain, q_adjusted = _pluck_comb_profile(f0, ratios, q, target, self.n_modes)
            else:
                tau_ws = F.softplus(self.tau_raw) + 0.005 if self.use_decay_envelope else None
                band_gain = _band_gain_profile(freqs_hz, q, target, self.n_modes, self.use_decay_envelope,
                                                input_amp=input_amp, tau=tau_ws)
            if band_gain is not None:
                # generalizzato a TUTTI gli exciter type (prima solo chaotic/pluck):
                # il profilo rolloff/3 sopra ignora del tutto e_low/e_mid/e_high, quindi
                # blow/bow/shaker/noise partivano da un profilo cieco alla ripartizione
                # di banda e dovevano scoprirla da zero (mode-collapse osservato anche
                # li', piu' lieve che su pluck ma stessa causa). Partire gia' dalla
                # ripartizione del target restringe il bacino di attrazione prima
                # ancora che il gradiente parta, per qualunque exciter (vedi
                # _band_gain_profile per il gating use_reach per regime fisico).
                gain = band_gain / band_gain.max()
                # profilo target per _gain_profile_loss: vincola gain_raw
                # DIRETTAMENTE durante il training (non solo a init), rompendo
                # la degenerazione a monte invece che lasciarla emergere a
                # valle dalle sole bande aggregate e_low/e_mid/e_high. Ricalcolato
                # a fine warmup in train_agent con Q/freq correnti - vedi nota li'.
                #
                # NON per blow: diagnosticato (log strumentato it0-299) che l'ancora
                # NON deriva (refresh periodico la lascia quasi ferma) e NON e'
                # neppure debole (diventa il termine di loss piu' grande di tutti,
                # ~0.91-0.94 pesato, da it80 in poi) - eppure gain_raw le scappa
                # comunque, concentrando energia su UN modo (osservato: 3567Hz,
                # gain 0.90->2.66). Causa: l'ancora assume distribuzione PIATTA
                # dentro banda, ma il target di blow ha spesso spread stretto
                # (qui 474Hz) che richiede energia CONCENTRATA, non spalmata sugli
                # 8 modi - le due cose sono strutturalmente incompatibili, e il
                # gradiente sceglie giustamente spread/harmonicity (guadagno enorme,
                # 1.75->0.53 e 0.55->0.05) sacrificando l'ancora (che non decresce
                # mai) E le bande (e_high overshoot risultante, 0.06->0.837 nei test
                # precedenti). band_gain resta comunque il warm start di gain_raw
                # (ripartizione iniziale ragionevole, vedi diagnosi: 0.54/0.46 vs
                # target 0.45/0.55) - tolto solo il vincolo che lo blocca li'.
                if exciter_type != "blow":
                    self.gain_profile_init = (gain / gain.sum()).detach().clone()
            self.gain_raw.copy_(_inv_softplus_t(gain.clamp(min=1e-3)))
            if q_adjusted is not None:
                self.q_raw.copy_(_inv_softplus_t(q_adjusted.clamp(min=q_floor)))
                self.q_anchor_target = q_adjusted.clamp(min=q_floor).detach().clone()
            if self.use_coupling:
                # guadagno iniziale piu' alto/soglia piu' bassa quando il
                # target chiede gia' molta inarmonicita'/rugosita' (vedi
                # random_target, range "chaotic") - stesso principio degli
                # altri warm start: partire vicino all'ordine di grandezza
                # giusto invece che da un default neutro sempre uguale.
                roughness_t = float(target.get("roughness", 0.1))
                inharm_t = float(target.get("inharmonicity", 0.1))
                strength = float(np.clip(0.5 * (roughness_t / 0.4 + inharm_t / 0.35), 0.0, 1.0))
                self.coupling_raw.copy_(torch.tensor(_inv_softplus_t(torch.tensor(0.3 + 3.0 * strength)).item()))
                self.coupling_threshold_raw.copy_(
                    torch.tensor(_inv_softplus_t(torch.tensor(0.25 - 0.15 * strength)).item()))

    def forward(self, excitation):
        n = excitation.shape[-1]
        X = torch.fft.rfft(excitation, n=n)
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sr).to(excitation.device)
        f0s = torch.exp(self.freq_raw).clamp(max=NYQUIST_MARGIN * self.sr)
        q = torch.clamp(F.softplus(self.q_raw) + 0.4, max=Q_MAX)  # margine di sicurezza: mai singolarita' esatta
        gain = F.softplus(self.gain_raw)
        # Forma normalizzata (non f_i^2-f^2 in Hz^2): senza normalizzare H ha
        # ampiezza ~1e-5 fuori risonanza e il gain dovrebbe crescere di 4-5
        # ordini di grandezza per compensare, rendendo il training instabile.
        ratio = freqs.unsqueeze(0) / f0s.unsqueeze(1)                         # (n_modes, n_freq)
        real = 1.0 - ratio ** 2
        imag = ratio / q.unsqueeze(1)
        if self.use_legacy_filter:
            # VECCHIO risonatore a 2 poli (skirt ~12dB/oct sopra risonanza):
            # H_i(f) = 1/((1-(f/f_i)^2)+j(f/f_i)/Q_i), picco=Q_i a f=f_i, ma
            # H_i(0)=1 (non zero) - e' la risposta spostamento/forza di un
            # oscillatore smorzato (compliance statica), non un vero
            # passabanda: ogni modo lascia passare l'eccitazione quasi
            # inalterata ben sotto la propria risonanza. Sommato su 12 modi
            # alza il pavimento spettrale tra i picchi (vedi nota su
            # DECAY_ENV_FLOOR/leak sotto 300Hz analizzata per pluck),
            # inflazionando la noisiness misurata - qui MANTENUTO
            # deliberatamente SOLO per noise/chaotic (vedi use_legacy_filter
            # in __init__): per loro il leak non nuoce al target (anzi aiuta
            # noise) e non tocchiamo una calibrazione gain/Q che gia'
            # funziona senza necessita'.
            H = 1.0 / torch.complex(real, imag)
        else:
            # NUOVO: passabanda vero a guadagno di picco costante (RBJ audio-
            # EQ-cookbook, forma standard in sintesi modale/Karplus-Strong
            # commuted synthesis proprio per evitare questo leak):
            # H_i(f) = (j*ratio/Q_i) / ((1-ratio^2)+j*ratio/Q_i), stesso
            # denominatore di prima (stessa banda/Q), ma zero VERO a f=0 (e
            # rolloff -6dB/oct simmetrico sopra risonanza, invece di -12dB/oct
            # solo sopra) - elimina il leak a monte invece di doverlo
            # correggere a valle con cap/split sul gain (vedi
            # _pluck_comb_profile, storia dei tentativi). Picco=1 (non Q):
            # la calibrazione esistente per gli exciter a inviluppo di
            # decadimento (reach=Q/f in _band_gain_profile/_pluck_comb_profile,
            # energia~gain^2*reach) gia' assumeva implicitamente un picco
            # Q-indipendente (vedi commento in _pluck_comb_profile: "un modo
            # con Q/f piu' alto accumula PIU' energia... a PARITA' di gain"
            # - vero solo se il picco non scala gia' con Q da solo) - il
            # nuovo filtro la rende COERENTE con se stessa invece di
            # introdurre una nuova assunzione.
            # FIX: H al quadrato (cascata di due sezioni identiche). H da sola
            # ha rolloff -6dB/oct sopra risonanza (contro i -12dB/oct del
            # vecchio filtro), non menzionato quando introdotta: gli skirt si
            # sovrappongono lungo tutto il banco e l'energia si accumula
            # verso l'alto (centroid/e_high in overshoot, harmonicity giu' -
            # osservato su strike/pluck/shaker/blow, non su bow dove Q e'
            # quasi ovunque al tetto Q_MAX e la banda di transizione resta
            # stretta indipendentemente dalla pendenza asintotica). H^2 ha
            # ancora picco=1 esatto in risonanza (0^2=0 -> H=1 -> H^2=1,
            # nessuna ricalibrazione di gain/Q necessaria) e zero vero a
            # f=0 (0^2=0), ma torna a -12dB/oct sopra risonanza: ripristina
            # la selettivita' che reach=Q/f gia' assumeva implicitamente,
            # senza reintrodurre il leak verso DC del filtro legacy.
            H = torch.complex(torch.zeros_like(imag), imag) / torch.complex(real, imag)
            # NOTA: H*H (cascata di 2 poli identici) provato e RITIRATO -
            # peggiora strike/pluck monotonicamente (loss 0.39->0.45->0.63,
            # 0.70->0.83->1.29), migliora forse shaker ma con varianza troppo
            # alta per concludere. Il polo doppio risultante ha inoltre una
            # risposta diversa da quella assunta da reach=Q/f (altro tentativo
            # di correzione, anch'esso ritirato - vedi nota li'). Tornato a H
            # singola: -6dB/oct sopra risonanza resta un leak noto (vedi nota
            # sopra), non ancora risolto, ma H^2 non e' la soluzione.
        Y_modes = gain.unsqueeze(1).to(torch.complex64) * H * X.unsqueeze(0)  # (n_modes, n_freq)
        if not self.use_decay_envelope and not self.use_coupling:
            return torch.fft.irfft(Y_modes.sum(0), n=n)
        # punto 8 / coupling: IFFT per modo (batched su dim 0, non un giro
        # python) invece di sommare in frequenza e fare una sola IFFT
        # condivisa - necessario sia per moltiplicare ogni modo per il
        # proprio inviluppo di decadimento PRIMA di sommarli (altrimenti la
        # durata del ring resterebbe determinata solo da Q come oggi), sia
        # per l'accoppiamento non lineare sotto (serve il segnale y_i(t) nel
        # TEMPO di ogni modo, non solo la somma in frequenza).
        y_modes = torch.fft.irfft(Y_modes, n=n)                     # (n_modes, n)
        if self.use_decay_envelope:
            tau = F.softplus(self.tau_raw) + 0.005                  # floor 5ms: mai decadimento istantaneo
            t = torch.arange(n, dtype=torch.float32, device=excitation.device) / self.sr
            env = torch.exp(-t.unsqueeze(0) / tau.unsqueeze(1))
            env = env.clamp(min=DECAY_ENV_FLOOR)   # vedi nota su DECAY_ENV_FLOOR sopra
            y_modes = y_modes * env
        y_out = y_modes.sum(0)
        if self.use_tail_noise:
            # coda di rumore bianco NON filtrato, inviluppata dal ring PIU'
            # LENTO tra i modi (env.max(0): finche' anche un solo modo sta
            # ancora suonando, la coda di rumore puo' esserci) - vedi nota su
            # tail_noise_raw in __init__. Bianco (non _colored_noise con lo
            # slope_raw dell'exciter): qui serve energia a banda larga NON
            # correlata alla forma spettrale del banco, per abbassare
            # davvero la flatness pesata per energia durante il ring, non
            # un ennesimo colore che il banco potrebbe comunque ri- "pulire"
            # nella propria selettivita' (non passa per H, e' sommato dopo).
            #
            # FIX 2026-09-08 (bis, dopo il primo test empirico su batch reale):
            # sigmoid pieno (0-1) ha fatto overshoot nella direzione opposta
            # su alcuni file (tt-tymp_edge-ff: h target 0.322 -> achieved
            # 0.067, prima era 0.590 overshoot in su - stesso gap, segno
            # ribaltato; Va-legno_batt-C5: collasso a h=0.000 esatto contro
            # target 0.163) - la leva e' troppo diretta/globale (un solo
            # scalare che mescola rumore non filtrato su TUTTO il ring) e il
            # gradiente puo' spingerla vicino a saturazione prima che
            # Q/gain/tau si assestino. Cap a 0.4 (harmonicity reale minima
            # osservata su strike in sample_calibration.json restava
            # comunque ben sopra "quasi tutto rumore" - vedi nota sul range
            # accorciato di StrikeExciter): limita quanto la coda puo'
            # diventare rumorosa, lasciando comunque margine per i target di
            # questa indagine (0.32-0.37) senza poter piu' collassare a zero.
            ring_env = env.max(0).values
            tail_mix = 0.4 * torch.sigmoid(self.tail_noise_raw)
            tail_noise = torch.randn(n, device=excitation.device)
            tail_noise = tail_noise / (tail_noise.std() + EPS)
            y_out = y_out + tail_mix * ring_env * tail_noise
        if self.use_coupling:
            y_out = y_out + self._nonlinear_coupling(y_modes)
        return y_out

    def _nonlinear_coupling(self, y_modes):
        """Accoppiamento non lineare QUADRATICO tra ogni coppia di modi
        (termine y_i(t)*y_j(t) - la forma standard del coupling nelle
        equazioni di von Karman per piastre/gusci sottili proiettate sui
        modi, vedi Touze' & Chaigne), attivo SOLO sopra una soglia di
        ampiezza allenabile: sotto soglia il banco resta lineare come per
        tutti gli altri exciter (la transizione lineare->caotica nei gong
        reali si osserva solo per colpi forti, non per vibrazioni deboli).
        y_i(t)*y_j(t) genera, per identita' trigonometrica, energia a
        f_i+f_j E |f_i-f_j| - toni di combinazione tra modi NON armonicamente
        correlati, cosa che il banco lineare (che filtra ma non crea mai
        energia a frequenze nuove) non puo' produrre per costruzione.
        Con guadagno/soglia allenati alti: il glide di pitch e il rumore a
        banda larga crescente del tam-tam graffiato/death whistle/scream.
        Con guadagno piu' debole (soglia piu' alta, si attiva solo sui
        picchi): i parziali di combinazione dei multifonici di fiato, dove
        due o piu' regimi restano quasi stabili invece di collassare in
        caos totale - stesso meccanismo, intensita' diversa, appresa dal
        gradiente in base al target (roughness/inharmonicity, vedi
        warm_start e random_target)."""
        idx_i, idx_j = torch.triu_indices(self.n_modes, self.n_modes, offset=1, device=y_modes.device)
        coupling_gain = F.softplus(self.coupling_raw) * 0.3   # scala contenuta: correzione non lineare, non segnale dominante
        threshold = F.softplus(self.coupling_threshold_raw) + 0.02
        yi, yj = y_modes[idx_i], y_modes[idx_j]
        env = 0.5 * (yi.abs() + yj.abs())
        gate = F.relu(env - threshold)   # 0 sotto soglia: banco lineare intatto finche' l'eccitazione e' debole
        return (coupling_gain * gate * yi * yj).sum(0)


def _inv_softplus_t(y):
    """log(expm1(y)) va in overflow (-> inf) in float32 per y>~88: con
    Q_MAX alto il warm start calcola Q fino a 600, quindi q_raw diventava
    inf fin dall'inizializzazione (Adam poi lo trasforma in nan in poche
    iterazioni - causa esatta del loss=nan osservato). Per y grande
    softplus(y)~=y, quindi la sua inversa e' ~=y: nessun overflow."""
    y = y.clamp(min=1e-6)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y.clamp(max=20.0))))


def _normalize_peak(x, target=0.9):
    """Scala lineare (nessuna distorsione) SOLO IN GIU', mai in su, PRIMA
    del limiter. Prima scalava sempre (anche boost fino a 4x su segnali
    sotto target): questo forzava ogni render allo stesso picco fisso,
    cancellando la dinamica reale tra exciter/parametri (es. amp_raw
    smette di avere effetto sul loudness finale - un colpo debole e uno
    forte finiscono entrambi a picco 0.9). Qui si scala in giu' solo
    quanto basta a restare sotto target quando si eccede (sommare 12 modi
    risonanti puo' produrre picchi >>1, misurato: 17x, che il limiter
    altrimenti comprime pesantemente generando distorsione a banda larga
    indistinguibile da rumore reale - un artefatto del rendering, non del
    risonatore); sotto target il segnale passa invariato, dinamica intatta."""
    peak = x.detach().abs().max() + EPS
    scale = torch.clamp(target / peak, max=1.0)   # mai > 1: nessun boost, solo attenuazione se serve
    return x * scale


def _limiter(x):
    x = _normalize_peak(x)
    over = x.abs() > LIMITER_DRIVE
    headroom = 1.0 - LIMITER_DRIVE
    excess = (x.abs() - LIMITER_DRIVE).clamp(min=0.0)
    soft = torch.sign(x) * (LIMITER_DRIVE + headroom * torch.tanh(excess / headroom))
    return torch.where(over, soft, x)


# ---------------------------------------------------------------------------
# EXCITER: un agente per tipo, 2-4 parametri fisici ciascuno
# ---------------------------------------------------------------------------
def _colored_noise(n, sr, slope_raw, center_hz=None, bw=None):
    """Rumore bianco con tilt spettrale 1/f^slope (rosa<->blu), stesso
    meccanismo gia' usato da NoiseExciter. Da' a ogni exciter un asse di
    colorazione spettrale PROPRIO, indipendente dal solo filtraggio del
    banco modale: verificato in sessione che NoiseExciter (l'unico con
    questo controllo) converge molto piu' in fretta e con piu' fedelta'
    sui descrittori spettrali (centroid/rolloff/bande) di strike/pluck/
    scratch, che partono da rumore piatto e delegano tutta la forma
    spettrale al banco modale.

    center_hz/bw (opzionali): enfasi gaussiana aggiuntiva in frequenza,
    indipendente dal tilt 1/f^slope - usata da bow/blow per concentrare il
    roughness vicino a f0 invece di lasciarlo spalmato su tutto lo spettro
    fino a Nyquist (criticita' identificata in analisi: il rumore
    disaccoppiato dal banco modale, punto 2, aveva perso ogni controllo di
    concentrazione spettrale, in conflitto diretto con target a banda
    stretta). Floor 15% fuori banda: mai un notch secco, resta rumore a
    banda larga con solo un'enfasi locale - coerente con la letteratura sul
    rumore di attrito, concentrato ma non discreto come il tono (Serafin et
    al., 'The sound of friction')."""
    X = torch.fft.rfft(torch.randn(n))
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr)
    slope = torch.tanh(slope_raw) * 2.0          # +-2: da rumore rosa a blu
    tilt = 1.0 / (freqs + 1.0) ** slope
    if center_hz is not None:
        emph = torch.exp(-0.5 * ((freqs - center_hz) / bw) ** 2)
        tilt = tilt * (0.15 + 0.85 * emph)
    y = torch.fft.irfft(X * tilt.to(torch.complex64), n=n)
    return y / (y.std() + EPS)


class StrikeExciter(nn.Module):
    """Impatto secco: rumore (con tilt spettrale) con inviluppo
    attacco/decadimento.

    FIX 2026-09-07 (diagnosi harmonicity/noisiness impulsivi): attack/decay
    range accorciato da 0.2-50ms/0.5-800ms a 0.1-3ms/0.3-6ms. Prima
    l'eccitazione (rumore a banda larga) restava attiva fino a
    attack+3*decay=2.4s, ri-eccitando ModalBank.forward() di rumore per
    tutta quella durata - il ring non era mai puramente guidato da Q
    (candidato gia' annotato in train_agent: "pavimento di rumore
    dell'eccitazione/limiter, non il posizionamento dei modi"). Range
    confermato empiricamente (physical_agents_train.py, verifica
    standalone): a parita' di warm start, ridurre la durata massima porta
    harmonicity da ~0.10-0.25 a ~0.17-0.28 GIA' PRIMA di qualunque training
    aggiuntivo, monotono al diminuire della durata; oltre ~1ms/3ms i
    guadagni si appiattiscono (gia' vicino al ceiling fisico osservato su
    sample_calibration.json: harmonicity reale strike media 0.287, max
    0.68) mentre range piu' estremi (0.3ms/1ms) iniziano a sovrastimare
    harmonicity anche per target esplicitamente rumorosi (noisiness
    target 0.96 -> achieved 0.82 invece di restare vicino al target) -
    perdendo la capacita' di rappresentare colpi genuinamente sporchi
    (spazzole, rullante). 0.1-3ms/0.3-6ms e' il compromesso: vicino al
    contact time fisico di un martello/battente rigido (letteratura di
    commuted synthesis/percussion physical modeling, ordine di pochi ms),
    lascia comunque margine per target ad alta noisiness (coperti
    principalmente da Q bassa/gain aperto sul banco, non piu' dalla durata
    dell'eccitazione - vedi warm_start, q_floor). NON tocca DECAY_ENV_FLOOR
    (che resta valido per la coda oltre il floor, non il transiente
    iniziale) ne' la calibrazione di gain_raw in warm_start (dipende dal
    target, non dalla durata dell'eccitatore)."""
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.amp_raw = nn.Parameter(torch.tensor(-1.0))
        self.attack_raw = nn.Parameter(_inv_softplus_t(torch.tensor(0.4)))   # softplus+0.1 -> attack_ms, init ~0.5ms, max 3ms
        self.decay_raw = nn.Parameter(_inv_softplus_t(torch.tensor(1.2)))    # softplus+0.3 -> decay_ms, init ~1.5ms, max 6ms
        self.slope_raw = nn.Parameter(torch.tensor(0.0))   # tilt spettrale, vedi _colored_noise

    def forward(self, n):
        idx = torch.arange(n, dtype=torch.float32)
        # range accorciato (fix 2026-09-07, vedi docstring sopra): 0.1-3ms/
        # 0.3-6ms invece di 0.2-50ms/0.5-800ms - l'eccitazione deve restare
        # vicina a un impulso vero (contact time fisico), non sostenere
        # rumore a banda larga per centinaia di ms mentre ri-eccita il
        # banco. softplus+offset invece di clamp diretto sul parametro
        # (stesso trattamento di q_raw): gradiente non nullo su tutto il
        # range utile.
        a = torch.clamp(F.softplus(self.attack_raw) + 0.1, max=3.0) * 1e-3 * self.sr
        d = torch.clamp(F.softplus(self.decay_raw) + 0.3, max=6.0) * 1e-3 * self.sr
        env = torch.where(idx < a, idx / a, torch.exp(-(idx - a) / d))
        gate = (idx < a + 3 * d).float()   # troncamento netto oltre 3 tau, non coda infinita
        return F.softplus(self.amp_raw) * env * gate * _colored_noise(n, self.sr, self.slope_raw)


class PluckExciter(nn.Module):
    """Pizzico: impulso quasi istantaneo, rumore (con tilt spettrale) con
    inviluppo di decadimento, PIU' un debole accoppiamento con la cassa/
    corpo (vedi in fondo a forward): la corda da sola resta quasi
    istantanea, l'accoppiamento da' invece un secondo salto di energia
    reale che flux (spectral_flux, solo incrementi) puo' misurare."""
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.amp_raw = nn.Parameter(torch.tensor(-1.0))
        self.decay_ms = nn.Parameter(torch.tensor(1.5))
        self.attack_raw = nn.Parameter(_inv_softplus_t(torch.tensor(1.5)))   # softplus+0.5 -> attack_ms, max 8ms (vedi forward)
        self.slope_raw = nn.Parameter(torch.tensor(0.0))   # tilt spettrale, vedi _colored_noise
        # NOTA: il vecchio 'brightness_raw' (mix col delta del segnale,
        # proxy di high-pass) e' stato rimosso - ridondante con slope_raw
        # sulla stessa dimensione (colorazione spettrale), la loro
        # combinazione destabilizzava il training (osservato: loss che
        # oscilla 1.5->2.5 senza convergere dopo l'introduzione del tilt).
        # NOTA 2: punto 6 (range allargato 0.5-500ms) provato e RITIRATO
        # per pluck - a differenza di strike (dove ha aiutato: 0.2151->
        # 0.1996), qui il target ha spesso e_high=0.0 (soppressione
        # totale di alcuni modi, non solo attenuazione): un decay piu'
        # lungo allarga lo spazio di ricerca in cui il banco deve trovare
        # quella soppressione, peggiorando la stabilita' (best-of-3 loss
        # 0.2018 range stretto -> 0.4390 range largo -> 0.3295 con
        # softplus, mai tornato al livello di partenza). Range stretto
        # originale ripristinato.
        #
        # body_*: accoppiamento con la cassa armonica. Verificato che i
        # pluck REALI hanno flux medio 0.118 (sample_calibration.json,
        # non un target irraggiungibile), ma l'attacco primario (max 8ms)
        # sta comunque per intero dentro il primo frame STFT (hop 23ms) -
        # nessuna forma d'onda del SOLO pizzico primario puo' dare un
        # secondo incremento di magnitudo da misurare, qualunque sia il
        # peso in loss (verificato: strutturale, non un problema di
        # ottimizzazione). Una cassa armonica reale non risponde
        # istantaneamente all'eccitazione della corda - si "gonfia" in
        # 10-30ms - questo strato lo modella: piu' debole del pizzico
        # primario (init softplus(-2.5)=0.08 contro softplus(-1.0)=0.31,
        # circa 1/4), quindi non ne alterA il carattere percettivo
        # "istantaneo", ma attraversa sempre almeno un confine di hop
        # (rise+decay tipicamente 30-70ms) dando a flux un vero secondo
        # salto da misurare.
        self.body_amp_raw = nn.Parameter(torch.tensor(-2.5))     # softplus -> ampiezza corpo, init debole
        self.body_rise_raw = nn.Parameter(_inv_softplus_t(torch.tensor(14.5)))    # softplus+0.5 -> rise_ms, init 15ms, max 60ms
        self.body_decay_raw = nn.Parameter(_inv_softplus_t(torch.tensor(29.5)))   # softplus+0.5 -> decay_ms, init 30ms, max 100ms

    def forward(self, n):
        idx = torch.arange(n, dtype=torch.float32)
        d = torch.clamp(self.decay_ms, 0.5, 8.0) * 1e-3 * self.sr
        # attacco breve (1-8ms, NON il decay - quel range e' gia' stato
        # allargato e ritirato, vedi NOTA 2 sopra): senza rampa l'evento e'
        # istantaneo (env=1.0 gia' a idx=0) e cade interamente dentro UN
        # frame STFT (n_fft=2048/hop=1024 @44100Hz, ~23-46ms) - flux (solo
        # incrementi frame-a-frame) osservato a 0.0 ESATTO per costruzione,
        # nessuna forma spettrale per quanto variabile nel tempo puo'
        # smuoverlo se l'intero evento sta in un frame solo. Una rampa che
        # attraversi almeno un confine di hop da' a flux qualcosa da vedere.
        a = torch.clamp(F.softplus(self.attack_raw) + 0.5, max=8.0) * 1e-3 * self.sr
        env = torch.where(idx < a, idx / a, torch.exp(-(idx - a) / d))
        gate = (idx < a + 3 * d).float()
        # un solo filtro FFT statico su tutto il buffer e' stazionario (stessa
        # forma spettrale in ogni frame, a meno del solo livello). Un pizzico
        # reale e' piu' brillante all'attacco (l'energia alta si dissipa piu'
        # in fretta) e si scurisce nel decadimento - due colorazioni indipendenti
        # incrociate nel tempo introducono l'evoluzione spettrale mancante,
        # stesso principio gia' usato da Shaker/Blow per l'inviluppo d'ampiezza,
        # qui sul tilt.
        bright = _colored_noise(n, self.sr, self.slope_raw - 0.6)
        dark = _colored_noise(n, self.sr, self.slope_raw + 0.6)
        mix = torch.exp(-idx / (d * 1.5))   # ~1 a t=0 (bright) -> ~0 a fine decadimento (dark)
        noise = mix * bright + (1.0 - mix) * dark
        pluck = F.softplus(self.amp_raw) * env * gate * noise

        # accoppiamento corpo/cassa: SALE (a differenza del pizzico primario,
        # che scatta subito) su body_rise poi decade su body_decay - vedi
        # nota nel costruttore sul perche' serve per il flux.
        body_rise = torch.clamp(F.softplus(self.body_rise_raw) + 0.5, max=60.0) * 1e-3 * self.sr
        body_decay = torch.clamp(F.softplus(self.body_decay_raw) + 0.5, max=100.0) * 1e-3 * self.sr
        body_env = (1.0 - torch.exp(-idx / body_rise)) * torch.exp(-idx / body_decay)
        body_gate = (idx < body_rise + 4 * body_decay).float()
        body_noise = _colored_noise(n, self.sr, self.slope_raw + 0.3)   # leggermente piu' scuro del pizzico primario
        body = F.softplus(self.body_amp_raw) * body_env * body_gate * body_noise
        return pluck + body


class BowExciter(nn.Module):
    """Arco: tono periodico a poche armoniche (sintesi additiva
    banda-limitata, decadimento 1/k^brightness - approssima il carattere
    del moto stick-slip senza i problemi di stabilita' e di leakage
    spettrale di un vero oscillatore a rilassamento in retroazione) PIU'
    una componente di rumore separata (roughness), con proprio tilt
    spettrale ED enfasi di concentrazione attorno a f0 (noise_focus_raw),
    sommata DOPO il banco modale (vedi _feedback_synthesize): non passa
    per gli stessi risonatori ad alto Q che shape-ano il tono, quindi
    resta rumore anche a valle invece di essere 'ripulita' in energia
    tonale. La concentrazione attorno a f0 (invece di solo tilt globale)
    da' al rumore la stessa capacita' del tono di soddisfare target a banda
    stretta, senza reintrodurre l'accoppiamento col banco modale che
    causava il floor di harmonicity.

    mod_rate/mod_depth: va-e-vieni ritmico dell'arco (analisi bow: 'scratch'
    era meccanicamente un sottoinsieme di questa stessa famiglia a frizione
    continua, con in piu' solo una modulazione periodica dell'ampiezza -
    ScratchExciter e' stato ritirato come classe separata e questa capacita'
    e' stata migrata qui invece di perderla). Init a modulazione quasi nulla
    (mod_depth_raw=-3 -> depth~0.047) per non alterare il comportamento bow
    gia' convergente: il gradiente la alza solo se il target lo richiede."""
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.pressure_raw = nn.Parameter(torch.tensor(-1.0))
        self.brightness_raw = nn.Parameter(torch.tensor(-0.5))
        # FIX 2026-09-11 (diagnosi bow, storia completa):
        # 1) gain_raw isolato in gruppo optimizer con LR*2 - RITIRATO,
        #    nessun effetto misurato (tone/noise gia' non era il collo di
        #    bottiglia: vedi 2).
        # 2) init roughness_raw abbassato a -4.5 (sweep_roughness.py:
        #    roughness=0.001->harmonicity 0.293, 0.01->0.089, 0.05+->piatto
        #    0.02-0.04 - la metrica, flatness spettrale non rapporto di
        #    energia tono/rumore, satura quasi subito) - RITIRATO: su 2/3
        #    file roughness risale comunque a 0.13-0.16 durante il training
        #    (torna da sola in zona satura), su 1/3 il training collassa in
        #    un plateau morto (gradiente ~0 ovunque da it60). Non era un
        #    problema di init/gradiente-zero: il gradiente c'e' ma va nella
        #    direzione sbagliata, dominato da e_low/e_mid/e_high (vedi
        #    HARMONICITY_WEIGHT_BOW_MULT sotto). Ripristinato -2.0
        #    (comportamento originale, nessuna regressione nota).
        self.roughness_raw = nn.Parameter(torch.tensor(-2.0))
        self.noise_slope_raw = nn.Parameter(torch.tensor(0.0))   # tilt del rumore disaccoppiato
        self.noise_focus_raw = nn.Parameter(torch.tensor(-2.0))  # softplus -> bw Hz dell'enfasi attorno a f0, init stretta
        self.mod_rate_hz = nn.Parameter(torch.tensor(6.0))       # Hz del va-e-vieni ritmico (ex ScratchExciter.mod_rate_hz)
        self.mod_depth_raw = nn.Parameter(torch.tensor(-3.0))    # sigmoid -> profondita' AM, init quasi nulla
        # FIX 2026-09-11 (stress test bow n=40: band_crest_3000-20000
        # abs_err mean=23.9 max=136.5, molto oltre gli altri descrittori mai
        # in loss - vedi ricerca letteratura in sessione, Engel et al. 2020
        # DDSP filtered-noise). hf_roughness: seconda banda di rumore
        # scorrelata dal registro (tilt fisso verso l'alto, non center_hz/bw
        # attorno a f0 come il roughness esistente) - il roughness esistente
        # non puo' strutturalmente raggiungere 3-20kHz per f0 bassi.
        # A differenza del jitter di frequenza/ampiezza tentato nello stesso
        # fix (RIMOSSO: nessun termine di loss lo referenzia, misurato solo
        # peggiorare i gap allenati - vedi commit precedente), hf_roughness
        # ha un vero segnale di training dietro: e_high/noisiness_high/
        # harmonicity_high sono gia' in LOSS_WEIGHTS e BANDS[2]=(3000,20000)
        # combacia esattamente con la banda valutata.
        self.hf_roughness_raw = nn.Parameter(torch.tensor(-3.0)) # softplus -> guadagno banda di rumore alta, scorrelata da f0

    def params(self):
        pressure = F.softplus(self.pressure_raw)
        p = torch.clamp(F.softplus(self.brightness_raw) + 0.3, max=4.0)  # esponente 1/k^p: piccolo=brillante, grande=scuro
        # NON limitato (era sigmoid*0.15): un arco/fiato puo' essere tanto
        # 'sporco' quanto pulito - misurato che anche una piccola quantita'
        # di rumore fa saturare la metrica di noisiness verso l'alto, quindi
        # serve poter salire con continuita' fino a dominare del tutto sul
        # tono per coprire target ad alta noisiness (0.25-1.0, come tutti
        # gli altri exciter).
        roughness = F.softplus(self.roughness_raw)
        # bw piccola (via softplus*400+80) = rumore quasi puntiforme su f0
        # (come una riga del tono), bw grande = quasi tilt globale come
        # prima - il gradiente sceglie in base al target.
        focus_bw = F.softplus(self.noise_focus_raw) * 400.0 + 80.0
        mod_rate = torch.clamp(self.mod_rate_hz, 0.5, 40.0)
        mod_depth = torch.sigmoid(self.mod_depth_raw)
        hf_roughness = F.softplus(self.hf_roughness_raw)
        # FIX 2026-09-11 (diagnosi bow, meccanica non loss - vedi tone_only_check.py/
        # cliff_check.py, standalone): tra le armoniche il tono generato e'
        # silenzio digitale ESATTO (zero numerico, non un pavimento basso come
        # in qualunque audio reale - microfono/stanza/crine dell'arco hanno
        # sempre rumore residuo). La metrica di noisiness (flatness spettrale,
        # media geometrica - sensibile a quanti bin sono vicini a zero, non a
        # quanta energia c'e') misurata iper-sensibile proprio a ridosso dello
        # zero: roughness=0 -> noisiness 0.018, roughness=0.00005 (0.005%
        # dell'ampiezza del tono) -> gia' 0.266. roughness=0.002-0.01 copre
        # gia' bene il range target reale (noisiness 0.76-0.84, gradiente
        # misurato ragionevole li'), ma il training INIZIALIZZA/torna sempre
        # oltre 0.05 (zona piatta, vedi HARMONICITY_WEIGHT_BOW_MULT - RITIRATO,
        # gradiente li' e' genuinamente ~0) e per arrivare alla finestra utile
        # deve attraversare questo burrone, dove il gradiente e' numericamente
        # instabile/discontinuo. Floor fisso (NON un nn.Parameter, nessun
        # gradiente, nessuna interazione con l'ottimizzatore) sposta il punto
        # operativo oltre il tratto piu' ripido: noise_floor=0.00005 da solo
        # -> noisiness~0.27, ben sotto il minimo osservato nel dataset reale
        # (0.501 su 287 esempi bow - un arco non e' MAI piu' pulito di cosi',
        # quindi non rende irraggiungibile nessun target reale).
        noise_floor = 0.00005
        return {"pressure": pressure, "p": p, "roughness": roughness,
                "noise_slope": self.noise_slope_raw, "focus_bw": focus_bw, "attack": None,
                "mod_rate": mod_rate, "mod_depth": mod_depth,
                "hf_roughness": hf_roughness, "noise_floor": noise_floor}


class BlowExciter(nn.Module):
    """Fiato: stesso tono additivo banda-limitato del bow (rumore separato,
    sommato dopo il banco modale, con la stessa enfasi di concentrazione
    attorno a f0 - vedi BowExciter), con in piu' una rampa d'attacco
    (respiro)."""
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.pressure_raw = nn.Parameter(torch.tensor(-1.0))
        self.brightness_raw = nn.Parameter(torch.tensor(-0.5))
        self.roughness_raw = nn.Parameter(torch.tensor(-2.0))
        self.attack_ms = nn.Parameter(torch.tensor(40.0))
        self.noise_slope_raw = nn.Parameter(torch.tensor(0.0))   # tilt del rumore disaccoppiato
        self.noise_focus_raw = nn.Parameter(torch.tensor(-2.0))  # vedi BowExciter

    def params(self):
        pressure = F.softplus(self.pressure_raw)
        p = torch.clamp(F.softplus(self.brightness_raw) + 0.3, max=4.0)
        roughness = F.softplus(self.roughness_raw)   # non limitato, vedi nota in BowExciter.params
        focus_bw = F.softplus(self.noise_focus_raw) * 400.0 + 80.0
        attack = torch.clamp(self.attack_ms, 5.0, 400.0) * 1e-3 * self.sr
        return {"pressure": pressure, "p": p, "roughness": roughness,
                "noise_slope": self.noise_slope_raw, "focus_bw": focus_bw, "attack": attack}


class ShakerExciter(nn.Module):
    """Shaker/maraca: meccanica impulsiva/granulare (particelle che
    colpiscono un contenitore), non frizione continua - a differenza di
    bow/scratch NON e' un sottoinsieme della famiglia a frizione (vedi
    analisi: 'scratch' lo era ed e' stato ritirato/migrato in BowExciter,
    'shaker' no). Ispirato al PhISEM di Cook (Physically Informed
    Stochastic Event Modeling: tante particelle in collisione, energia del
    sistema che decade nel tempo) ma approssimato in forma differenziabile
    per il training a gradiente - un vero treno di eventi discreti non e'
    differenziabile rispetto al loro tasso. L'AMPIEZZA del rumore colorato
    e' modulata da un SECONDO rumore passa-basso e raddrizzato (invece di
    un'AM sinusoidale liscia come in bow): il battito irregolare che ne
    risulta e' la firma percettiva di collisioni multiple sovrapposte,
    ottenuta statisticamente invece di simulare ogni singola particella."""
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.amp_raw = nn.Parameter(torch.tensor(-1.0))
        self.decay_raw = nn.Parameter(_inv_softplus_t(torch.tensor(150.0)))  # softplus+20 -> decay_ms, dissipazione energia sistema
        self.rattle_hz_raw = nn.Parameter(torch.tensor(0.0))                 # softplus*40+5 -> cutoff Hz del battito granulare
        self.slope_raw = nn.Parameter(torch.tensor(0.0))                     # tilt spettrale, vedi _colored_noise

    def warm_start(self, target):
        """decay_raw parte da un default fisso (150ms) mai informato dal
        target: tau_raw del banco (ModalBank) puo' SOLO accorciare il
        decadimento (env=exp(-t/tau)<=1), mai allungarlo oltre il tetto
        fissato qui - per un t_centroid alto (fino a 250ms, vedi random_target)
        serve quasi triplicare decay_raw in sole poche centinaia di
        iterazioni. Warm-start diretto da t_centroid, stesso principio di
        ModalBank.warm_start per freq/Q/gain."""
        if "t_centroid" not in target:
            return
        with torch.no_grad():
            decay_ms = max(float(target["t_centroid"]) * 1000.0, 25.0)
            self.decay_raw.copy_(_inv_softplus_t(torch.tensor(decay_ms - 20.0)))

    def forward(self, n):
        idx = torch.arange(n, dtype=torch.float32)
        d = (F.softplus(self.decay_raw) + 20.0) * 1e-3 * self.sr
        env = torch.exp(-idx / d)
        gate = (idx < 6 * d).float()   # coda piu' lunga di strike/pluck (6 tau, non 3): una scossa reale suona anche dopo il picco

        cutoff = F.softplus(self.rattle_hz_raw) * 40.0 + 5.0
        X = torch.fft.rfft(torch.randn(n))
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sr)
        lp = 1.0 / (1.0 + (freqs / cutoff) ** 2)          # passa-basso 2 poli sul rumore modulante
        rattle = torch.fft.irfft(X * lp.to(torch.complex64), n=n)
        rattle = rattle.abs()                              # raddrizzato: inviluppo sempre positivo, non bipolare
        rattle = rattle / (rattle.mean() + EPS)             # media unitaria: non altera il livello complessivo

        noise = _colored_noise(n, self.sr, self.slope_raw)
        return F.softplus(self.amp_raw) * env * gate * rattle * noise


class NoiseExciter(nn.Module):
    """Rumore puro colorato: solo ampiezza + tilt spettrale (1/f^slope)."""
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.amp_raw = nn.Parameter(torch.tensor(-1.0))
        self.slope_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, n):
        return F.softplus(self.amp_raw) * _colored_noise(n, self.sr, self.slope_raw)


class ChaoticExciter(nn.Module):
    """Eccitatore per contenuto fortemente inarmonico/caotico (tam-tam
    graffiato, death whistle, scream, multifonici di fiati): fornisce
    energia a banda larga colorata all'agente condiviso ModalBank, che con
    use_coupling=True (vedi train_agent) genera la vera non-linearita'
    (_nonlinear_coupling) - stessa scelta gia' presa per bow/blow
    (_feedback_synthesize): un oscillatore a rilassamento vero in
    retroazione e' instabile/non affidabile in un training a gradiente (si
    assesta su un punto fisso o diverge, vedi nota li'); qui la struttura
    caotica emerge dal coupling non lineare tra modi, non dal driver.

    Sostenuto o impulsivo (sustain_raw, non un tipo fisso come strike/
    pluck): il target puo' essere tanto un colpo secco/graffio (tam-tam)
    quanto un flusso continuo (death whistle/scream/fiato in multifonico).
    turbulence_raw aggiunge un secondo rumore passa-basso raddrizzato che
    modula l'ampiezza (stesso pattern di ShakerExciter.rattle, qui piu'
    lento: il "respiro"/turbolenza del getto d'aria, non il battito
    granulare di particelle)."""
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self.amp_raw = nn.Parameter(torch.tensor(-1.0))
        self.attack_raw = nn.Parameter(_inv_softplus_t(torch.tensor(4.5)))    # softplus+0.5 -> attack_ms, init 5.0
        self.decay_raw = nn.Parameter(_inv_softplus_t(torch.tensor(59.5)))    # softplus+0.5 -> decay_ms, init 60.0
        self.sustain_raw = nn.Parameter(torch.tensor(-1.0))    # sigmoid -> mix impulsivo<->sostenuto, init verso impulsivo
        self.slope_raw = nn.Parameter(torch.tensor(0.0))       # tilt spettrale, vedi _colored_noise
        self.turbulence_raw = nn.Parameter(torch.tensor(-1.0))       # sigmoid -> intensita' del respiro/turbolenza
        self.turbulence_hz_raw = nn.Parameter(torch.tensor(0.0))     # softplus*8+1 -> Hz del battito di turbolenza (lento)

    def forward(self, n):
        idx = torch.arange(n, dtype=torch.float32)
        a = torch.clamp(F.softplus(self.attack_raw) + 0.5, max=100.0) * 1e-3 * self.sr
        d = torch.clamp(F.softplus(self.decay_raw) + 0.5, max=1500.0) * 1e-3 * self.sr
        transient_env = torch.where(idx < a, idx / a, torch.exp(-(idx - a) / d))
        sustain = torch.sigmoid(self.sustain_raw)
        sustained_env = torch.clamp(idx / a, max=1.0)   # rampa d'attacco poi resta a 1 (nessun decadimento): flusso continuo
        env = (1.0 - sustain) * transient_env + sustain * sustained_env

        noise = _colored_noise(n, self.sr, self.slope_raw)
        turb_cut = F.softplus(self.turbulence_hz_raw) * 8.0 + 1.0
        X = torch.fft.rfft(torch.randn(n))
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sr)
        lp = 1.0 / (1.0 + (freqs / turb_cut) ** 2)          # passa-basso 2 poli sul rumore modulante, come ShakerExciter
        turb = torch.fft.irfft(X * lp.to(torch.complex64), n=n)
        turb = turb.abs()
        turb = turb / (turb.mean() + EPS)                    # media unitaria: non altera il livello complessivo

        turb_mix = torch.sigmoid(self.turbulence_raw)
        excitation = noise * (1.0 - turb_mix + turb_mix * turb)
        return F.softplus(self.amp_raw) * env * excitation


EXCITERS = {
    "strike": StrikeExciter, "pluck": PluckExciter, "bow": BowExciter,
    "blow": BlowExciter, "shaker": ShakerExciter, "noise": NoiseExciter,
    "chaotic": ChaoticExciter,
}


def _feedback_synthesize(exciter, f0, n, sr, n_harm, band_target=None):
    """Tono periodico banda-limitato (n_harm armoniche esatte di f0,
    ampiezza 1/k^p; n_harm scala con f0, vedi _n_harm_feedback) invece del
    rumore in open-loop usato prima: un filtro LTI (il banco modale)
    applicato a un ingresso a spettro discreto produce SOLO quelle
    frequenze in uscita, quindi harmonicity resta alta indipendentemente da
    come il banco modale pesa i modi. Verificato: un vero oscillatore a
    rilassamento in retroazione (stato ricorsivo + attrito non lineare) e'
    stato scartato in fase di verifica numerica - senza un vero termine di
    smorzamento fisico si assesta su un punto fisso (non oscilla) o
    diverge, a seconda del coupling; questa forma additiva evita il
    problema per costruzione.

    band_target: SOLO blow (None per bow, comportamento invariato) - se
    presente, l'ampiezza per-armonica viene da _tone_amp_profile (bande del
    target) invece che da 1/k^p (vedi li' il perche': 1/k^p e' un tetto
    strutturale monotono che blow supera spesso, bow quasi mai).

    Ritorna (tono, rumore) SEPARATI invece di un unico segnale mescolato:
    il chiamante fa passare solo il tono per il banco modale (shaping
    formantico/timbrico sulle armoniche) e somma il rumore DOPO, gia'
    colorato con tilt e concentrazione attorno a f0 indipendenti
    (_colored_noise, vedi nota su focus_bw in BowExciter). Prima i due
    erano mescolati qui e attraversavano insieme lo stesso banco ad alto Q,
    che 'ripuliva' anche il rumore in energia tonale (osservato: harmonicity
    di bow bloccata a ~0.5 contro target 0.107 per tutto il training,
    invariata nonostante roughness fosse allenabile e illimitato)."""
    p = exciter.params()
    pressure, p_exp, roughness = p["pressure"], p["p"], p["roughness"]
    noise_slope, focus_bw, attack = p["noise_slope"], p["focus_bw"], p["attack"]
    mod_rate, mod_depth = p.get("mod_rate"), p.get("mod_depth")   # solo BowExciter (vedi nota mod_rate/mod_depth li'); default no-op altrove
    hf_roughness = p.get("hf_roughness")   # solo BowExciter (vedi classe, FIX 2026-09-11)
    noise_floor = p.get("noise_floor")     # solo BowExciter (vedi classe, FIX 2026-09-11)

    k = torch.arange(1, n_harm + 1, dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32) / sr
    if band_target is not None:
        amp = _tone_amp_profile(f0, n_harm, band_target)
    else:
        amp = 1.0 / (k ** p_exp)
        amp = amp / amp.sum()
    phase = 2.0 * np.pi * k.unsqueeze(1) * float(f0) * t.unsqueeze(0)   # (n_harm, n)
    tone = (amp.unsqueeze(1) * torch.sin(phase)).sum(0)
    tone = tone / (tone.std() + EPS)

    noise_total = roughness * _colored_noise(n, sr, noise_slope, center_hz=float(f0), bw=focus_bw)
    if hf_roughness is not None:
        # seconda banda di rumore, scorrelata dal registro (niente center_hz/
        # bw attorno a f0): tilt fisso verso l'alto (rumore di attrito
        # arco-crine/colofonia, a banda larga - vedi Serafin, 'The sound of
        # friction'), guadagno proprio cosi' il gradiente puo' aggiungerla
        # solo se il target la richiede senza toccare il roughness esistente
        # (gia' convergente, vedi nota focus_bw). Misurato: band_crest_
        # 3000-20000 fuori scala nello stress test bow n=40 (2026-09-11),
        # struttralmente irraggiungibile dal solo roughness concentrato
        # vicino a f0 quando f0 e' basso.
        hf_noise = _colored_noise(n, sr, torch.tensor(-1.2))
        noise_total = noise_total + hf_roughness * hf_noise
    if noise_floor is not None:
        # rumore bianco fisso, indipendente da roughness/hf_roughness - vedi
        # nota in BowExciter.params(). torch.no_grad non necessario: noise_floor
        # e' un float python, non un tensore che richiede gradiente.
        noise_total = noise_total + noise_floor * _colored_noise(n, sr, torch.tensor(0.0))

    if attack is not None:
        env = torch.clamp(t * sr / attack, max=1.0)
        tone = tone * env
        noise_total = noise_total * env
    env_mod_tone = None
    if mod_depth is not None:
        # FIX 2026-09-11 (diagnosi bow tremolo, meccanica non loss - vedi
        # tremolo_check.py, standalone): target flux fino a 0.30 su file
        # tremolo, anche a mod_depth=0.8 (quasi il tetto pratico del
        # sigmoid) flux achieved restava a 0.039 - la modulazione qui
        # applicata al tono PRIMA del banco modale viene quasi azzerata dal
        # filtro: Q arriva a 600, tempo di ring ~Q/(pi*f)~0.8s a queste
        # frequenze, molto piu' lento della modulazione (6Hz~167ms/ciclo) -
        # un risonatore cosi' selettivo ha memoria lunga e appiattisce una
        # modulazione rapida in ingresso. Misurato: stessa modulazione
        # applicata DOPO il banco (sul rumore, che lo bypassa per
        # costruzione e la riceve gia' piena) da' flux=0.186 a depth=0.8
        # contro 0.039 prima - 5x. Inoltre un vero tremolo d'arco e' anche
        # una sequenza di ri-articolazioni nette ad ogni cambio di
        # direzione, non una AM sinusoidale liscia: seno raddrizzato^4
        # (picchi netti, non morbidi) invece di sin() puro, misurato
        # flux=0.38 (oltre il target) contro 0.186 della sinusoide a parita'
        # di depth/rate. env_mod_tone ritornato SEPARATO, NON applicato qui
        # a tone: il chiamante lo applica dopo modal(tone), cosi' la
        # modulazione non attraversa il filtro ad alta Q che la smorzerebbe.
        rect = torch.abs(torch.sin(np.pi * mod_rate * t)) ** 4.0
        env_mod = 1.0 - mod_depth * (1.0 - rect)
        noise_total = noise_total * env_mod   # il rumore bypassa il banco, qui l'effetto e' gia' pieno
        env_mod_tone = env_mod

    return pressure * tone, pressure * noise_total, env_mod_tone


# ---------------------------------------------------------------------------
# TRAINING (breve: pochi parametri + warm start analitico)
# ---------------------------------------------------------------------------
def _weighted_loss_terms(achieved, target, weight_mult=None, db_space=False):
    """Come _descriptor_loss_terms (synth_torch_before.py) ma con peso
    doppio su harmonicity/noisiness (sono i due termini che restano
    indietro rispetto a centroid/rolloff/bande a parita' di iterazioni) e
    con un ramo dedicato per flux - vedi FLUX_NORM_FLOOR.

    weight_mult (opzionale): dict chiave->moltiplicatore applicato SOPRA
    LOSS_WEIGHTS, per pesi extra specifici di un solo exciter type senza
    toccare LOSS_WEIGHTS globale (usato da tutti) - vedi
    HARMONICITY_WEIGHT_BOW_MULT, FIX 2026-09-11.

    db_space (2026-09-11, SOLO bow per ora - vedi call site in train_agent):
    harmonicity/noisiness in (achieved-target)**2 confrontano due sigmoidi
    sature (vedi descriptor_vector_torch, NOISINESS_SHARPNESS) - gradiente
    quasi nullo per gran parte del range (HARMONICITY_WEIGHT_BOW_MULT
    misurato senza effetto: non e' un problema di peso, il gradiente li' e'
    genuinamente ~0). Qui il termine e' ricalcolato in spazio dB PRIMA
    della sigmoide: achieved_db da achieved['flatness'] (gia' presente,
    differenziabile), target_db ricavato invertendo la formula LINEARE con
    cui analyzer.py ha calcolato il target stesso (NON la sigmoide -
    nessuna relazione con la calibrazione LOW/HIGH condivisa da tutti gli
    exciter, quella resta intoccata). Errore quadratico in dB, lineare
    ovunque: nessuna zona piatta. Riportato/loggato invariato altrove
    (achieved['harmonicity']/['noisiness'] restano le sigmoidi di sempre) -
    cambia solo la strada che il gradiente segue dentro la loss."""
    terms = []
    for k, v in target.items():
        if k not in achieved:
            continue
        v = float(v)
        w = LOSS_WEIGHTS.get(k, 1.0) * (weight_mult.get(k, 1.0) if weight_mult else 1.0)
        if db_space and k in ("harmonicity", "noisiness") and "flatness" in achieved:
            achieved_db = 10.0 * torch.log10(achieved["flatness"] + EPS)
            v_noisiness = v if k == "noisiness" else (1.0 - v)
            db_range = NOISINESS_DB_HIGH - NOISINESS_DB_LOW
            target_db = NOISINESS_DB_LOW + v_noisiness * db_range
            raw = w * ((achieved_db - target_db) / db_range) ** 2
        elif k == "flux":
            raw = w * ((achieved[k] - v) / (v + FLUX_NORM_FLOOR)) ** 2
        elif k in RATIO_KEYS:
            raw = w * torch.log2((achieved[k] + EPS) / (v + EPS)) ** 2
        else:
            raw = w * (achieved[k] - v) ** 2
        # Fix 2026-09-09: tanh (coda esponenziale) satura in float32 gia'
        # per raw/TERM_CLAMP > ~7 (tanh a distanza < 1e-6 da 1.0, derivata
        # sotto la precisione rappresentabile) - isolato su strike: warm start
        # con raw error fino a ~44 (spread) manda la maggior parte dei termini
        # in questa zona SUBITO, azzerando il gradiente su q_raw (nessuna anchor
        # loss lo protegge, a differenza di gain_raw/freq_raw) per l'intera run
        # su file dove nessun'altra loss riesce a smuovere raw sotto soglia
        # (verificato via .grad.norm() strumentato: ~1e-11, contro ~1e-3 non
        # appena un file esce dalla saturazione). Clamp razionale al posto di
        # tanh: stesso asintoto TERM_CLAMP, derivata ~TERM_CLAMP^2/(TERM_CLAMP+raw)^2
        # (decade come 1/x^2, non esponenziale) - resta numericamente utile anche
        # a raw~40-50.
        terms.append(TERM_CLAMP * torch.tanh(raw / TERM_CLAMP))
    return terms


def _named_loss_terms(achieved, target, db_space=False):
    """Come _weighted_loss_terms ma con le chiavi, per il breakdown di
    log per-descrittore (diagnosi loss alta: quale termine domina).
    db_space: vedi _weighted_loss_terms, solo per coerenza col numero
    effettivamente ottimizzato nel log (altrimenti il breakdown stampato
    per bow non corrisponderebbe al termine che il gradiente vede davvero)."""
    out = {}
    for k, v in target.items():
        if k not in achieved:
            continue
        v = float(v)
        w = LOSS_WEIGHTS.get(k, 1.0)
        if db_space and k in ("harmonicity", "noisiness") and "flatness" in achieved:
            achieved_db = 10.0 * torch.log10(achieved["flatness"].detach() + EPS)
            v_noisiness = v if k == "noisiness" else (1.0 - v)
            db_range = NOISINESS_DB_HIGH - NOISINESS_DB_LOW
            target_db = NOISINESS_DB_LOW + v_noisiness * db_range
            out[k] = w * float(((achieved_db - target_db) / db_range) ** 2)
        elif k == "flux":
            out[k] = w * float(((achieved[k].detach() - v) / (v + FLUX_NORM_FLOOR)) ** 2)
        elif k in RATIO_KEYS:
            out[k] = w * float(torch.log2((achieved[k].detach() + EPS) / (v + EPS)) ** 2)
        else:
            out[k] = w * float((achieved[k].detach() - v) ** 2)
    return out


def _aggregate_loss(terms):
    """max() puro (vero minimax) verificato instabile con Adam: ad ogni
    step il gradiente passa solo dal termine peggiore, che appena migliora
    passa il testimone a un altro termine -> oscillazione, confermata nel
    test dell'utente (strike/pluck/scratch con loss che sale e scende senza
    convergere). LogSumExp e' un upper bound liscio del max (max <= LSE <=
    max + log(n_terms)/beta): il gradiente si distribuisce su tutti i
    termini vicini al peggiore, non su uno solo. Verificato su un caso
    giocattolo con parametri condivisi tra i termini (come qui: freq/Q/gain
    influenzano piu' descrittori insieme): a parita' di iterazioni LSE
    ottiene un worst-case finale piu' basso E una coda ~20x meno oscillante
    del max puro."""
    stacked = torch.stack(terms)
    m = stacked.max()
    return m + torch.log(torch.exp(MINIMAX_BETA * (stacked - m)).sum()) / MINIMAX_BETA


def _inharmonicity_proxy(modal, f0):
    """Deviazione RMS relativa dei modi dalla serie armonica pura k*f0
    (generalizzazione della B di corda rigida ai rapporti liberi del
    banco), calcolata da freq_raw - differenziabile, nessun render audio.
    Da' gradiente reale al warm start gia' inarmonico di strike/pluck
    (MEMBRANE_RATIOS/STRING_INHARM_B): senza, quell'inarmonicita' resta il
    valore fisso di inizializzazione, mai un obiettivo di training."""
    k = torch.arange(1, modal.n_modes + 1, dtype=torch.float32, device=modal.freq_raw.device)
    ideal = f0 * k
    freq_now = torch.exp(modal.freq_raw)
    return torch.sqrt((((freq_now - ideal) / ideal) ** 2).mean())


def _roughness_proxy(modal):
    """Rugosita' percettiva stile Sethares/Plomp-Levelt (battimento a
    coppie tra i modi, pesato per banda critica: cbw=1.72*fmin^0.65,
    r=x*exp(-3.5x)*min(a1,a2) - vedi Sethares, 'Tuning, Timbre, Spectrum,
    Scale'), calcolata da freq_raw/gain_raw del banco, non dall'audio.
    Applicata SOLO a strike/pluck (vedi random_target): il roughness_raw
    di BowExciter e' rumore a banda larga, un concetto diverso dal
    battimento tra parziali che questo proxy modella - la raccomandazione
    precedente lo assegnava erroneamente a bow, corretto qui."""
    freq = torch.exp(modal.freq_raw)
    amp = F.softplus(modal.gain_raw)
    # frazioni di energia (sommano a 1), non gain assoluti: invariante per
    # scala uniforme del banco (un banco piu' o meno "forte" nel complesso
    # non e' piu' o meno rugoso, solo la distribuzione RELATIVA tra i modi
    # conta) - BUG trovato in test: dividere per total_gain^2 come prima
    # esplode senza limite quando il training spinge tutti i gain verso 0
    # (numeratore ~O(gain), denominatore ~O(gain^2) -> rapporto diverge),
    # osservato numericamente (roughness cresciuta a 2e8 massimizzandola
    # con gradient ascent isolato). Con le frazioni normalizzate la somma
    # e' limitata per costruzione (ogni termine <= 0.105*min(p_i,p_j),
    # niente divisione vicino a zero).
    p = amp / (amp.sum() + EPS)
    fi, fj = freq.unsqueeze(1), freq.unsqueeze(0)
    pi, pj = p.unsqueeze(1), p.unsqueeze(0)
    cbw = 1.72 * torch.clamp(torch.minimum(fi, fj), min=1.0) ** 0.65
    xn = torch.abs(fi - fj) / (cbw + EPS)
    r = xn * torch.exp(-3.5 * xn) * torch.minimum(pi, pj)
    mask = 1.0 - torch.eye(freq.shape[0], device=freq.device)   # esclude la coppia (i,i), battimento nullo con se stesso
    return (r * mask).sum()


def _jitter_proxy(f0, amp, roughness, focus_bw, noise_slope, hf_roughness, n_harm):
    """Proxy differenziabile per 'jitter' (analyzer.py: std/mean delle
    frequenze dei partial rilevati IN UN FRAME - dispersione spettrale
    istantanea, non deriva nel tempo; vedi rimozione jitter_freq/jitter_amp,
    FIX 2026-09-11: quel meccanismo temporale non tocca questa quantita' per
    costruzione ed era scollegato da ogni loss). Grounded nella teoria
    classica di stima di frequenza in rumore (Cramer-Rao / Rife & Boorstyn
    1974): la varianza di stima della frequenza di un parziale cresce con
    l'inverso del SNR locale attorno a quella frequenza - un peak-picker
    disperde la frequenza rilevata in proporzione al rumore relativo al
    tono li' vicino. Stessa formula analitica di inviluppo di _colored_noise
    (niente render/FFT, calcolata in forma chiusa dai parametri, come
    _roughness_proxy/_inharmonicity_proxy) quindi resta coerente per
    costruzione col rumore realmente generato da BowExciter."""
    # BUG 2026-09-11 (trovato nel primo test su 3 file: jitter achieved
    # 78-216 contro target ~0.6, loss saturata al ceiling 6.0 su tutti e 3):
    # tilt_focus/tilt_hf qui sotto sono le stesse forme non normalizzate di
    # _colored_noise, che pero' li' hanno senso solo DOPO la rinormalizzazione
    # a std unitario dell'INTERO segnale (y/(y.std()+EPS)) - usate punto per
    # punto come densita' di potenza assoluta sulle sole frequenze armoniche
    # esplodono (specialmente tilt_hf, slope negativo -> (freq+1)^|slope|
    # non ha limite superiore verso Nyquist). Normalizzate a somma 1 sulle
    # armoniche valutate (come amp, gia' normalizzata) restano pesi RELATIVI
    # ben posti, indipendenti dalla loro scala assoluta - la direzione
    # (piu' rumore vicino/lontano da un'armonica -> quell'armonica pesa di
    # piu' nella dispersione) resta la stessa, solo la scala e' sana.
    k = torch.arange(1, n_harm + 1, dtype=torch.float32)
    freqs = k * float(f0)
    slope = torch.tanh(noise_slope) * 2.0
    tilt_focus = (1.0 / (freqs + 1.0) ** slope) * (0.15 + 0.85 * torch.exp(-0.5 * ((freqs - float(f0)) / focus_bw) ** 2))
    tilt_focus = tilt_focus / (tilt_focus.sum() + EPS)
    tilt_hf = 1.0 / (freqs + 1.0) ** (-1.2)   # stesso tilt fisso di _feedback_synthesize (hf_noise)
    tilt_hf = tilt_hf / (tilt_hf.sum() + EPS)
    noise_psd = roughness ** 2 * tilt_focus + hf_roughness ** 2 * tilt_hf
    tone_power = amp ** 2
    snr = tone_power / (noise_psd + EPS)
    return torch.sqrt((1.0 / (snr + EPS)).mean())


def _gain_profile_loss(modal):
    """Penalizza (KL pesata) la distribuzione normalizzata di gain_raw
    rispetto al profilo per-modo target salvato in warm_start (gain-fix
    pluck/chaotic: ripartizione e_low/e_mid/e_high per modo, non il
    rolloff/3 generico). None per gli altri exciter (vedi
    ModalBank.gain_profile_init) -> no-op, comportamento identico a prima
    per loro. Vincola gain_raw a monte invece di lasciare che siano solo le
    bande aggregate e_low/e_mid/e_high a farlo indirettamente a valle, dove
    assegnazioni per-modo diverse possono dare la stessa energia di banda
    (la degenerazione dietro al mode-collapse)."""
    q = modal.gain_profile_init
    if q is None:
        return None
    amp = F.softplus(modal.gain_raw)
    p = amp / (amp.sum() + EPS)
    return (q * (torch.log((q + EPS) / (p + EPS)))).sum()


def _active_modes_loss(modal, min_active):
    """Penalizza il collasso su pochi modi (proxy: perplexity della
    distribuzione di gain normalizzata, exp(entropia) - un 'numero
    effettivo' di modi attivi continuo e differenziabile, non un conteggio
    discreto). Serve solo dove un singolo picco non basta a rendere il
    suono percepito come tonale: noise e' un'eccitazione CONTINUA (il
    pavimento di rumore a banda larga si rialimenta di continuo, a
    differenza di strike/pluck dove dopo il transiente resta solo il
    ringdown dei modi) - un solo modo ad alto Q, per quanto stretto, pesa
    poco sull'integrale di potenza totale rispetto al pavimento, quindi la
    flatness/harmonicity resta bloccata (osservato: noise, 2/12 modi
    attivi, harmonicity 0.038 contro target 0.58). Serve un pettine di piu'
    modi simultaneamente attivi, non solo un modo piu' forte."""
    amp = F.softplus(modal.gain_raw)
    p = amp / (amp.sum() + EPS)
    entropy = -(p * torch.log(p + EPS)).sum()
    eff_n = torch.exp(entropy)
    return F.relu(min_active - eff_n) ** 2


def _freq_drift_loss(modal):
    """Deviazione quadratica media di freq_raw (log-freq: la differenza e'
    gia' un log-rapporto, quindi la deriva RELATIVA) dal warm start salvato
    in ModalBank.freq_init. None per is_feedback (freq_raw non e' allenabile
    li', vedi train_agent) -> no-op. Senza questo vincolo un modo puo'
    migrare liberamente di banda durante il training, invalidando qualunque
    profilo per-modo ancorato all'indice di modo (vedi _gain_profile_loss:
    stesso gain su un modo che ha cambiato banda produce comunque il
    descrittore di banda sbagliato)."""
    init = modal.freq_init
    if init is None:
        return None
    return ((modal.freq_raw - init) ** 2).mean()


def _q_anchor_loss(modal):
    """Ancora indipendente per q_raw verso Q_MAX*(1-0.9*noisiness)/ratios**0.25
    (stessa formula analitica di warm_start, salvata in modal.q_anchor_target e
    ricalcolata con freq_raw CORRENTE nel refresh periodico di train_agent - vedi
    li'). Log2-ratio (Q e' una quantita' moltiplicativa, stesso stile di RATIO_KEYS)
    invece di errore assoluto: Q spazia 0.5-600, un MSE diretto sarebbe dominato dai
    modi ad alta frequenza/Q alto. Stesso pattern di _gain_profile_loss/_freq_drift_loss:
    gradiente SEMPRE vivo, indipendente da achieved/_weighted_loss_terms/_aggregate_loss."""
    target_q = modal.q_anchor_target
    if target_q is None:
        return None
    q = torch.clamp(F.softplus(modal.q_raw) + 0.4, max=Q_MAX)
    return (torch.log2((q + EPS) / (target_q + EPS)) ** 2).mean()


def _mode_repulsion_loss(modal):
    """Penalizza coppie di modi troppo vicine in log-frequenza (sotto
    MODE_MIN_SPACING) - vedi nota su MODE_REPULSION_WEIGHT: una membrana
    reale non ha risonanze coincidenti, il collasso e' un artefatto del
    gradiente che spreca anche un grado di liberta' utile al gain-fit."""
    diff = (modal.freq_raw.unsqueeze(1) - modal.freq_raw.unsqueeze(0)).abs()
    mask = 1.0 - torch.eye(diff.shape[0], device=diff.device)
    return (F.relu(MODE_MIN_SPACING - diff) ** 2 * mask).sum() / 2.0   # /2: ogni coppia contata due volte


def train_agent(exciter_type, target, f0=220.0, n_modes=12, seconds=1.0,
                 sr=SR_DEFAULT, iters=300, lr=0.06, noise_samples=3, log=None,
                 use_predictor=True):
    """noise_samples: media la loss su K realizzazioni stocastiche
    dell'eccitatore per step, altrimenti il gradiente e' stimato su una
    singola estrazione di rumore e la loss oscilla senza convergere
    (osservato: loss non monotona a K=1).

    use_predictor: se un resonator_predictor.py addestrato esiste per
    exciter_type (strike/pluck/shaker/noise - vedi predictor_dataset_gen.py),
    corregge il warm_start con un delta imparato PRIMA di questo loop -
    mitiga i minimi locali del gradient descent per-istanza (osservato: il
    mode-collapse di pluck). No-op silenzioso se il checkpoint non esiste
    (import lazy per non creare una dipendenza circolare/obbligatoria:
    resonator_predictor.py importa DA questo modulo)."""
    exciter = EXCITERS[exciter_type](sr)
    if hasattr(exciter, "warm_start"):
        exciter.warm_start(target)   # es. ShakerExciter.decay_raw da target["t_centroid"]
    if exciter_type == "bow" and "jitter" not in target:
        # FIX 2026-09-11: target["extra"]["jitter"] esiste gia' (analyze_reference,
        # AnalyzerV3) ma restava solo metadato, mai un obiettivo di training -
        # vedi _jitter_proxy. Promosso a chiave top-level SOLO per bow: e'
        # quello che _weighted_loss_terms (guidata da target.items()) confronta
        # con achieved.
        _extra = target.get("extra", {})
        if "jitter" in _extra:
            target["jitter"] = _extra["jitter"]
    is_feedback = exciter_type in FEEDBACK_TYPES
    # per bow/blow la frequenza dei modi resta congelata al warm start
    # (vedi sotto) esattamente sulle n_modes_eff armoniche sintetizzate
    # dall'eccitatore (vedi _n_harm_feedback): modi oltre quel numero non
    # ricevono mai energia in ingresso alla loro frequenza (verificato nei
    # log, col vecchio conteggio fisso N_HARM_FEEDBACK=8: gain~0 e nessun
    # contributo utile sui modi 9-12) - allineare n_modes elimina questa
    # capacita' sprecata.
    n_modes_eff = _n_harm_feedback(f0, target) if is_feedback else n_modes
    # T60 esplicito (punto 8) solo per eccitazioni impulsive: vedi nota in
    # ModalBank.__init__ sul perche' non ha senso per bow/blow/shaker/noise.
    use_decay_env = exciter_type in ("strike", "pluck", "shaker")
    # shaker aggiunto: l'exciter ha gia' un inviluppo decadente (ShakerExciter,
    # non e' un'eccitazione sostenuta), quindi la stessa logica di strike/pluck
    # si applica - prima t_centroid doveva passare SOLO da ShakerExciter.decay_raw
    # (leva condivisa con la forma spettrale via Q), ora ha una leva indipendente.
    use_coupling = exciter_type == "chaotic"   # accoppiamento non lineare tra modi, vedi ModalBank._nonlinear_coupling
    use_legacy_filter = exciter_type in ("noise", "chaotic")   # vedi nota in ModalBank.__init__/forward
    use_tail_noise = exciter_type in ("strike", "pluck")   # vedi nota use_tail_noise in ModalBank.__init__: NON shaker
    modal = ModalBank(sr, n_modes=n_modes_eff, f0_init=f0, use_decay_envelope=use_decay_env,
                       use_coupling=use_coupling, use_legacy_filter=use_legacy_filter,
                       use_tail_noise=use_tail_noise)
    # p iniziale del tono additivo (vedi BowExciter/BlowExciter.params, "p"):
    # serve al gain-fix per compensare l'attenuazione 1/k**p gia' presente
    # nell'ingresso al banco per is_feedback - vedi _band_gain_profile.
    # exciter_type!='blow': blow non usa piu' 1/k^p (vedi _tone_amp_profile,
    # _feedback_synthesize) quindi non c'e' piu' nulla da compensare a valle -
    # None qui si propaga da solo a input_amp=None ovunque sotto (warm_start,
    # _band_gain_profile, refresh periodico), senza toccare bow.
    tone_amp_exponent = float(exciter.params()["p"].detach()) if (is_feedback and exciter_type != "blow") else None
    modal.warm_start(target, f0=f0, is_feedback=is_feedback, exciter_type=exciter_type,
                      tone_amp_exponent=tone_amp_exponent)
    if use_predictor and not is_feedback:
        try:
            from resonator_predictor import predict_and_apply
            predict_and_apply(exciter_type, exciter, modal, target, f0)
        except Exception:
            pass   # predittore non allenato/non disponibile: warm_start analitico resta l'unico init, comportamento identico a prima

    n = int(round(seconds * sr))
    freq_params = []
    # decay_raw/attack_raw (Strike/Shaker/Chaotic): parametri d'inviluppo
    # temporale, stessa patologia gia' risolta per q_raw - col clip_grad_norm_
    # globale il loro gradiente resta diluito e il salto necessario dal default
    # fisso (es. Shaker 150ms -> t_centroid target fino a 250ms) non si compie
    # nel budget di iterazioni. LR dedicato piu' alto, come q_raw.
    env_params = [p for name, p in exciter.named_parameters() if name in ("decay_raw", "attack_raw")]
    env_param_ids = {id(p) for p in env_params}
    exciter_base_params = [p for p in exciter.parameters() if id(p) not in env_param_ids]
    gain_params = []
    if is_feedback:
        # la frequenza dei modi resta fissa al warm start (f0*k, k=1..n_modes):
        # l'eccitatore genera armoniche esatte di f0, se freq_raw derivasse
        # durante il training i modi si disallineerebbero dalle armoniche
        # dell'eccitatore e harmonicity ne risentirebbe.
        # FIX 2026-09-11 (diagnosi bow: 40/40 file su stress test n=40 con
        # harmonicity achieved < target, zero eccezioni - bias direzionale
        # sistematico, non dispersione statistica): per bow/blow il rumore
        # bypassa il banco modale (va dritto in uscita a piena scala - vedi
        # _feedback_synthesize), il tono invece deve attraversare il
        # risonatore. gain_raw e' l'UNICA leva che puo' alzare l'energia
        # tonale dopo il filtraggio (freq_raw e' congelato, a differenza di
        # shaker dove puo' aiutare a ridistribuire energia) - ma condivideva
        # lo stesso gruppo/LR di roughness/hf_roughness/focus_bw ecc. Gruppo
        # optimizer dedicato con LR raddoppiato, stesso pattern gia' in uso
        # per q_raw/env_params/tau_params (gradiente diluito nel gruppo
        # comune, vedi note li').
        other_params = exciter_base_params
        gain_params = [modal.gain_raw]
    else:
        other_params = exciter_base_params + [modal.gain_raw]
        freq_params = [modal.freq_raw]
    tau_params = []
    tail_noise_params = [modal.tail_noise_raw] if use_tail_noise else []
    # tail_noise_raw in un gruppo LR TUTTO SUO (base lr, non lr*2 come q_raw/
    # tau_params): e' una leva coarse/globale (un solo scalare che mescola
    # rumore su tutto il ring), a differenza di Q che e' per-modo - un LR
    # aggressivo su una leva cosi' diretta e' probabilmente la causa
    # dell'overshoot/collasso osservato nel primo test empirico (vedi nota
    # in forward()), non solo il cap mancante.
    if use_decay_env:
        if exciter_type == "pluck":
            # tau_raw isolato in un proprio gruppo LR (come q_raw/env_params sotto), invece di
            # restare diluito in other_params insieme a gain/exciter-base (molto piu' numerosi) -
            # stessa patologia gia' risolta per Q. Qui serve a scollegare la DURATA del ring
            # (reach, oggi read solo da Q in _band_gain_profile) dalla nitidezza spettrale che Q
            # deve dare per harmonicity: con tau_raw libero di muoversi sul serio, Q puo' assestarsi
            # sul target di harmonicity/noisiness mentre tau compensa la reach, invece di dover fare
            # da solo entrambi i compiti in conflitto (osservato: harmonicity 0.55 contro target
            # 0.748 con e_mid/e_high invertiti). SOLO pluck per ora: strike/shaker restano con
            # tau_raw in other_params, comportamento identico a prima.
            tau_params = [modal.tau_raw]
        else:
            other_params = other_params + [modal.tau_raw]
    if use_coupling:
        other_params = other_params + [modal.coupling_raw, modal.coupling_threshold_raw]
    params = other_params + freq_params + [modal.q_raw] + env_params + tau_params + tail_noise_params
    # LR piu' alto dedicato a Q: col clip_grad_norm_ globale il suo gradiente
    # (piu' debole di freq/gain) veniva diluito e Q non usciva mai dal warm
    # start, tenendo harmonicity/noisiness bloccate indipendentemente dal
    # target.
    # freq_raw per chaotic a LR ridotto (0.2x): a lr pieno harmonicity
    # migliora ma poi drift 79% e inversione e_low/e_mid (i termini
    # t_centroid/centroid, enormi a inizio training, lo trascinano fuori
    # dalla banda gia' corretta dal warm_start prima che gain/Q si
    # assestino). Il freeze totale provato prima peggiora ancora di piu'
    # (disallinea freq_raw dallo scheduler cosine). LR ridotto, non zero:
    # si muove ma piano, senza le patologie di entrambi gli estremi.
    freq_lr_mult = 1.0  # riduzione LR provata per chaotic e bocciata: migliora e_low/inharmonicity
                        # ma peggiora rolloff/t_centroid, loss aggregata piu' alta (5.27 vs 3.62)
    # PLUCK_FREQ_LR_MULT (0.35x) provato e RITIRATO qui: run seed 0/1 mostrano drift
    # dimezzato (17%->11%) ma harmonicity INVARIATA (0.371->0.366, 0.303->0.311) - la
    # causa del gap harmonicity non e' la mobilita' di freq_raw come ipotizzato. Nel
    # frattempo togliere quella mobilita' ha tolto una leva a centroid/rolloff/spread,
    # che l'hanno scaricata sul gain - stesso parametro del leak DC (vedi
    # _pluck_comb_profile) - e infatti e_low e' tornato a fare leak (0.134 su run1,
    # prima 0.0) e e_mid/e_high sono peggiorati in entrambi i run nonostante lo split
    # gain/Q. freq_raw resta a LR pieno finche' non si trova la causa reale del gap
    # di harmonicity (candidato: pavimento di rumore dell'eccitazione/limiter, non il
    # posizionamento dei modi).
    opt = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": freq_params, "lr": lr * freq_lr_mult},
        {"params": [modal.q_raw], "lr": lr * 2},
        {"params": env_params, "lr": lr * 2},
        {"params": tau_params, "lr": lr * 2},
        {"params": tail_noise_params, "lr": lr},   # base lr, non lr*2 - vedi nota su tail_noise_params sopra
        {"params": gain_params, "lr": lr * 2},     # solo is_feedback (bow/blow) - vedi nota sopra, FIX 2026-09-11
    ])
    # decay del lr sulle iterazioni: senza, il passo resta fisso a lr anche
    # a fine training e il minimax LogSumExp continua a rincorrere il
    # termine peggiore a piena ampiezza, producendo overshoot (osservato:
    # pluck, centroid 0.05->1.98 e loss 0.67->2.07 tra it16 e it79 invece
    # di convergere). Cosine annealing come in synth_torch_before.fit_to_target.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    achieved, loss_val = {}, None
    for it in range(iters):
        opt.zero_grad()
        loss_acc, achieved = 0.0, {}
        for _ in range(noise_samples):
            if is_feedback:
                bt = target if exciter_type == "blow" else None
                tone, noise, env_mod_tone = _feedback_synthesize(exciter, f0, n, sr, n_harm=n_modes_eff, band_target=bt)
                y_tone = modal(tone)   # solo il tono passa per il banco, il rumore no (vedi _feedback_synthesize)
                if env_mod_tone is not None:
                    y_tone = y_tone * env_mod_tone   # tremolo applicato DOPO il banco - vedi FIX 2026-09-11 li'
                y = _limiter(y_tone + noise)
            else:
                y = _limiter(modal(exciter(n)))
            achieved = descriptor_vector_torch(y, sr)
            if exciter_type == "bow" and is_feedback:
                # Gradient routing (FIX 2026-09-11, risposta al trade-off del
                # tentativo db_space RITIRATO sopra): harmonicity/noisiness per
                # bow ricalcolati su y_tone.detach()+noise - stesso valore
                # numerico (detach non cambia il forward), ma il gradiente di
                # questi due termini non attraversa piu' freq_raw/gain_raw/
                # q_raw del ModalBank condiviso con rolloff/centroid/spread/
                # e_low - solo i parametri di rumore dedicati (roughness_raw,
                # hf_roughness_raw, noise_floor, noise_slope_raw,
                # noise_focus_raw, pressure_raw via il ramo noise). Stesso
                # principio della separazione sorgente/filtro nei modelli
                # source-filter differenziabili (NSF, Wang et al. 2019).
                #
                # ESTESO a blow il 2026-09-11, RITIRATO il 2026-09-12: gap
                # harmonicity/noisiness peggiorato (n=40 seed=0, 0.149->0.162)
                # invece di migliorare. Prima esclusa l'ipotesi floor/cliff su
                # roughness_raw (blow_roughness_correlate.py: std=0.183 ~
                # mean=0.183, corr(roughness,gap)=-0.056, range 0.015-0.714).
                # Con l'interferenza di gradiente col ModalBank condiviso
                # anch'essa esclusa dalla misura, la causa del gap resta
                # aperta - non sono la stessa causa di bow. Indagine da
                # ripartire dalla meccanica di generazione di BlowExciter
                # (_tone_amp_profile/band_target) invece di riusare fix gia'
                # pronti da bow.
                y_hn = _limiter(y_tone.detach() + noise)
                achieved_hn = descriptor_vector_torch(y_hn, sr)
                achieved["harmonicity"] = achieved_hn["harmonicity"]
                achieved["noisiness"] = achieved_hn["noisiness"]
            if exciter_type in ("strike", "chaotic"):
                # strike: vedi nota storica sotto (pluck escluso, regressione
                # misurata). chaotic: qui inharmonicity/roughness NON sono un
                # affinamento opzionale ma l'obiettivo primario dell'agente
                # (materiale intenzionalmente molto inarmonico, vedi
                # random_target) - freq_raw e' libero (come strike, mai
                # congelato sulle armoniche) quindi il proxy da' gradiente
                # reale, stesso meccanismo che gia' funziona li'.
                #
                # SOLO strike/chaotic, non pluck: su run reale (log utente) i
                # due proxy competevano con centroid/e_bands su pluck e
                # vincevano, collassando il risultato (7/12 modi attivi,
                # e_high 0.988 invece di ~0.47 target, noisiness invertita
                # 0.71 vs 0.25 target) SENZA nemmeno avvicinarsi al proprio
                # target di inarmonicita' (0.81 raggiunto vs 0.008 target) -
                # regressione netta rispetto al loss storico di pluck
                # (~0.20 -> 0.75). pluck era gia' il piu' fragile (vedi NOTA
                # in PluckExciter: e_high spesso 0.0, range decay gia'
                # ristretto per instabilita' simile) - due gradi di liberta'
                # in piu' sullo stesso freq_raw lo destabilizzano oltre la
                # soglia utile. strike/chaotic (piu' modi tipicamente attivi,
                # warm start meno estremo) li assorbono bene.
                achieved["inharmonicity"] = _inharmonicity_proxy(modal, f0)
                achieved["roughness"] = _roughness_proxy(modal)
            elif exciter_type == "bow" and "jitter" in target:
                p_now = exciter.params()
                k_now = torch.arange(1, n_modes_eff + 1, dtype=torch.float32)
                amp_now = 1.0 / (k_now ** p_now["p"])
                amp_now = amp_now / amp_now.sum()
                achieved["jitter"] = _jitter_proxy(
                    f0, amp_now, p_now["roughness"], p_now["focus_bw"],
                    p_now["noise_slope"], p_now["hf_roughness"], n_modes_eff)
            # db_space (FIX 2026-09-11, vedi _weighted_loss_terms) provato per bow:
            # RITIRATO 2026-09-11 - misurato su n=40 seed=0, stesse impostazioni
            # del baseline: harmonicity/noisiness mean gap PEGGIORA (0.100->0.123),
            # rolloff/centroid/spread sistematicamente peggiori (mean log2-ratio
            # err ~0.15-0.16). Il gradiente onesto in dB space costa piu' agli
            # altri termini nel LogSumExp di quanto renda - trade-off strutturale
            # nella direzione sbagliata, non un problema di peso (HARMONICITY_
            # WEIGHT_BOW_MULT rimosso a parita' non ha cambiato nulla). Funzione
            # lasciata con db_space=False sempre (mai richiamata a True) invece di
            # rimossa - stesso trattamento delle altre idee ritirate in questo file.
            _hw_mult = ({"harmonicity": HARMONICITY_WEIGHT_BOW_MULT, "noisiness": HARMONICITY_WEIGHT_BOW_MULT}
                        if exciter_type == "bow" else None)
            terms = _weighted_loss_terms(achieved, target, weight_mult=_hw_mult)
            loss_acc = loss_acc + _aggregate_loss(terms)   # minimax smussato (LogSumExp): aderenza uniforme, stabile
            if exciter_type in DECAY_SLOPE_TYPES and "decay_slope" in target:
                # Fix 2026-09-09 (isolamento q_raw, vedi nota su DECAY_SLOPE_TYPES
                # e decay_slope_db_per_band in synth_torch_before.py): termine
                # ADDITIVO esterno a _weighted_loss_terms/_aggregate_loss - stesso
                # pattern di gain_profile_loss/freq_drift_loss, non un altro
                # descrittore dentro achieved (evita il collo di bottiglia
                # tanh/LogSumExp gia' diagnosticato li').
                achieved_slope = decay_slope_db_per_band(y, sr)
                ds_err = ((achieved_slope - target["decay_slope"]) / DECAY_SLOPE_NORM) ** 2
                loss_acc = loss_acc + DECAY_SLOPE_WEIGHT * ds_err.mean()
        loss = loss_acc / noise_samples
        warmup_it = int(iters * GAIN_FREQ_WARMUP_FRAC)
        if (modal.gain_profile_init is not None and it >= warmup_it
                and (it - warmup_it) % GAIN_PROFILE_REFRESH_EVERY == 0):
            # a fine warmup Q (e per i non-feedback anche freq_raw) puo' essere
            # gia' derivato dal valore di warm start (e' libero durante il
            # warmup, solo gain_raw/freq_raw restano congelati - vedi sotto):
            # il profilo calcolato in warm_start con la Q PRE-warmup e' ora
            # stale proprio nel momento in cui _gain_profile_loss inizia a
            # mordere (gain_raw si sblocca qui). Ricalcolarlo con Q/freq
            # CORRENTI mantiene l'ancora coerente con lo stato reale del
            # banco invece di tirare gain_raw verso un'assunzione superata
            # (causa diagnosticata della regressione blow/shaker dopo
            # l'estensione del gain-fix a tutti gli exciter type). Ripetuto
            # ogni GAIN_PROFILE_REFRESH_EVERY iterazioni (non solo a fine
            # warmup, one-shot): per is_feedback il profilo dipende anche da
            # tone_amp_exponent (p dell'exciter), che puo' continuare a
            # muoversi nella seconda meta' del training - un refresh singolo
            # torna stale se p si sposta dopo quel punto. Per pluck questo
            # vale ANCHE per il comb fisico (_pluck_comb_profile): un'ancora
            # scelta una volta sola sul reach di inizio training (prima
            # versione) resta corretta come FORMA (posizione/durezza) ma
            # sbagliata in SCALA non appena Q si sposta - misurato: e_mid/
            # e_high ribaltati nella direzione opposta rispetto a prima del
            # comb. Ricalcolare qui con Q/freq correnti la rende dinamica
            # come per tutti gli altri exciter, invece di congelarla.
            with torch.no_grad():
                q_now = torch.clamp(F.softplus(modal.q_raw) + 0.4, max=Q_MAX)
                freqs_now = torch.exp(modal.freq_raw).clamp(max=NYQUIST_MARGIN * modal.sr)
                input_amp_now = None
                if is_feedback and tone_amp_exponent is not None:
                    p_now = float(exciter.params()["p"].detach())
                    k_idx = torch.arange(1, modal.n_modes + 1, dtype=torch.float32)
                    input_amp_now = 1.0 / (k_idx ** p_now)
                if exciter_type == "pluck":
                    ratios_now = freqs_now / f0
                    # q_adjusted ignorata qui: il refresh periodico aggiorna solo
                    # l'ancora di gain_profile_init, Q resta un parametro libero
                    # durante il training (nessun freq_drift-like anchor su q_raw).
                    bg, _q_adjusted_unused = _pluck_comb_profile(f0, ratios_now, q_now, target, modal.n_modes)
                else:
                    tau_now = F.softplus(modal.tau_raw) + 0.005 if modal.use_decay_envelope else None
                    bg = _band_gain_profile(freqs_now, q_now, target, modal.n_modes, modal.use_decay_envelope,
                                             input_amp=input_amp_now, tau=tau_now)
                if bg is not None:
                    modal.gain_profile_init = (bg / bg.sum()).detach().clone()
        if (modal.q_anchor_target is not None and exciter_type in DECAY_SLOPE_TYPES
                and it >= warmup_it and (it - warmup_it) % GAIN_PROFILE_REFRESH_EVERY == 0):
            # Fix 2026-09-09: refresh indipendente da quello di gain_profile_init sopra (non annidato
            # nel suo "is not None", cosi' resta valido anche per exciter senza quel profilo) - stessa
            # formula di warm_start, con freqs_now correnti invece che di warm start (i modi si muovono).
            with torch.no_grad():
                freqs_now_q = torch.exp(modal.freq_raw).clamp(max=NYQUIST_MARGIN * modal.sr)
                ratios_now_q = freqs_now_q / f0
                noisiness_now = float(target.get("noisiness", 0.3))
                q_floor_now = 3.0 if exciter_type == "shaker" else 0.5
                q_target_now = Q_MAX * (1.0 - 0.9 * noisiness_now) / (ratios_now_q ** 0.25)
                modal.q_anchor_target = q_target_now.clamp(min=q_floor_now).detach().clone()
        # (Diagnosi blow gia' fatta e rimossa: log strumentato it0-299 aveva
        # mostrato l'ancora ferma ma dominante in loss (~0.91-0.94 pesato) senza
        # mai vincere contro spread/harmonicity - vedi nota su gain_profile_init
        # in warm_start, causa e fix ora li'.)
        gp_loss = _gain_profile_loss(modal)
        if gp_loss is not None:
            gp_mult = (GAIN_PROFILE_WEIGHT_PLUCK_MULT if exciter_type == "pluck" else
                       GAIN_PROFILE_WEIGHT_STRIKE_MULT if exciter_type == "strike" else 1.0)
            gp_weight = GAIN_PROFILE_WEIGHT * gp_mult
            loss = loss + gp_weight * gp_loss
        fd_loss = _freq_drift_loss(modal) if not is_feedback else None
        if fd_loss is not None:
            loss = loss + FREQ_DRIFT_WEIGHT * fd_loss
        q_anchor_loss = _q_anchor_loss(modal) if exciter_type in DECAY_SLOPE_TYPES else None
        if q_anchor_loss is not None:
            loss = loss + Q_ANCHOR_WEIGHT * q_anchor_loss
        am_loss = _active_modes_loss(modal, MIN_ACTIVE_MODES) if exciter_type == "noise" else None
        if am_loss is not None:
            loss = loss + ACTIVE_MODES_WEIGHT * am_loss
        mr_loss = _mode_repulsion_loss(modal) if exciter_type in ("strike", "pluck") else None
        if mr_loss is not None:
            loss = loss + MODE_REPULSION_WEIGHT * mr_loss
        loss.backward()
        _q_grad_dbg = None
        if exciter_type == "strike" and log and (it % max(1, iters // 5) == 0 or it == iters - 1):
            # diagnostica gradiente Q (2026-09-09): su Va-legno_batt-C5 q_min/q_max e la loss
            # di harmonicity restano IDENTICI per l'intera run (vedi nota sopra) - verifica se
            # e' un gradiente davvero azzerato su q_raw o solo un update troppo piccolo per
            # essere visibile a 1 cifra decimale nel log esistente.
            g = modal.q_raw.grad
            _q_grad_dbg = "None" if g is None else f"norm={float(g.norm()):.3e} max_abs={float(g.abs().max()):.3e}"
        if it < int(iters * GAIN_FREQ_WARMUP_FRAC):
            modal.gain_raw.grad = None
            if not is_feedback:
                modal.freq_raw.grad = None
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        if it < warmup_it and modal.use_decay_envelope:
            # FIX: durante il warmup gain_raw resta ESPLICITAMENTE congelato
            # (grad azzerato sopra), ma Q_raw e' libero e si sposta (voluto,
            # vedi nota su Q_MAX/LR dedicato: senza, il suo gradiente troppo
            # debole non usciva mai dal warm start) - la calibrazione
            # gain<->energia (reach=Q/f) pero' resta quella della Q INIZIALE
            # per tutto il warmup, non della Q corrente: la ripartizione
            # e_low/e_mid/e_high accuratamente calcolata in warm_start si
            # rompe silenziosamente proprio nei primi 50 step, prima ancora
            # che gain_raw possa reagire (il suo gradiente e' zero, non puo'
            # correggersi da solo). Ri-derivare gain_raw qui (no_grad, non un
            # anchor morbido come gain_profile_init sotto - quello e' un
            # peso in loss che comunque non tocca gain_raw finche' e' congelato)
            # con la Q appena aggiornata mantiene la ripartizione di banda
            # coerente per tutta la durata del warmup.
            with torch.no_grad():
                q_now = torch.clamp(F.softplus(modal.q_raw) + 0.4, max=Q_MAX)
                freqs_now = torch.exp(modal.freq_raw).clamp(max=NYQUIST_MARGIN * modal.sr)
                if exciter_type == "pluck":
                    ratios_now = freqs_now / f0
                    bg, _ = _pluck_comb_profile(f0, ratios_now, q_now, target, modal.n_modes)
                else:
                    tau_now2 = F.softplus(modal.tau_raw) + 0.005 if modal.use_decay_envelope else None
                    bg = _band_gain_profile(freqs_now, q_now, target, modal.n_modes, modal.use_decay_envelope,
                                             tau=tau_now2)
                if bg is not None:
                    modal.gain_raw.copy_(_inv_softplus_t(bg.clamp(min=1e-3) / bg.max()
                                                           * F.softplus(modal.gain_raw).max()))
        loss_val = float(loss.detach())
        if log and (it % max(1, iters // 5) == 0 or it == iters - 1):
            breakdown = _named_loss_terms(achieved, target)
            bstr = "  ".join(f"{k}={v:.4f}" for k, v in breakdown.items())
            gp_str = f"  gain_profile={float(gp_loss.detach()):.4f}" if gp_loss is not None else ""
            fd_str = f"  freq_drift={float(fd_loss.detach()):.4f}" if fd_loss is not None else ""
            am_str = f"  active_modes={float(am_loss.detach()):.4f}" if am_loss is not None else ""
            if exciter_type == "bow":
                # DEBUG 2026-09-11 (fix LR*2 su gain_raw falsificato: harmonicity
                # achieved invariata su 3/3 file dopo il fix, gain_raw si muove
                # eccome - il collo di bottiglia non e' li'. Misura diretta di
                # RMS tono (dopo il banco, PRIMA del limiter) vs RMS rumore
                # (bypassa il banco per costruzione) per capire se il gain-boost
                # sposta davvero il rapporto nel segnale grezzo o se roughness
                # scala insieme/di piu'.
                with torch.no_grad():
                    _gain_sum_dbg = float(F.softplus(modal.gain_raw).sum())
                    _gain_grad_dbg = "None" if modal.gain_raw.grad is None else f"norm={float(modal.gain_raw.grad.norm()):.3e}"
                    _tone_rms_dbg = float(modal(tone).std()) if 'tone' in dir() else float("nan")
                    _noise_rms_dbg = float(noise.std()) if 'noise' in dir() else float("nan")
                    _p_dbg2 = exciter.params()
                    _rough_dbg = float(_p_dbg2["roughness"])
                    _hfr_dbg = float(_p_dbg2["hf_roughness"])
                am_str += f"  gain_sum={_gain_sum_dbg:.4f}  gain_grad=({_gain_grad_dbg})  tone_rms={_tone_rms_dbg:.4f}  noise_rms={_noise_rms_dbg:.4f}  roughness={_rough_dbg:.4f}  hf_roughness={_hfr_dbg:.4f}"
            if exciter_type == "strike":
                with torch.no_grad():
                    _amp_dbg = F.softplus(modal.gain_raw)
                    _p_dbg = _amp_dbg / (_amp_dbg.sum() + EPS)
                    _eff_n_dbg = float(torch.exp(-(_p_dbg * torch.log(_p_dbg + EPS)).sum()))
                    _q_dbg = torch.clamp(F.softplus(modal.q_raw) + 0.4, max=Q_MAX)
                am_str += f"  eff_n={_eff_n_dbg:.2f}  q_min={float(_q_dbg.min()):.1f}  q_max={float(_q_dbg.max()):.1f}"
                if _q_grad_dbg is not None:
                    am_str += f"  q_grad=({_q_grad_dbg})"
                _flat_dbg = float(achieved["flatness"].detach()) if "flatness" in achieved else float("nan")
                _db_dbg = 10.0 * np.log10(_flat_dbg + 1e-12)
                _h_dbg = float(achieved["harmonicity"].detach()) if "harmonicity" in achieved else float("nan")
                _sat = "SAT" if (_db_dbg <= -48.0 or _db_dbg >= -1.0) else "free"
                am_str += f"  db={_db_dbg:.1f}({_sat})  h_ach={_h_dbg:.4f}"
                if q_anchor_loss is not None:
                    am_str += f"  q_anchor={float(q_anchor_loss.detach()):.4f}"
                    am_str += f"  q_tgt=({float(modal.q_anchor_target.min()):.1f},{float(modal.q_anchor_target.max()):.1f})"
                if it == 0:
                    # Diagnostica limiter (2026-09-09): _normalize_peak+_limiter
                    # possono generare distorsione a banda larga se il picco pre-scala
                    # e' molto alto (commento in _normalize_peak: misurato fino a 17x) -
                    # verifica se e' questo a rendere lo spettro "sempre rumoroso" a
                    # prescindere da Q, non solo il percorso descrittori/loss.
                    with torch.no_grad():
                        _raw = modal(exciter(n))
                        _raw_peak = float(_raw.abs().max())
                        _clip_frac = float((y.abs() > LIMITER_DRIVE - 1e-6).float().mean())
                    am_str += f"  raw_peak={_raw_peak:.2f}  clip_frac={_clip_frac:.3f}"
            mr_str = f"  mode_repulsion={float(mr_loss.detach()):.4f}" if mr_loss is not None else ""
            log(f"  [{exciter_type:7s}] it {it:3d}  loss={loss_val:.4f}  | {bstr}{gp_str}{fd_str}{am_str}{mr_str}")

    achieved_f = {k: float(v.detach()) for k, v in achieved.items()}
    return exciter, modal, achieved_f, loss_val


def train_agent_best(exciter_type, target, restarts=3, log=None, seed=None, on_restart=None, **kwargs):
    """Esegue train_agent piu' volte e tiene il risultato con loss finale
    piu' bassa, invece di accettare il primo tentativo anche se e' finito
    su una traiettoria sfortunata. Costo: puramente piu' calcolo (nessun
    cambio a modello/loss).

    seed: se dato, ri-semina torch.manual_seed(seed+r) prima di OGNI
    restart (r=0..restarts-1) - senza, l'unica fonte di varianza tra
    restart era il seguito implicito dello stream RNG globale di torch tra
    una chiamata e l'altra (deterministico dato il seed iniziale, ma non
    esplicito ne' indipendente dal numero di draw random di ogni restart).
    Con seed esplicito ogni restart riparte da un punto RNG indipendente e
    riproducibile - vedi single_train.py per la varianza tra restart alta
    osservata (punto 3 della review), che questo rende piu' facile da
    quantificare su tanti restart.

    on_restart: se dato, callback(r, loss_val) dopo ogni restart - usato
    per raccogliere le loss di tutti i restart (non solo il migliore) per
    la diagnosi di spread, senza cambiare il valore di ritorno di questa
    funzione (restano compatibili tutte le chiamate esistenti)."""
    best = None
    for r in range(restarts):
        if seed is not None:
            torch.manual_seed(seed + r)
        result = train_agent(exciter_type, target, log=(log if r == 0 else None), **kwargs)
        loss_val = result[-1]
        if log and restarts > 1:
            log(f"  [{exciter_type:7s}] restart {r}: loss={loss_val:.4f}")
        if on_restart is not None:
            on_restart(r, loss_val)
        if best is None or loss_val < best[-1]:
            best = result
    return best


_INTERVENTION_HINTS = {
    "harmonicity": "basso -> possibile coda silenziosa che gonfia noisiness (flatness non pesata per energia, vedi DECAY_ENV_FLOOR e punto 1 review), o modi troppo larghi/Q bassa. ALTO rispetto al target su materiale a ring lungo (Q alto, es. tam-tam/timpani a bordo pelle) -> il ring e' quasi sempre puramente tonale per costruzione (nessuna eccitazione oltre ~21ms), vedi tail_noise_raw/ring_env in ModalBank.forward (2026-09-08): se resta alto a training concluso, tail_noise_raw non e' salito abbastanza - guarda se e_low/e_mid/e_high lo stanno tenendo giu'.",
    "noisiness": "alto -> stessa causa di harmonicity basso, o rumore additivo dell'exciter troppo energico.",
    "centroid": "fuori target -> ripartizione gain tra modi bassi/alti (gain_profile_init, reach=Q/f in _band_gain_profile) o drift di freq_raw.",
    "rolloff": "fuori target -> energia in coda spettrale: Q dei modi alti, GAIN_PROFILE_WEIGHT.",
    "e_low": "fuori target -> gain_profile_init sui modi bassi.",
    "e_mid": "fuori target -> gain_profile_init sui modi medi.",
    "e_high": "fuori target -> gain_profile_init sui modi alti, o Q troppo alta/bassa su quei modi.",
    "flux": "fuori target -> forma temporale dell'inviluppo (tau_raw) o attacco exciter - vedi FLUX_NORM_FLOOR per il limite fisico su pluck (punto 5 review).",
    "spread": "fuori target -> concentrazione energia tra modi: mode_repulsion, gain_profile_init.",
    "t_centroid": "fuori target -> tau_raw (decay envelope) troppo lento/veloce.",
    "inharmonicity": "fuori target -> drift di freq_raw vs rapporti armonici attesi (_freq_drift_loss, MEMBRANE_RATIOS/STRING_INHARM_B).",
    "roughness": "fuori target -> modi troppo vicini in frequenza: mode_repulsion, MODE_MIN_SPACING_OCT.",
}


def _descriptor_gaps(achieved, target):
    """Come _named_loss_terms ma su valori gia' float (achieved_f, dopo
    .detach() a fine train_agent) invece che tensori live - serve per la
    diagnosi POST-training (single_train.py, __main__), dove achieved non
    e' piu' un tensore con grad."""
    out = {}
    for k, v in target.items():
        if k not in achieved:
            continue
        v = float(v)
        a = float(achieved[k])
        w = LOSS_WEIGHTS.get(k, 1.0)
        if k == "flux":
            out[k] = w * ((a - v) / (v + FLUX_NORM_FLOOR)) ** 2
        elif k in RATIO_KEYS:
            out[k] = w * float(np.log2((a + EPS) / (v + EPS))) ** 2
        else:
            out[k] = w * (a - v) ** 2
    return out


def _diagnose_run(exciter_type, achieved, target, modal, f0, restart_losses=None, log=print):
    """Diagnostica di fine training: quali descrittori pesano di piu' sulla
    loss finale (con suggerimento su quale leva del codice guardare, vedi
    _INTERVENTION_HINTS), stato dei modi con il picco CORRETTO per tipo di
    filtro (fix punto 2 review: pk=gain*Q solo per use_legacy_filter,
    altrimenti pk=gain perche' il filtro nuovo ha picco=1 a risonanza), e -
    se restart_losses e' passato (vedi on_restart di train_agent_best) - lo
    spread tra restart, per non scambiare rumore di inizializzazione per un
    effetto reale (punto 3 review)."""
    gaps = _descriptor_gaps(achieved, target)
    ranked = sorted(gaps.items(), key=lambda kv: -kv[1])
    log("  --- diagnosi ---")
    log("  descrittori piu' pesanti sulla loss (contributo, achieved vs target):")
    for k, val in ranked[:4]:
        a, t = float(achieved[k]), float(target[k])
        hint = _INTERVENTION_HINTS.get(k, "")
        log(f"    {k}: contrib={val:.4f}  achieved={a:.3f} target={t:.3f}  {hint}")

    with torch.no_grad():
        kk = torch.arange(1, modal.n_modes + 1, dtype=torch.float32)
        freq_init = f0 * kk
        freq_now = torch.exp(modal.freq_raw)
        gain_now = F.softplus(modal.gain_raw)
        q_now = torch.clamp(F.softplus(modal.q_raw) + 0.4, max=Q_MAX)
        peak = gain_now * q_now if modal.use_legacy_filter else gain_now
        active = gain_now > 0.05
        drift = ((freq_now[:len(freq_init)] - freq_init).abs() / freq_init).mean() if len(freq_init) else torch.tensor(0.0)
        order = torch.argsort(freq_now)
        rows = "  ".join(f"{freq_now[i]:.0f}Hz(g={gain_now[i]:.2f},Q={q_now[i]:.1f},pk={peak[i]:.2f})"
                          for i in order)
    log(f"  f0={f0:.1f}Hz  modi attivi(gain>0.05)={int(active.sum())}/{modal.n_modes}  "
        f"drift medio freq vs init={float(drift)*100:.1f}%")
    log(f"  modi ordinati per freq (pk corretto per use_legacy_filter={modal.use_legacy_filter}): {rows}")

    if restart_losses:
        lo, hi = min(restart_losses), max(restart_losses)
        spread = (hi - lo) / max(lo, 1e-6)
        flag = "  <-- ATTENZIONE: spread alto, un singolo confronto prima/dopo non e' conclusivo (punto 3 review)" if spread > 0.5 else ""
        losses_str = ", ".join(f"{l:.4f}" for l in restart_losses)
        log(f"  restart losses=[{losses_str}]  spread={spread*100:.0f}%{flag}")


def _band_isolation_probe(exciter_type, exciter, modal, f0, seconds, sr, target=None):
    """Isola i modi 'alti' (freq>=3000Hz) da quelli 'bassi/medi' e li rende
    da soli: se anche isolato un modo alto produce poco e_high, e' un
    limite del modo stesso (es. ringdown troppo corto rispetto al gain/Q);
    se isolato va bene ma nel mix va male, e' mascheramento/interferenza
    con gli altri modi nella somma in frequenza.

    target: passato solo per coerenza col tono usato in training (blow usa
    _tone_amp_profile, non 1/k^p - vedi _feedback_synthesize); None per
    bow, comportamento invariato."""
    is_feedback = exciter_type in FEEDBACK_TYPES
    n = int(round(seconds * sr))
    with torch.no_grad():
        if is_feedback:
            bt = target if (exciter_type == "blow" and target is not None) else None
            tone, noise, env_mod_tone = _feedback_synthesize(exciter, f0, n, sr, n_harm=modal.n_modes, band_target=bt)
        else:
            tone, noise, env_mod_tone = exciter(n), 0.0, None
        freq_now = torch.exp(modal.freq_raw)
        high_mask = freq_now >= 3000.0

        def render_masked(mask):
            saved = modal.gain_raw.data.clone()
            modal.gain_raw.data[~mask] = -20.0   # softplus(-20)~0: silenzia i modi esclusi
            y_tone = modal(tone)
            if env_mod_tone is not None:
                y_tone = y_tone * env_mod_tone
            y = _limiter(y_tone + noise)
            d = descriptor_vector_torch(y, sr)
            modal.gain_raw.data.copy_(saved)
            return {k: float(v) for k, v in d.items()}

        d_high = render_masked(high_mask) if bool(high_mask.any()) else None
        d_low = render_masked(~high_mask) if bool((~high_mask).any()) else None
    if d_high:
        print(f"  probe modi alti soli ({int(high_mask.sum())} modi >=3000Hz): "
              f"e_low={d_high['e_low']:.3f} e_mid={d_high['e_mid']:.3f} e_high={d_high['e_high']:.3f}")
    if d_low:
        print(f"  probe modi bassi/medi soli ({int((~high_mask).sum())} modi <3000Hz): "
              f"e_low={d_low['e_low']:.3f} e_mid={d_low['e_mid']:.3f} e_high={d_low['e_high']:.3f}")


def render(exciter, modal, seconds, sr=SR_DEFAULT, target=None):
    """target: SOLO blow, opzionale - se non passato (uso tipico fuori dal
    training, dove il target dei descrittori non esiste piu') ricade su
    1/k^p (vedi _feedback_synthesize/_tone_amp_profile): il preset esportato
    dovra' portare con se' e_low/e_mid/e_high per riprodurre esattamente il
    tono allenato, non ancora implementato in questo export."""
    with torch.no_grad():
        n = int(round(seconds * sr))
        if isinstance(exciter, (BowExciter, BlowExciter)):
            f0 = float(torch.exp(modal.freq_raw[0]))
            bt = target if (isinstance(exciter, BlowExciter) and target is not None) else None
            tone, noise, env_mod_tone = _feedback_synthesize(exciter, f0, n, sr, n_harm=modal.n_modes, band_target=bt)
            y_tone = modal(tone)
            if env_mod_tone is not None:
                y_tone = y_tone * env_mod_tone
            y = _limiter(y_tone + noise)
        else:
            y = _limiter(modal(exciter(n)))
    return y.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# TARGET PSEUDO-RANDOM (solo descrittori, nessun file audio) + demo
# ---------------------------------------------------------------------------
def random_target(rng, f0_range=(80, 1000), n_modes=12, sr=SR_DEFAULT, exciter_type=None):
    """centroid/rolloff DERIVATI dalle stesse band_energy (non campionati
    indipendenti): prima il bug faceva capitare target come rolloff=865Hz
    (90% energia sotto 865Hz) insieme a e_high=0.629 (63% sopra 3000Hz) -
    matematicamente incompatibili, irraggiungibili per costruzione, non per
    limite del modello.

    noisiness: floor a 0.25, non 0.0 - verificato empiricamente (sweep
    durata eccitazione x Q) che un'eccitazione stocastica filtrata da un
    banco modale in questo range di Q/durata non scende sotto ~0.2-0.35 di
    noisiness con questa metrica: e' il pavimento fisico di 'rumore
    filtrato' vs 'tono deterministico', non un difetto di ottimizzazione.

    noise (floor alzato a 0.7, non 0.25): la metrica harmonicity/noisiness
    (synth_torch_before.descriptor_vector_torch) e' flatness spettrale
    MEDIATA per-frame SENZA pesare per energia (a differenza di centroid/
    bande, gia' energy-pooled - vedi fix 2026-08-29 li'). Per un'eccitazione
    impulsiva (strike/pluck/shaker) la maggioranza dei frame del buffer e'
    ringdown pulito post-transiente (poche righe spettrali, flatness bassa)
    che domina la media e rende raggiungibile bassa noisiness anche con un
    attacco rumoroso. "noise" e' continuo: OGNI frame ha lo stesso pavimento
    a banda larga che si rialimenta di continuo, senza una coda pulita che
    trascini la media - non ha lo stesso vantaggio strutturale, e in pratica
    resta ben sopra il floor generico anche quando il gradiente spinge verso
    target di noisiness moderata (osservato: target noisiness=0.42,
    convergenza a ~0.80 nonostante 12/12 modi attivi e training completo).
    Floor a 0.7 riflette questo limite invece di campionare target
    strutturalmente irraggiungibili per questo exciter type.

    bow/blow (is_feedback): banco a PETTINE DISCRETO (n_harm righe a k*f0),
    non banda continua - il vecchio schema (bande + media geometrica,
    condiviso con gli altri exciter) poteva chiedere energia in bande dove
    cade una sola armonica o nessuna, con centroid derivato che arrivava a
    cadere SOTTO f0 stesso: impossibile per un pettine con energia solo a
    f>=f0 (osservato: bow con target centroid<f0, collassato su 1/8 modi
    attivi, Q al tetto, restart deterministici - vedi analisi). Qui il
    target e' costruito DIRETTAMENTE dalle armoniche raggiungibili, quindi
    sempre soddisfacibile per costruzione."""
    noisiness_lo = 0.7 if exciter_type == "noise" else 0.25
    noisiness = float(rng.uniform(noisiness_lo, 1.0))
    f0 = float(rng.uniform(*f0_range))
    is_feedback = exciter_type in FEEDBACK_TYPES
    # flux/spread: comuni a tutte le famiglie (descriptor_vector_torch li
    # calcola sempre, vedi synth_torch_before.py), campionati dalla
    # distribuzione reale per famiglia quando sample_calibration.json e'
    # disponibile, altrimenti range euristico neutro - vedi _sample_calibrated.
    flux_val = _sample_calibrated(rng, exciter_type, "flux", (0.0, 0.4))
    spread_val = _sample_calibrated(rng, exciter_type, "spread",
                                     (max(f0 * 0.3, 50.0), max(f0 * 6.0, 6000.0)))
    # tetto 6000 (non 3000): calibrazione reale shaker (sample_calibration.json)
    # mean=2604 std=941 max osservato=4899 - con tetto 3000 il campione
    # clippava sistematicamente li' (achieved log: spread target=3000.0
    # esatto, non un valore genuino), perdendo la vera dispersione mean+-std.

    if is_feedback:
        n_harm = _n_harm_feedback(f0)
        k = np.arange(1, n_harm + 1)
        harm_freqs = f0 * k
        max_reach = min(harm_freqs[-1], sr / 2.0 * 0.9)
        band_edges = [20.0, 300.0, 3000.0, max_reach]
        band_of = np.clip(np.searchsorted(band_edges, harm_freqs, side="right") - 1, 0, 2)
        reachable = np.array([bool(np.any(band_of == b)) for b in range(3)])
        weights = np.where(reachable, 1.0, 0.02)
        e = rng.dirichlet(weights)   # e_low/e_mid/e_high normalizzate a 1, pesate per raggiungibilita'
        decay = 1.0 / k.astype(float)   # inviluppo 1/k tipico (Helmholtz), coerente col p di BowExciter
        harm_e = np.zeros(n_harm)
        for b in range(3):
            mask = band_of == b
            if mask.any():
                w = decay[mask]
                harm_e[mask] = e[b] * w / w.sum()
        harm_e = harm_e / harm_e.sum()
        centroid = float((harm_e * harm_freqs).sum())
        order = np.argsort(harm_freqs)
        cum = np.cumsum(harm_e[order])
        idx90 = min(int(np.searchsorted(cum, 0.9)), n_harm - 1)
        rolloff = float(harm_freqs[order][idx90])
        return {
            "harmonicity": 1.0 - noisiness, "noisiness": noisiness,
            "centroid": centroid, "rolloff": rolloff,
            "e_low": float(e[0]), "e_mid": float(e[1]), "e_high": float(e[2]),
            "flux": flux_val, "spread": spread_val,
        }, f0

    max_reach = min(f0 * n_modes, sr / 2.0 * 0.9)

    # i modi sono f0, 2*f0, ..., n_modes*f0 -> una banda che non si
    # sovrappone a [f0, max_reach] non e' raggiungibile da NESSUN modo
    # (es. f0=400Hz: nessun modo puo' mai cadere in 20-300Hz). Peso quasi
    # nullo (non zero secco, per non degenerare il Dirichlet) alle bande
    # non raggiungibili, invece di chiedere energia dove non puo' arrivare.
    band_edges = [20.0, 300.0, 3000.0, max_reach]
    band_edges = [min(b, max_reach) for b in band_edges]
    band_mid = [(band_edges[i] * band_edges[i + 1]) ** 0.5 for i in range(3)]  # media geometrica
    weights = []
    for i in range(3):
        lo, hi = band_edges[i], band_edges[i + 1]
        overlap = max(0.0, min(hi, max_reach) - max(lo, f0))
        weights.append(1.0 if overlap > 0 else 0.02)
    e = rng.dirichlet(weights)   # e_low/e_mid/e_high normalizzate a 1, pesate per raggiungibilita'
    centroid = float(sum(ei * mi for ei, mi in zip(e, band_mid)))
    # spread_val (sopra) e' campionato INDIPENDENTEMENTE dalla ripartizione
    # e_low/e_mid/e_high - stessa classe di bug gia' risolta per centroid/
    # rolloff (vedi nota li' sopra sui target incompatibili): puo' chiedere
    # insieme uno spread molto stretto E una ripartizione di energia su bande
    # lontane, due cose incompatibili per costruzione (osservato: pluck, spread
    # target=260Hz - il minimo del range - con e_mid=0.535/e_high=0.465, energia
    # su due bande separate da un'ottava - la loss 'spread' non convergeva mai
    # in training, dominando la LSE per tutte le 300 iterazioni indipendentemente
    # da qualsiasi intervento su gain/freq/pesi). Il piu' piccolo spread
    # fisicamente compatibile con la ripartizione e' la sua deviazione RMS pesata
    # per energia attorno al centroid.
    implied_spread = float(np.sqrt(sum(ei * (mi - centroid) ** 2 for ei, mi in zip(e, band_mid))))
    spread_val = max(spread_val, implied_spread)

    cum = np.cumsum(e)
    b_idx = int(min(np.searchsorted(cum, 0.9), 2))
    lo, hi = band_edges[b_idx], band_edges[b_idx + 1]
    frac_prev = cum[b_idx - 1] if b_idx > 0 else 0.0
    frac_in = (0.9 - frac_prev) / max(e[b_idx], 1e-6)
    rolloff = lo + float(np.clip(frac_in, 0.0, 1.0)) * (hi - lo)

    out = {
        "harmonicity": 1.0 - noisiness, "noisiness": noisiness,
        "centroid": centroid, "rolloff": rolloff,
        "e_low": float(e[0]), "e_mid": float(e[1]), "e_high": float(e[2]),
        "flux": flux_val, "spread": spread_val,
    }
    if exciter_type in ("strike", "pluck", "shaker", "chaotic"):
        # tau_raw (ModalBank, decadimento per-modo indipendente da Q, punto 8)
        # e' finora inerte (init 1000s -> exp(-t/tau)~1, no-op) per strike/
        # pluck. exp(-t/tau) e' un fattore <=1 monotono decrescente: puo'
        # SOLO accorciare il decadimento rispetto a quello naturale gia'
        # dato da Q, mai allungarlo. Target campionato sotto il t_centroid
        # osservato finora a tau inerte (~0.03-0.06s) per dare un compito
        # reale a tau; se il target e' irraggiungibile (piu' lungo del
        # decadimento naturale di Q), tau resta al suo valore inerte senza
        # conflitto di gradiente. shaker ora ha tau_raw anche lui (vedi
        # train_agent, use_decay_env) - range piu' ampio, coerente col
        # decadimento naturalmente piu' lungo (init exciter 150ms).
        t_lo, t_hi = ((0.03, 0.25) if exciter_type == "shaker" else
                      (0.02, 0.35) if exciter_type == "chaotic" else (0.015, 0.065))
        # chaotic: range piu' ampio di shaker - il materiale copre sia il
        # graffio/colpo secco (tam-tam) sia il flusso sostenuto (death
        # whistle/scream/multifonico), estremi temporali entrambi plausibili.
        out["t_centroid"] = float(rng.uniform(t_lo, t_hi))
    if exciter_type in ("strike", "chaotic"):
        # inharmonicity/roughness: strike (non piu' anche pluck - vedi nota
        # in train_agent, run reale ha mostrato pluck instabile con questi
        # due target extra) e ora chaotic, dove SONO l'obiettivo primario
        # (non un affinamento). Target per i proxy differenziabili dai
        # parametri del banco (vedi _inharmonicity_proxy/_roughness_proxy),
        # dove freq_raw resta libero di muoversi. NON bow/blow: la' freq_raw
        # resta congelato esatto sulle armoniche (is_feedback in train_agent),
        # un target di inarmonicita' sarebbe strutturalmente irraggiungibile.
        # strike: calibrazione da AnalyzerV3 su audio reale (extra_summary,
        # sample_calibration.json) se disponibile. chaotic: range euristico
        # piu' ampio (0.05-0.4, non 0-0.08/0-0.15) - materiale
        # intenzionalmente molto piu' inarmonico/rumoroso di uno strike
        # "normale", nessuna calibrazione reale ancora raccolta per questa
        # famiglia (_sample_calibrated ricade comunque sul fallback_range se
        # sample_calibration.json non ha una voce "chaotic").
        i_range = (0.0, 0.08) if exciter_type == "strike" else (0.05, 0.4)
        r_range = (0.0, 0.15) if exciter_type == "strike" else (0.05, 0.4)
        out["inharmonicity"] = _sample_calibrated(rng, exciter_type, "inharmonicity", i_range, extra=True)
        out["roughness"] = _sample_calibrated(rng, exciter_type, "roughness", r_range, extra=True)
    return out, f0


if __name__ == "__main__":
    # seed torch: senza, torch.randn() (rumore di eccitatori/noise_samples)
    # non e' riproducibile tra run -> confronti prima/dopo una modifica al
    # codice sono confusi dalla varianza random, non dalla modifica stessa
    # (osservato: stesso codice, run diversi, pluck passa da loss=2.07 a
    # loss=6.20 senza alcun cambio all'ottimizzazione tra i due run).
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for etype in EXCITERS:
        target, f0 = random_target(rng, exciter_type=etype)
        restart_losses = []
        exciter, modal, achieved, loss = train_agent_best(
            etype, target, f0=f0, restarts=3, log=print, seed=0,
            on_restart=lambda r, l: restart_losses.append(l))
        print(f"  target={ {k: round(v,3) for k,v in target.items()} }")
        print(f"  achieved={ {k: round(v,3) for k,v in achieved.items()} }")
        print(f"  loss finale={loss:.4f}")
        _diagnose_run(etype, achieved, target, modal, f0, restart_losses=restart_losses, log=print)
        _band_isolation_probe(etype, exciter, modal, f0, seconds=1.0, sr=SR_DEFAULT, target=target)
        print()
