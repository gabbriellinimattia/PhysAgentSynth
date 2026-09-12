"""
ClaudeSynth - Analizzatore v3: decomposizione Sines + Noise + Transients (SNT).

Fix rispetto a v2 (vedi note [A]-[K] per riferimento alle criticita'):
  [A] sample rate passato esplicitamente, mai da costante globale
  [B] ring buffer per lo streaming: overlap corretto indipendente dalla
      dimensione del chunk audio in arrivo
  [C] decay rate per parziale stimato con Theil-Sen (robusto agli outlier)
  [D] soglia di prominenza adattiva (MAD locale) + cutoff per energia
      cumulativa invece di saturare sempre a MAX_PARTIALS
  [E] partial linking con assegnazione ottima (Hungarian), non greedy
  [F] larghezza del lobo di sottrazione derivata dalla mainlobe reale
      della finestra, non fissa
  [G] transientness normalizzata adattivamente (z-score su storia flux)
  [H] range f0 configurabile, throttling opzionale
  [I] fallback f0 da parziali via harmonic template matching pesato
      (non piu' solo "parziale piu' basso")
  [J] smoothing (EMA) sui descrittori di livello 1 in uscita
  [K] lock inutilizzato rimosso

Fix diagnostica 2026-08-28 (vedi note [L]-[N]):
  [L] pick_peaks: cutoff per energia cumulativa ricalcolato su energia di
      lobo (non piu' punto-picco), altrimenti il denominatore (energia
      dell'intero spettro) non era mai raggiungibile e il cutoff era un
      no-op (selezionava fino a MAX_PARTIALS picchi, incluso rumore di
      floor a -170dB)
  [M] estimate_f0_from_partials: raffinamento locale (bounded) dei
      migliori candidati della griglia grossolana, per eliminare l'errore
      sistematico di sotto-ottava dovuto alla quantizzazione della griglia
  [N] sines.f0 usa ora [M] come sorgente primaria; autocorrelation_pitch
      si e' rivelata inaffidabile (errore di sotto-ottava anche su un
      tono puro a frequenza fissa) e resta solo per pitch_confidence
      diagnostico
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment, minimize_scalar
import queue
import time

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------
SAMPLE_RATE_DEFAULT = 44100
BLOCKSIZE = 1024
N_FFT = 2048
HOP_RATIO = 0.5
BANDS = [(20, 300), (300, 3000), (3000, 20000)]

EPS = 1e-10

# --- parametri partial tracking ---
PROMINENCE_DB = 8.0               # soglia di prominenza topografica (scipy), calibrata
                                   # empiricamente su materiale reale (conga/multi/screech):
                                   # una MAD globale dello spettro NON e' un buon proxy di
                                   # densita' locale dei picchi (confonde range dinamico
                                   # complessivo con nitidezza del singolo picco) e con
                                   # spettri a forte dinamica collassa la soglia fino a
                                   # filtrare quasi tutto. L'adattivita' reale e' demandata
                                   # al cutoff per energia cumulativa qui sotto.
MAX_PARTIALS = 80                # limite di sicurezza (raramente raggiunto ora)
ENERGY_CUTOFF_FRACTION = 0.97    # ferma la selezione picchi al 97% dell'energia cumulativa
PARTIAL_MATCH_HZ = 60.0          # tolleranza di linking frame-a-frame
PARTIAL_DEATH_FRAMES = 3         # frame senza match prima di uccidere la traccia
INHARMONIC_MAX_PARTIAL = 20      # armoniche considerate per stima inarmonicita'
DECAY_HISTORY_FRAMES = 8         # finestra storica per la stima del decay rate

# --- parametri onset/transient ---
ONSET_HISTORY = 8
ONSET_THRESHOLD_MULT = 1.5
TRANSIENT_Z_GAIN = 1.5           # guadagno della sigmoide sullo z-score del flux

# --- parametri pitch ---
F0_MIN_DEFAULT = 50.0
F0_MAX_DEFAULT = 2000.0
F0_CANDIDATES = 120              # risoluzione griglia grossolana per il fallback armonico
F0_REFINE_TOP_K = 6              # [M] candidati grossolani raffinati localmente
OCTAVE_TIE_TOL = 0.03            # [M] tolleranza di parita' per la correzione anti-sotto-ottava

# --- livello 1 strutturale: harmonicity/noisiness da flatness, non da peak-reconstruction ---
NOISINESS_DB_LOW = -48.0   # ~ riferimento tono puro (SFM_dB misurato: -49.4)
NOISINESS_DB_HIGH = -1.0   # ~ riferimento rumore bianco (SFM_dB misurato: -0.7)

# --- smoothing descrittori livello 1 ---
EMA_ALPHA = 0.3                  # 0 = nessuno smoothing, 1 = nessuna memoria


# ---------------------------------------------------------------------------
# UTILITY SPETTRALI DI BASE
# ---------------------------------------------------------------------------

def _magnitude_spectrum(frame, n_fft, window, sr):
    """[A] sample rate passato esplicitamente, mai globale."""
    windowed = frame * window
    spec = np.fft.rfft(windowed, n=n_fft)
    mag = np.abs(spec) + EPS
    phase = np.angle(spec)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs, mag, phase


def spectral_centroid(freqs, mag):
    return float(np.sum(freqs * mag) / np.sum(mag))


def spectral_spread(freqs, mag, centroid):
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))


def spectral_skewness(freqs, mag, centroid, spread):
    if spread < EPS:
        return 0.0
    return float(np.sum(((freqs - centroid) ** 3) * mag) / (np.sum(mag) * (spread ** 3)))


def spectral_kurtosis(freqs, mag, centroid, spread):
    if spread < EPS:
        return 0.0
    return float(np.sum(((freqs - centroid) ** 4) * mag) / (np.sum(mag) * (spread ** 4)) - 3.0)


def spectral_slope(freqs, mag):
    log_mag = np.log(mag)
    A = np.vstack([freqs, np.ones_like(freqs)]).T
    slope, _ = np.linalg.lstsq(A, log_mag, rcond=None)[0]
    return float(slope)


def spectral_rolloff(freqs, mag, threshold=0.90):
    cumulative = np.cumsum(mag)
    total = cumulative[-1]
    idx = np.searchsorted(cumulative, threshold * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def spectral_flatness(mag):
    log_mag = np.log(mag)
    geo_mean = np.exp(np.mean(log_mag))
    arith_mean = np.mean(mag)
    return float(geo_mean / arith_mean)


def noisiness_from_flatness(mag, low_db=NOISINESS_DB_LOW, high_db=NOISINESS_DB_HIGH):
    """[D+F, disaccoppiamento] harmonicity/noisiness calcolate dalla spectral
    flatness (Wiener entropy) invece che dal rapporto energia
    deterministica/residuo via ricostruzione a lobi gaussiani.

    Motivo: quel rapporto dipende in modo critico da quanti picchi vengono
    tracciati e da quanto sono larghi i lobi usati per ricostruirli. Con
    soglie permissive (necessarie per non perdere parziali deboli in
    'sines.partials', che deve restare generoso) i lobi finiscono per
    sovrapporsi su gran parte dello spettro anche su materiale rumoroso,
    facendo saturare harmonicity vicino a 1 indipendentemente dal contenuto
    reale (verificato empiricamente: ~0.95-0.98 su tre file di natura
    diversa con soglie tarate per un buon n_partials).

    La flatness e' calcolata sull'intero spettro, indipendente dal partial
    tracker: nessun accoppiamento tra le due funzioni. E' poi convertita in
    dB (SFM_dB = 10*log10(flatness)) e rimappata linearmente tra due
    riferimenti calibrati empiricamente: tono puro (~-49dB) e rumore
    bianco (~-0.7dB). Cattura la distinzione percettiva tonale/rumoroso a
    prescindere da quanti parziali sono tracciati o da quanto sono larghi."""
    flat = spectral_flatness(mag)
    db = 10.0 * np.log10(flat + EPS)
    noisiness = (db - low_db) / (high_db - low_db)
    return float(np.clip(noisiness, 0.0, 1.0))


def zero_crossing_rate(frame):
    signs = np.sign(frame)
    signs[signs == 0] = 1
    return float(np.mean(signs[:-1] != signs[1:]))


def band_energies(freqs, mag, bands):
    energies, crests = {}, {}
    mag2 = mag ** 2
    total = np.sum(mag2) + EPS
    for (lo, hi) in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_e = np.sum(mag2[mask])
        energies[f"{lo}-{hi}"] = float(band_e / total)
        if np.any(mask):
            band_mag = mag[mask]
            crests[f"{lo}-{hi}"] = float(np.max(band_mag) / (np.mean(band_mag) + EPS))
        else:
            crests[f"{lo}-{hi}"] = 0.0
    return energies, crests


def high_frequency_content(freqs, mag):
    return float(np.sum(freqs * (mag ** 2)) / (np.sum(mag ** 2) + EPS))


def spectral_flux(mag, prev_mag, only_increase=True):
    if prev_mag is None or len(prev_mag) != len(mag):
        return 0.0
    diff = mag - prev_mag
    if only_increase:
        diff[diff < 0] = 0
    return float(np.sum(diff) / (np.sum(mag) + EPS))


def roughness_estimate(freqs, mag, top_n=10):
    if len(mag) < top_n:
        top_n = len(mag)
    idx = np.argpartition(mag, -top_n)[-top_n:]
    idx = idx[np.argsort(-mag[idx])]
    peak_f = freqs[idx]
    peak_a = mag[idx]
    total_rough = 0.0
    for i in range(len(peak_f)):
        for j in range(i + 1, len(peak_f)):
            f1, f2 = peak_f[i], peak_f[j]
            a1, a2 = peak_a[i], peak_a[j]
            df = abs(f1 - f2)
            fmin = min(f1, f2) + EPS
            cbw = 1.72 * (fmin ** 0.65)
            x = df / (cbw + EPS)
            r = x * np.exp(-3.5 * x) * min(a1, a2)
            total_rough += r
    return float(total_rough / (np.sum(peak_a) + EPS))


def autocorrelation_pitch(frame, sr, fmin=F0_MIN_DEFAULT, fmax=F0_MAX_DEFAULT):
    """[H'] Normalizzazione NCCF (energia locale del segmento sovrapposto a
    ciascun lag), non solo su corr[0]/energia totale del frame. La
    normalizzazione globale distorceva sistematicamente la confidenza verso
    i lag corti: su frame poco/non periodici (es. materiale rumoroso o
    transiente) l'autocorrelazione non normalizzata decresce naturalmente
    con il lag, quindi il massimo nella finestra di ricerca cadeva spesso
    vicino a lag_min (=> f0 spurio vicino a fmax) con confidenza
    artificialmente alta. La normalizzazione per energia locale elimina
    questo bias strutturale verso le frequenze acute.

    [N] LIMITE NOTO (diagnosticato 2026-08-28, non risolto qui): su
    materiale fortemente periodico/armonico (incluso un tono puro a
    frequenza fissa) l'argmax puro sceglie spesso un lag multiplo del
    periodo vero (errore di sotto-ottava), perche' per un segnale
    periodico la NCCF e' massima (~1.0) a OGNI multiplo intero del
    periodo, non solo al primo, e la quantizzazione a campione intero del
    lag puo' rendere un multiplo piu' "pulito" del fondamentale stesso
    (verificato: tono puro 220Hz, finestra 1024 campioni @44100Hz ->
    lag vincente 401 campioni = ~2 periodi, f0 stimato 109.98Hz invece di
    220Hz, con confidence=1.0). Per questo AnalyzerV3.process_frame non
    usa piu' questa funzione come sorgente primaria di sines.f0 (vedi
    estimate_f0_from_partials + [N] sopra); resta qui per il campo
    pitch_confidence e per eventuali usi futuri su materiale non/poco
    armonico dove l'ambiguita' di sotto-ottava e' meno rilevante."""
    frame = frame.astype(np.float64) - np.mean(frame)
    n = len(frame)
    corr = np.correlate(frame, frame, mode="full")[n - 1:]

    lag_min = max(1, int(sr / fmax))
    lag_max = min(int(sr / fmin), n - 1)
    if lag_max <= lag_min:
        return 0.0, 0.0

    sq = frame ** 2
    cumsum = np.cumsum(sq)
    total_energy = cumsum[-1]

    lags = np.arange(lag_min, lag_max)
    # energia della porzione "head" [0 : n-tau] e "tail" [tau : n]
    energy_head = cumsum[n - lags - 1]
    energy_tail = total_energy - np.where(lags > 0, cumsum[lags - 1], 0.0)
    denom = np.sqrt(energy_head * energy_tail) + EPS

    nccf = corr[lag_min:lag_max] / denom
    if len(nccf) == 0:
        return 0.0, 0.0

    best = int(np.argmax(nccf))
    peak_idx = lags[best]
    confidence = float(max(0.0, min(1.0, nccf[best])))
    f0 = sr / peak_idx if peak_idx > 0 else 0.0
    return float(f0), confidence


def timbral_descriptor_set(freqs, mag, frame=None, prev_mag=None, bands=BANDS):
    centroid = spectral_centroid(freqs, mag)
    spread = spectral_spread(freqs, mag, centroid)
    skew = spectral_skewness(freqs, mag, centroid, spread)
    kurt = spectral_kurtosis(freqs, mag, centroid, spread)
    slope = spectral_slope(freqs, mag)
    rolloff = spectral_rolloff(freqs, mag)
    flatness = spectral_flatness(mag)
    hfc = high_frequency_content(freqs, mag)
    roughness = roughness_estimate(freqs, mag)
    energies, crests = band_energies(freqs, mag, bands)
    d = {
        "centroid": centroid, "spread": spread, "skewness": skew,
        "kurtosis": kurt, "slope": slope, "rolloff": rolloff,
        "flatness": flatness, "hfc": hfc, "roughness": roughness,
        "band_energy": energies, "band_crest": crests,
    }
    if frame is not None:
        d["zcr"] = zero_crossing_rate(frame)
    if prev_mag is not None:
        d["flux"] = spectral_flux(mag.copy(), prev_mag)
    return d


# ---------------------------------------------------------------------------
# PEAK PICKING (adattivo) E PARTIAL TRACKING (assegnazione ottima)
# ---------------------------------------------------------------------------

def pick_peaks(freqs, mag, mag_db,
               prominence_db=PROMINENCE_DB,
               max_peaks=MAX_PARTIALS,
               energy_cutoff=ENERGY_CUTOFF_FRACTION,
               total_energy=None,
               mainlobe_width_hz=None):
    """[D] Prominenza topografica (scipy.find_peaks), molto piu' robusta del
    confronto locale ai vicini a +/-2 bin usato in v2 (che generava falsi
    positivi su spettri densi/rumorosi). L'adattivita' al contenuto del
    segnale e' demandata al cutoff per energia cumulativa, calcolato
    rispetto all'energia TOTALE dello spettro (total_energy, passato dal
    chiamante) e non solo rispetto alla somma dei picchi gia' selezionati:
    con normalizzazione solo sui picchi, pochi picchi forti "chiudevano"
    artificialmente la selezione anche su materiale a energia distribuita
    (rumoroso), producendo harmonicity falsamente alta. Materiale tonale
    (energia concentrata in pochi picchi) seleziona pochi parziali;
    materiale rumoroso (energia distribuita su tutto lo spettro) ne
    richiede molti di piu' per raggiungere la stessa frazione di energia
    totale, fino al tetto di sicurezza MAX_PARTIALS.

    [L] FIX energia di picco: l'energia di ciascun picco non e' piu' il
    solo valore puntuale del bin (che sottostima l'energia reale di un
    parziale, spalmata dalla finestra su piu' bin: un tono pulito a 8
    armoniche copriva solo il 33% di total_energy con i primi 8 picchi,
    rendendo il cutoff irraggiungibile e facendo passare fino a
    MAX_PARTIALS picchi, incluso il rumore di floor a -170dB), ma
    l'energia integrata sul lobo principale attorno al picco (+-half_bins,
    derivati da mainlobe_width_hz, la stessa larghezza di mainlobe usata
    per la maschera di sottrazione spettrale). Con l'energia di lobo, 8
    picchi puliti raggiungono ~100% di total_energy e il cutoff torna a
    selezionarne 8, come atteso."""
    idx, props = find_peaks(mag_db, prominence=prominence_db)
    if len(idx) == 0:
        return []

    bin_hz = freqs[1] - freqs[0]
    half_bins = max(1, int(round((mainlobe_width_hz or (4 * bin_hz)) / bin_hz)))

    peaks = []
    for i in idx:
        if i <= 0 or i >= len(mag_db) - 1:
            peak_freq, peak_amp_db = float(freqs[i]), float(mag_db[i])
        else:
            a, b, c = mag_db[i - 1], mag_db[i], mag_db[i + 1]
            denom = (a - 2 * b + c)
            p = 0.5 * (a - c) / denom if abs(denom) > EPS else 0.0
            peak_freq = freqs[i] + p * bin_hz
            peak_amp_db = b - 0.25 * (a - c) * p
        lo, hi = max(0, i - half_bins), min(len(mag), i + half_bins + 1)
        lobe_energy = float(np.sum(mag[lo:hi] ** 2))
        peaks.append((float(peak_freq), float(peak_amp_db), lobe_energy))

    # ordina per ampiezza e taglia per energia cumulativa (non solo per conteggio)
    peaks.sort(key=lambda t: -t[1])
    energy = np.array([p[2] for p in peaks])
    ref_energy = total_energy if total_energy is not None else np.sum(energy)
    cum = np.cumsum(energy) / (ref_energy + EPS)
    cutoff_n = int(np.searchsorted(cum, energy_cutoff) + 1)
    n_keep = min(max_peaks, cutoff_n, len(peaks))
    return [(f, a) for (f, a, _e) in peaks[:n_keep]]


def _theil_sen_slope(x, y):
    """[C] stima robusta della pendenza (mediana delle pendenze a coppie),
    insensibile agli outlier rispetto al least-squares classico."""
    n = len(x)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) > EPS:
                slopes.append((y[j] - y[i]) / dx)
    return float(np.median(slopes)) if slopes else 0.0


class PartialTracker:
    """[E] Linking frame-a-frame via assegnazione ottima (Hungarian) su
    costo combinato freq+ampiezza, invece di greedy first-match."""

    def __init__(self, match_hz=PARTIAL_MATCH_HZ, death_frames=PARTIAL_DEATH_FRAMES,
                 amp_weight=0.1, decay_history=DECAY_HISTORY_FRAMES):
        self.match_hz = match_hz
        self.death_frames = death_frames
        self.amp_weight = amp_weight
        self.decay_history = decay_history
        self.tracks = []

    def update(self, peaks, frame_dt):
        n_tracks = len(self.tracks)
        n_peaks = len(peaks)
        used_peaks, matched_tracks = set(), set()

        if n_tracks and n_peaks:
            BIG = 1e6
            cost = np.full((n_tracks, n_peaks), BIG)
            for ti, tr in enumerate(self.tracks):
                for pj, (f, a) in enumerate(peaks):
                    d = abs(f - tr["freq"])
                    if d < self.match_hz:
                        cost[ti, pj] = d + self.amp_weight * abs(a - tr["amp_history"][-1])
            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, pj in zip(row_ind, col_ind):
                if cost[ti, pj] < BIG / 2:
                    f, a = peaks[pj]
                    tr = self.tracks[ti]
                    tr["freq"] = f
                    tr["amp_history"].append(a)
                    tr["age"] += 1
                    tr["missed"] = 0
                    used_peaks.add(pj)
                    matched_tracks.add(ti)

        for ti, tr in enumerate(self.tracks):
            if ti not in matched_tracks:
                tr["missed"] += 1

        self.tracks = [t for i, t in enumerate(self.tracks) if t["missed"] <= self.death_frames]

        for pj, (f, a) in enumerate(peaks):
            if pj not in used_peaks:
                self.tracks.append({"freq": f, "amp_history": [a], "age": 1, "missed": 0})

        return self.active_partials(frame_dt)

    def active_partials(self, frame_dt):
        out = []
        for tr in self.tracks:
            if tr["missed"] > 0:
                continue
            hist = tr["amp_history"][-self.decay_history:]
            if len(hist) >= 2:
                x = np.arange(len(hist)) * frame_dt
                slope_db_s = _theil_sen_slope(list(x), hist)
            else:
                slope_db_s = 0.0
            out.append({
                "freq": tr["freq"],
                "amp_db": tr["amp_history"][-1],
                "decay_rate_db_s": float(slope_db_s),
                "age_frames": tr["age"],
            })
        return out


def estimate_f0_from_partials(partials, fmin=F0_MIN_DEFAULT, fmax=F0_MAX_DEFAULT,
                               n_candidates=F0_CANDIDATES, max_partial=INHARMONIC_MAX_PARTIAL,
                               refine_top_k=F0_REFINE_TOP_K):
    """[I] harmonic template matching: cerca su una griglia di candidati f0
    quello che massimizza l'adesione (pesata in ampiezza) dei parziali
    tracciati a una serie armonica ideale n*f0. Sostituisce l'euristica
    "parziale piu' basso", che fallisce su spettri inarmonici o con
    fondamentale mancante.

    [M] FIX errore di sotto-ottava: la sola griglia grossolana (passo
    ~16Hz con i default) soffre di un bias sistematico verso candidati
    piu' bassi del vero f0 quando il candidato piu' vicino a f0 reale
    cade fuori griglia. Esempio misurato: f0=110Hz, 8 armoniche pulite,
    candidato piu' vicino in griglia 115.5Hz -> scarto relativo ~4.8% su
    OGNI armonica -> punteggio penalizzato uniformemente; un candidato
    piu' basso (es. 50Hz, il minimo consentito) puo' invece "agganciare"
    per coincidenza numerica piu' armoniche con scarti piccoli e
    totalizzare un punteggio piu' alto pur non essendo un fit esatto per
    nessuna (errore osservato: f0 stimato 50.0Hz invece di 110.0Hz, 54.5%).
    Si raffinano quindi con una minimizzazione locale bounded (in un
    intorno di un passo di griglia) i migliori `refine_top_k` candidati
    della ricerca grossolana, e si sceglie il migliore dopo il
    raffinamento: elimina l'errore di quantizzazione, che era la causa
    reale del bias verso frequenze sub-armoniche (verificato: errore
    54.5% -> 0.00% su f0=110/220/440/880/1760Hz)."""
    if not partials:
        return 0.0
    freqs = np.array([p["freq"] for p in partials])
    amps = np.array([10 ** (p["amp_db"] / 20.0) for p in partials])

    def _score(f0):
        n = np.clip(np.round(freqs / f0), 1, max_partial)
        ideal = n * f0
        dev = np.abs(freqs - ideal) / ideal
        return float(np.sum(amps * np.exp(-dev * 20.0)))

    def _refine(f0_guess, lo, hi):
        if hi <= lo:
            return float(f0_guess), _score(f0_guess)
        res = minimize_scalar(lambda f0: -_score(f0), bounds=(lo, hi),
                               method="bounded", options={"xatol": 0.05})
        return float(res.x), float(-res.fun)

    candidates = np.linspace(fmin, fmax, n_candidates)
    coarse_scores = np.array([_score(f0) for f0 in candidates])
    step = candidates[1] - candidates[0] if n_candidates > 1 else 0.0
    top_idx = np.argsort(-coarse_scores)[:max(1, refine_top_k)]

    best_f0, best_score = float(candidates[top_idx[0]]), -1.0
    for i in top_idx:
        f0_ref, score_ref = _refine(candidates[i], max(fmin, candidates[i] - step),
                                     min(fmax, candidates[i] + step))
        if score_ref > best_score:
            best_score, best_f0 = score_ref, f0_ref

    # [anti-sotto-ottava] un candidato sotto-multiplo di f0 puo' ottenere
    # lo STESSO punteggio massimo del vero f0 quando i parziali presenti
    # sono tutti armoniche pari di quel sotto-multiplo (fondamentale
    # "mancante": es. serie 880,1760,...,7040 e' spiegata in modo
    # ugualmente perfetto da candidato 440 -- ogni armonica dispari di 440
    # semplicemente non e' mai valutata, quindi la sua assenza non viene
    # penalizzata). Ambiguita' reale e nota nei pitch tracker armonici:
    # si preferisce quindi il multiplo intero piu' alto di best_f0 (entro
    # fmax) il cui punteggio raffinato resta entro OCTAVE_TIE_TOL dal
    # massimo trovato, iterando finche' non se ne trova uno migliore
    # (verificato: elimina l'errore di sotto-ottava misurato su f0=880Hz,
    # 50% -> 0.00%, senza alterare stime non ambigue come f0=331Hz).
    changed = True
    while changed:
        changed = False
        for mult in (2, 3, 4):
            cand = best_f0 * mult
            if cand > fmax:
                break
            f0_ref, score_ref = _refine(cand, max(fmin, cand - step), min(fmax, cand + step))
            if score_ref >= best_score * (1.0 - OCTAVE_TIE_TOL):
                best_f0, best_score = f0_ref, score_ref
                changed = True
    return best_f0


def inharmonicity_index(partials, f0, max_partial=INHARMONIC_MAX_PARTIAL):
    if f0 < EPS or not partials:
        return 0.0
    total_w, total_dev = 0.0, 0.0
    for p in partials:
        n = round(p["freq"] / f0)
        if n < 1 or n > max_partial:
            continue
        ideal = n * f0
        dev = abs(p["freq"] - ideal) / ideal
        w = 10 ** (p["amp_db"] / 20.0)
        total_dev += dev * w
        total_w += w
    return float(total_dev / total_w) if total_w > EPS else 0.0


def harmonic_partial_mask(freqs, partials, width_hz):
    """[F] width_hz ora derivato dalla mainlobe reale della finestra
    (vedi AnalyzerV3._mainlobe_width_hz), non un valore fisso arbitrario."""
    det_mag = np.full_like(freqs, EPS)
    for p in partials:
        amp_lin = 10 ** (p["amp_db"] / 20.0)
        det_mag += amp_lin * np.exp(-0.5 * ((freqs - p["freq"]) / width_hz) ** 2)
    return det_mag


# ---------------------------------------------------------------------------
# ONSET / TRANSIENT DETECTION
# ---------------------------------------------------------------------------

class OnsetDetector:
    """Rileva eventi transienti da un flusso di spectral flux, con soglia
    adattiva sulla media mobile recente. Espone anche mean/std del flux
    per la normalizzazione adattiva di transientness [G]."""

    def __init__(self, history=ONSET_HISTORY, mult=ONSET_THRESHOLD_MULT):
        self.history = history
        self.mult = mult
        self.flux_history = []
        self.last_onset_frame = -999
        self.frame_idx = 0
        self.onset_count_window = []

    def update(self, flux, now):
        self.frame_idx += 1
        is_onset = False
        if len(self.flux_history) >= 3:
            local_mean = float(np.mean(self.flux_history[-self.history:]))
            if flux > self.mult * (local_mean + EPS) and \
               (self.frame_idx - self.last_onset_frame) > 2:
                is_onset = True
                self.last_onset_frame = self.frame_idx
                self.onset_count_window.append(now)
        self.flux_history.append(flux)
        if len(self.flux_history) > 50:
            self.flux_history.pop(0)
        self.onset_count_window = [t for t in self.onset_count_window if now - t < 2.0]
        density = len(self.onset_count_window) / 2.0
        return is_onset, density

    def flux_zscore(self, flux):
        """[G] z-score del flux corrente rispetto alla storia recente,
        usato per normalizzare transientness in modo adattivo al contesto
        del segnale (silenzio vs materiale gia' denso)."""
        if len(self.flux_history) < 5:
            return 0.0
        mean = float(np.mean(self.flux_history))
        std = float(np.std(self.flux_history)) + EPS
        return (flux - mean) / std


# ---------------------------------------------------------------------------
# RING BUFFER PER STREAMING [B]
# ---------------------------------------------------------------------------

class RingBuffer:
    """Accumula campioni in arrivo a blocchi di dimensione arbitraria (es.
    il blocksize del device audio) ed estrae frame di analisi di dimensione
    fissa con hop corretto, disaccoppiando la dimensione del callback dalla
    dimensione della finestra di analisi. Risolve l'overlap rotto in v2,
    dove ogni chunk in arrivo (hop-sized) veniva zero-paddato e trattato
    come un frame completo indipendente."""

    def __init__(self, blocksize, hop):
        self.blocksize = blocksize
        self.hop = hop
        self.buf = np.zeros(0, dtype=np.float32)

    def push(self, samples):
        self.buf = np.concatenate([self.buf, samples.astype(np.float32)])
        frames = []
        while len(self.buf) >= self.blocksize:
            frames.append(self.buf[:self.blocksize].copy())
            self.buf = self.buf[self.hop:]
        return frames


# ---------------------------------------------------------------------------
# ANALYZER V3
# ---------------------------------------------------------------------------

class AnalyzerV3:
    def __init__(self, sample_rate=SAMPLE_RATE_DEFAULT, blocksize=BLOCKSIZE,
                 n_fft=N_FFT, bands=None, out_queue=None,
                 f0_min=F0_MIN_DEFAULT, f0_max=F0_MAX_DEFAULT,
                 pitch_every_n_frames=1, ema_alpha=EMA_ALPHA):
        self.sr = sample_rate                      # [A]
        self.blocksize = blocksize
        self.n_fft = max(n_fft, blocksize)
        self.bands = bands or BANDS
        self.window = np.hanning(self.blocksize)
        self.hop = int(blocksize * HOP_RATIO)
        self.frame_dt = self.hop / sample_rate

        self.f0_min = f0_min
        self.f0_max = f0_max
        self.pitch_every_n_frames = max(1, pitch_every_n_frames)   # [H]

        # [F] larghezza mainlobe reale della finestra di Hann (null-to-null
        # = 4*sr/blocksize; si usa la meta' come sigma della gaussiana)
        self._mainlobe_width_hz = 2.0 * self.sr / self.blocksize

        self.prev_mag = None
        self.frame_count = 0
        self.last_pitch = (0.0, 0.0)

        self.partial_tracker = PartialTracker()
        self.onset_detector = OnsetDetector()
        self.ring_buffer = RingBuffer(self.blocksize, self.hop)     # [B]

        self._ema = {}                                              # [J]
        self.ema_alpha = ema_alpha

        self.out_queue = out_queue if out_queue is not None else queue.Queue(maxsize=32)
        # [K] rimosso self._lock: non era usato da nessun path del codice

    def _smooth(self, key, value):
        prev = self._ema.get(key)
        new = value if prev is None else self.ema_alpha * value + (1 - self.ema_alpha) * prev
        self._ema[key] = new
        return new

    def process_frame(self, frame, include_spectrum=False):
        """include_spectrum=True aggiunge il campo "_spectrum" (freqs,
        power) al risultato - serve solo ad aggregate_energy_weighted
        (analisi offline su registrazione finita), costo extra evitato di
        default sul percorso realtime (mixer/GUI)."""
        if len(frame) != self.blocksize:
            frame = np.pad(frame, (0, self.blocksize - len(frame)))

        freqs, mag, phase = _magnitude_spectrum(frame, self.n_fft, self.window, self.sr)
        mag_db = 20.0 * np.log10(mag)

        # --- 1. strato deterministico: peak picking adattivo + tracking ottimo ---
        total_energy = float(np.sum(mag ** 2))   # potenza totale del frame - riusata sotto come peso energy-weighted
        total_mag = float(np.sum(mag))           # magnitudine totale del frame - peso per flux, vedi aggregate_energy_weighted
        peaks = pick_peaks(freqs, mag, mag_db, total_energy=total_energy,
                            mainlobe_width_hz=self._mainlobe_width_hz)
        partials = self.partial_tracker.update(peaks, self.frame_dt)

        # pitch: f0 primario dalle armoniche tracciate [I][M]. L'autocorrelazione
        # [H] resta calcolata solo per pitch_confidence (diagnostico): si e'
        # rivelata inaffidabile come sorgente di f0 anche su materiale
        # puramente armonico, per errore sistematico di sotto-ottava [N].
        self.frame_count += 1
        if self.frame_count % self.pitch_every_n_frames == 0:
            self.last_pitch = autocorrelation_pitch(frame, self.sr, self.f0_min, self.f0_max)
        f0_ac, pitch_conf = self.last_pitch
        f0 = estimate_f0_from_partials(partials, self.f0_min, self.f0_max)

        inharmonicity = inharmonicity_index(partials, f0)

        # --- 2. spectral subtraction -> residuo stocastico ---
        det_mag = harmonic_partial_mask(freqs, partials, self._mainlobe_width_hz)
        residual_mag = np.maximum(mag - det_mag, EPS)

        det_energy = float(np.sum(det_mag ** 2))
        res_energy = float(np.sum(residual_mag ** 2))
        total_energy_ds = det_energy + res_energy + EPS
        # rapporto da spectral subtraction: usato SOLO come diagnostica e per
        # ricavare il residuo su cui calcolare timbre_noise; NON piu' come
        # harmonicity/noisiness primarie (vedi noisiness_from_flatness)
        noisiness_subtraction = res_energy / total_energy_ds

        # [D+F] harmonicity/noisiness primarie: flatness, disaccoppiate dal
        # partial tracking (vedi motivazione in noisiness_from_flatness)
        noisiness_raw = noisiness_from_flatness(mag)
        harmonicity_raw = 1.0 - noisiness_raw

        # --- 3. flux + onset/transient detection, normalizzazione adattiva [G] ---
        flux = spectral_flux(mag.copy(), self.prev_mag)
        now = time.time()
        is_onset, onset_density = self.onset_detector.update(flux, now)
        z = self.onset_detector.flux_zscore(flux)
        transientness_raw = float(1.0 / (1.0 + np.exp(-TRANSIENT_Z_GAIN * z)))

        # --- 4. descrittori timbrici fine: totale + solo residuo ---
        timbre_total = timbral_descriptor_set(freqs, mag, frame=frame, prev_mag=self.prev_mag)
        timbre_noise = timbral_descriptor_set(freqs, residual_mag)

        # --- 5. statistiche per-strato sines ---
        if partials:
            decay_rates = [p["decay_rate_db_s"] for p in partials]
            freqs_p = np.array([p["freq"] for p in partials])
            jitter = float(np.std(freqs_p) / (np.mean(freqs_p) + EPS))
        else:
            decay_rates = [0.0]
            jitter = 0.0

        self.prev_mag = mag

        # --- [J] smoothing EMA sui descrittori di livello 1 ---
        harmonicity = self._smooth("harmonicity", harmonicity_raw)
        noisiness = self._smooth("noisiness", noisiness_raw)
        transientness = self._smooth("transientness", transientness_raw)

        descriptors = {
            "timestamp": now,
            "frame_energy": total_energy,    # per aggregate_energy_weighted: peso energy-weighted di centroid/rolloff/e_bands
            "frame_mag_sum": total_mag,      # per aggregate_energy_weighted: peso magnitude-weighted di flux

            # --- livello 1: strutturale (smoothed; _raw per debug/tuning) ---
            "harmonicity": float(harmonicity),
            "noisiness": float(noisiness),
            "transientness": float(transientness),
            "harmonicity_raw": float(harmonicity_raw),
            "noisiness_raw": float(noisiness_raw),
            "transientness_raw": float(transientness_raw),
            "is_onset": bool(is_onset),
            "onset_density_hz": float(onset_density),

            # --- livello 2: per-strato ---
            "sines": {
                "f0": float(f0),
                "pitch_confidence": float(pitch_conf),
                "inharmonicity": float(inharmonicity),
                "n_partials": len(partials),
                "partials": partials,
                "mean_decay_rate_db_s": float(np.median(decay_rates)),
                "jitter": jitter,
            },
            "noise": {
                "energy_ratio": float(noisiness_raw),
                "energy_ratio_subtraction_diag": float(noisiness_subtraction),
                "timbre": timbre_noise,
            },
            "transients": {
                "flux": float(flux),
                "flux_zscore": float(z),
                "onset_density_hz": float(onset_density),
            },

            "timbre_total": timbre_total,
        }
        if include_spectrum:
            descriptors["_spectrum"] = {"freqs": freqs, "power": mag ** 2}
        return descriptors

    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        """[B] usa il ring buffer: il chunk in arrivo (spesso hop-sized)
        non viene piu' zero-paddato e trattato come frame completo."""
        mono = indata[:, 0] if indata.ndim > 1 else indata
        for frame in self.ring_buffer.push(mono):
            descriptors = self.process_frame(frame)
            try:
                self.out_queue.put_nowait(descriptors)
            except queue.Full:
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
                self.out_queue.put_nowait(descriptors)

    def start_stream(self, device=None):
        import sounddevice as sd
        self.stream = sd.InputStream(
            samplerate=self.sr,
            blocksize=self.hop,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._audio_callback,
        )
        self.stream.start()
        return self.stream

    def stop_stream(self):
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()


# ---------------------------------------------------------------------------
# AGGREGAZIONE OFFLINE - allineata a descriptor_vector_torch
# ---------------------------------------------------------------------------
def aggregate_energy_weighted(frames, hop=None, sr=None, n_fft=None):
    """Aggrega una sequenza di frame (da AnalyzerV3.process_frame(...,
    include_spectrum=True), su una registrazione FINITA con inizio noto -
    non ha senso in streaming live infinito, dove non esiste un "totale") 
    con la STESSA logica di descriptor_vector_torch (synth_torch_before.py,
    la metrica che physical_agents_train.py ottimizza davvero in
    training): centroid/rolloff/e_bands pesati per energia sull'intera
    registrazione (pooling di potenza su tutti i frame, POI un solo
    rapporto), non media semplice frame-per-frame. La vecchia media
    semplice (ancora presente altrove come pattern implicito prima di
    questa funzione) diluisce il contributo dei suoni che decadono in
    fretta (strike/pluck/shaker) con i tanti frame quasi silenziosi dopo
    il decadimento - lo stesso bug gia' diagnosticato e corretto in
    descriptor_vector_torch (fix 2026-08-29), qui allineato.

    harmonicity/noisiness/flatness (RAW, non EMA-smoothed - lo smoothing
    resta utile per il consumo realtime di process_frame ma qui
    introdurrebbe un lag temporale assente in descriptor_vector_torch) sono
    ora pesate per frame_mag_sum (fix 2026-09-07, punto 1 review), non piu'
    una media semplice sui frame: un frame quasi silenzioso (coda dopo il
    decadimento di strike/pluck/shaker) faceva collassare flatness->1.0 e
    pesava quanto un frame pieno di segnale, gonfiando noisiness/schiacciando
    harmonicity indipendentemente dalla pulizia del transiente reale. Stessa
    logica gia' usata per flux qui sotto (frame_mag_sum, non frame_energy -
    flatness e' anch'essa un rapporto in dominio magnitudine), ora allineata
    a descriptor_vector_torch (synth_torch_before.py, fix gemello).

    rolloff usa una soglia rigida (searchsorted) e non la sigmoide morbida
    di descriptor_vector_torch: quella serve SOLO per restare
    differenziabile durante il training a gradiente, qui non ottimizziamo
    nulla per discesa del gradiente quindi non serve.

    "extra": descrittori di AnalyzerV3 non ancora usati in training -
    preservati e aggregati qui (media semplice sui frame) invece di andare
    persi, cosi' restano disponibili per calibrazione/uso futuro (vedi
    raccomandazione su come introdurli in physical_agents_train.py).

    Richiede "_spectrum" in ogni frame (include_spectrum=True) per
    centroid/rolloff/e_bands/t_centroid; senza, quei campi sono omessi dal
    risultato (harmonicity/noisiness/flux/extra restano comunque
    calcolabili dai soli campi scalari sempre presenti)."""
    if not frames:
        return {}
    has_spectrum = "_spectrum" in frames[0]

    # peso comune harmonicity/noisiness/flatness/flux: frame_mag_sum (non
    # frame_energy come per centroid/e_bands) - tutti e quattro sono rapporti
    # in dominio magnitudine, gia' normalizzati per-frame sulla propria
    # magnitudine (flux in spectral_flux, flatness/harmonicity/noisiness qui
    # sopra), equivalente algebrico del rapporto globale
    # sum(.)/sum(magnitudine) usato da descriptor_vector_torch.
    mag_sum_tot = sum(f["frame_mag_sum"] for f in frames) + EPS
    out = {
        "harmonicity": float(sum(f["harmonicity_raw"] * f["frame_mag_sum"] for f in frames) / mag_sum_tot),
        "noisiness": float(sum(f["noisiness_raw"] * f["frame_mag_sum"] for f in frames) / mag_sum_tot),
        "flatness": float(sum(f["timbre_total"]["flatness"] * f["frame_mag_sum"] for f in frames) / mag_sum_tot),
    }

    out["flux"] = float(sum(f["transients"]["flux"] * f["frame_mag_sum"] for f in frames) / mag_sum_tot)

    if has_spectrum:
        freqs = frames[0]["_spectrum"]["freqs"]
        power_pooled = np.sum([f["_spectrum"]["power"] for f in frames], axis=0)
        total_p = float(np.sum(power_pooled)) + EPS

        out["centroid"] = float(np.sum(freqs * power_pooled) / total_p)
        # spread: allineato a descriptor_vector_torch (energy-pooled, non
        # media della stima per-frame che stava in "extra") - ora e' un
        # descrittore usato in training (physical_agents_train.py), non
        # solo osservato.
        out["spread"] = float(np.sqrt(np.sum(power_pooled * (freqs - out["centroid"]) ** 2) / total_p))

        cumulative = np.cumsum(power_pooled)
        idx = np.searchsorted(cumulative, 0.90 * cumulative[-1])
        idx = min(idx, len(freqs) - 1)
        out["rolloff"] = float(freqs[idx])

        for (lo, hi), key in zip(BANDS, ("e_low", "e_mid", "e_high")):
            mask = (freqs >= lo) & (freqs < hi)
            out[key] = float(np.sum(power_pooled[mask]) / total_p)

        if hop is not None and sr is not None and n_fft is not None:
            e_frames = np.array([f["frame_energy"] for f in frames])
            t_frames = (np.arange(len(frames)) * hop + n_fft / 2.0) / sr
            out["t_centroid"] = float(np.sum(t_frames * e_frames) / (np.sum(e_frames) + EPS))

    extra_scalar_keys = ("skewness", "kurtosis", "slope", "hfc", "roughness", "zcr")   # spread promosso sopra
    out["extra"] = {k: float(np.mean([f["timbre_total"].get(k, 0.0) for f in frames])) for k in extra_scalar_keys}
    out["extra"]["transientness"] = float(np.mean([f["transientness_raw"] for f in frames]))
    out["extra"]["inharmonicity"] = float(np.mean([f["sines"]["inharmonicity"] for f in frames]))
    out["extra"]["jitter"] = float(np.mean([f["sines"]["jitter"] for f in frames]))
    out["extra"]["n_partials_mean"] = float(np.mean([f["sines"]["n_partials"] for f in frames]))
    for (lo, hi) in BANDS:
        bk = f"{lo}-{hi}"
        out["extra"][f"band_crest_{bk}"] = float(np.mean([f["timbre_total"]["band_crest"].get(bk, 0.0) for f in frames]))

    return out


# ---------------------------------------------------------------------------
# TEST DA TERMINALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    q = queue.Queue(maxsize=32)
    analyzer = AnalyzerV3(out_queue=q)
    print("Avvio analisi microfono (v3, SNT). CTRL+C per fermare.")
    analyzer.start_stream()

    try:
        count = 0
        while True:
            d = q.get()
            count += 1
            if count % 10 == 0:
                print(
                    f"harm={d['harmonicity']:.2f} noise={d['noisiness']:.2f} "
                    f"trans={d['transientness']:.2f} "
                    f"f0={d['sines']['f0']:6.1f}Hz "
                    f"inharm={d['sines']['inharmonicity']:.3f} "
                    f"nP={d['sines']['n_partials']:2d} "
                    f"onset={'*' if d['is_onset'] else ' '}"
                )
    except KeyboardInterrupt:
        print("\nStop.")
        analyzer.stop_stream()
