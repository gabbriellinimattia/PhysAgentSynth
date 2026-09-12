"""
ClaudeSynth - synth_torch: motore di sintesi differenziabile (sines+noise+
transient) per il fit per-discesa-del-gradiente dei descrittori target,
sostituisce level0_optimize.py (CMA-ES) e la resintesi 1:1 di
resynth_loop.py per il caso "sintesi per analisi" (target descrittori
indipendenti, non frame copiati dall'analyzer).

VERSIONE PRE-MICROSTRUTTURA (baseline per il confronto di non regressione
del punto 3) - salvata prima di aggiungere vibrato/tremolo/beating,
texture granulare e risonanze modali.

Fix 2026-08-29 (vedi physical_agents_train.py): descriptor_vector_torch
usava torch.stft(..., center=True), che aggiunge frame ai bordi costruiti
per riflessione del segnale. Su segnali che iniziano/finiscono vicino al
silenzio (impulsi, transienti) questi frame di bordo hanno una spectral
flatness anomala (fino a 4 ordini di grandezza sopra un frame pulito), e
la media finale e' fatta in spazio LINEARE (non dB): bastano 2 frame
corrotti su ~44 per far schizzare la noisiness misurata indipendentemente
dalla qualita' reale del segnale. center=False allinea la framing ad
analyzer.py (che non riflette mai ai bordi) ed elimina il bug.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9
SR_DEFAULT = 44100
N_FFT_DEFAULT = 2048
HOP_DEFAULT = N_FFT_DEFAULT // 2
BANDS = [(20, 300), (300, 3000), (3000, 20000)]
NOISINESS_DB_LOW, NOISINESS_DB_HIGH = -48.0, -1.0
LIMITER_DRIVE = 0.9


def _load_audio(path, seconds, sr=SR_DEFAULT):
    """Carica un file audio, mono, resamplato a sr, troncato a `seconds`,
    normalizzato al picco - inlined da resynth_loop.py (rimosso: era
    l'unico pezzo ancora usato di quella pipeline legacy, il resto
    - Mixer/sineSynth/noiseSynth/transientSynth - e' sostituito dal synth
    fisico-modale in physical_agents_train.py)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, file_sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(sr, file_sr)
        audio = resample_poly(audio, sr // g, file_sr // g)
    n = min(len(audio), int(seconds * sr))
    audio = audio[:n].astype(np.float64)
    peak = np.max(np.abs(audio)) + 1e-9
    return audio / peak


class DiffSines(nn.Module):
    def __init__(self, sr, f0, n_partials=16, learn_f0=False):
        super().__init__()
        self.sr = sr
        self.n_partials = n_partials
        k = torch.arange(1, n_partials + 1, dtype=torch.float32)
        self.register_buffer("k", k)
        f0_t = torch.tensor(float(f0))
        self.f0 = nn.Parameter(f0_t) if learn_f0 else None
        if not learn_f0:
            self.register_buffer("f0_fixed", f0_t)
        self.level_db = nn.Parameter(torch.tensor(-12.0))
        self.rolloff_db_oct = nn.Parameter(torch.tensor(6.0))
        self.decay_db_s = nn.Parameter(torch.tensor(5.0))
        self.B_raw = nn.Parameter(torch.tensor(-3.0))
        self.jitter = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_jitter_noise", torch.randn(4096))

    def forward(self, n_samples):
        f0 = self.f0 if self.f0 is not None else self.f0_fixed
        t = torch.arange(n_samples, dtype=torch.float32, device=self.k.device) / self.sr
        B = torch.nn.functional.softplus(self.B_raw)
        freqs = f0 * self.k * torch.sqrt(1.0 + B * self.k ** 2)
        amp_db = (self.level_db - self.rolloff_db_oct * torch.log2(self.k)
                  ).unsqueeze(1) - self.decay_db_s * t.unsqueeze(0)
        amp_lin = 10.0 ** (amp_db / 20.0)
        jn = self._jitter_noise[:n_samples] if n_samples <= 4096 else \
            self._jitter_noise.repeat(n_samples // 4096 + 1)[:n_samples]
        detune = 1.0 + self.jitter * jn.unsqueeze(0) * 0.01
        phase = 2.0 * np.pi * freqs.unsqueeze(1) * detune * t.unsqueeze(0)
        return (amp_lin * torch.sin(phase)).sum(0)


class DiffNoise(nn.Module):
    def __init__(self, sr, n_fft=N_FFT_DEFAULT, hop=HOP_DEFAULT, bands=None):
        super().__init__()
        self.sr, self.n_fft, self.hop = sr, n_fft, hop
        self.bands = bands or BANDS
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr)
        self.register_buffer("freqs", freqs)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        self.slope_raw = nn.Parameter(torch.tensor(0.0))
        self.shelf_raw = nn.Parameter(torch.tensor(3.0))
        self.tilt_raw = nn.Parameter(torch.tensor(0.0))
        self.gain_low = nn.Parameter(torch.tensor(0.0))
        self.gain_mid = nn.Parameter(torch.tensor(0.0))
        self.gain_high = nn.Parameter(torch.tensor(0.0))
        self.level_db = nn.Parameter(torch.tensor(-20.0))
        self._noise = None

    def _envelope(self):
        f = self.freqs
        env = torch.exp(torch.clamp(self.slope_raw * f, -50.0, 50.0))
        shelf_hz = torch.nn.functional.softplus(self.shelf_raw) * 1000.0 + 50.0
        env = env / (1.0 + (f / shelf_hz) ** 4)
        for (lo, hi), g in zip(self.bands, (self.gain_low, self.gain_mid, self.gain_high)):
            mask = (f >= lo) & (f < hi)
            env = torch.where(mask, env * (torch.nn.functional.softplus(g) + 1e-3), env)
        env = env * torch.exp(torch.clamp(self.tilt_raw * f * 1e-4, -50.0, 50.0))
        return env / (torch.sqrt((env ** 2).sum()) + EPS)

    def resample_noise(self, n_frames):
        n_freq = self.freqs.shape[0]
        self._noise = torch.complex(torch.randn(n_freq, n_frames), torch.randn(n_freq, n_frames)) \
            / np.sqrt(2.0)

    def forward(self, n_samples):
        n_frames = n_samples // self.hop + 4
        if self._noise is None or self._noise.shape[1] != n_frames:
            self.resample_noise(n_frames)
        env = self._envelope().unsqueeze(1)
        spec = env.to(torch.complex64) * self._noise
        wave = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop,
                            window=self.window, center=True, length=n_samples)
        return wave * (10.0 ** (self.level_db / 20.0))


class DiffTransient(nn.Module):
    def __init__(self, sr, n_samples_max=SR_DEFAULT * 6):
        super().__init__()
        self.sr = sr
        raw = torch.randn(n_samples_max)
        emphasized = raw.clone()
        emphasized[1:] = raw[1:] - 0.95 * raw[:-1]
        self.register_buffer("noise", emphasized)
        self.amp_raw = nn.Parameter(torch.tensor(-2.0))
        self.attack_ms = nn.Parameter(torch.tensor(2.0))
        self.decay_ms = nn.Parameter(torch.tensor(60.0))

    def forward(self, n_samples):
        idx = torch.arange(n_samples, dtype=torch.float32, device=self.noise.device)
        attack_s = torch.clamp(self.attack_ms, 0.1, 200.0) * 1e-3 * self.sr
        tau_s = torch.clamp(self.decay_ms, 1.0, 2000.0) * 1e-3 * self.sr
        env = torch.where(idx < attack_s, idx / attack_s, torch.exp(-(idx - attack_s) / tau_s))
        amp = torch.nn.functional.softplus(self.amp_raw)
        return amp * env * self.noise[:n_samples]


class EventSynth(nn.Module):
    def __init__(self, sr=SR_DEFAULT, n_fft=N_FFT_DEFAULT, hop=HOP_DEFAULT,
                 f0=220.0, n_partials=16, learn_f0=False):
        super().__init__()
        self.sr = sr
        self.sines = DiffSines(sr, f0, n_partials, learn_f0)
        self.noise = DiffNoise(sr, n_fft, hop)
        self.trans = DiffTransient(sr)
        self.w_sines_raw = nn.Parameter(torch.tensor(0.0))
        self.w_noise_raw = nn.Parameter(torch.tensor(0.0))
        self.w_trans_raw = nn.Parameter(torch.tensor(-2.0))
        self.master_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, seconds):
        n = int(round(seconds * self.sr))
        s = self.sines(n)
        no = self.noise(n)
        tr = self.trans(n)
        mix = (torch.sigmoid(self.w_sines_raw) * s
               + torch.sigmoid(self.w_noise_raw) * no
               + torch.sigmoid(self.w_trans_raw) * tr)
        mix = mix * torch.nn.functional.softplus(self.master_raw)
        over = mix.abs() > LIMITER_DRIVE
        headroom = 1.0 - LIMITER_DRIVE
        excess = (mix.abs() - LIMITER_DRIVE).clamp(min=0.0)
        soft = torch.sign(mix) * (LIMITER_DRIVE + headroom * torch.tanh(excess / headroom))
        return torch.where(over, soft, mix)


def descriptor_vector_torch(x, sr, n_fft=N_FFT_DEFAULT, hop=HOP_DEFAULT, bands=None):
    bands = bands or BANDS
    window = torch.hann_window(n_fft, periodic=True, device=x.device)
    # center=False (non center=True): vedi nota di fix in testa al file.
    # torch.stft(center=True) inserisce frame di bordo costruiti per
    # riflessione del segnale, con flatness anomala su segnali vicini al
    # silenzio ai bordi (impulsi/transienti) - corrompe la media, fatta in
    # spazio lineare. center=False allinea la framing ad analyzer.py.
    if x.shape[-1] < n_fft:
        x = torch.nn.functional.pad(x, (0, n_fft - x.shape[-1]))
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window,
                       center=False, return_complex=True)
    mag = spec.abs() + EPS
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(x.device).unsqueeze(1)

    # Fix 2026-08-29 (bis): centroid/rolloff/bande erano una MEDIA di un
    # rapporto per-frame (.mean(0)/.mean() su una quantita' gia'
    # normalizzata per singolo frame), che pesa ogni frame allo stesso
    # modo indipendentemente dalla sua energia. Per un suono che decade
    # in fretta (es. strike: pochi ms di contenuto vero poi silenzio per
    # il resto del buffer) i molti frame quasi silenziosi dopo il
    # decadimento diluiscono la media. Un modo risonante a bassa
    # frequenza decade piu' lentamente (tau ~ Q/(pi*f)) di uno a
    # frequenza alta a parita' di Q, quindi occupa piu' frame "vivi" e
    # finisce per dominare la media anche quando l'energia totale e'
    # correttamente concentrata in alto (verificato: isolando i soli
    # modi >=3000Hz di un banco allenato, e_high=0.86; nello stesso
    # segnale completo, dove i modi bassi durano molto di piu', e_high
    # crolla a 0.02 pur essendo l'energia alta genuinamente presente).
    # Fix: aggregare per energia sull'intero segnale (somma su frequenza
    # E tempo, poi dividi/soglia) invece di mediare un rapporto per-frame.
    power = mag ** 2
    total_p_all = power.sum() + EPS

    centroid = (freqs * power).sum() / total_p_all
    # spread: stessa aggregazione energy-pooled di centroid (non media di
    # una deviazione per-frame) - dispersione spettrale attorno al
    # centroid sull'intero evento. Serve come target allenabile (punto 10:
    # allargare lo spazio target oltre harmonicity/noisiness/bande, che da
    # sole non distinguono uno spettro stretto-e-spostato da uno largo
    # centrato uguale).
    spread = torch.sqrt(((freqs - centroid) ** 2 * power).sum() / total_p_all)
    # Fix 2026-09-07 (punto 1 review): come per centroid/rolloff/bande (fix
    # 2026-08-29), la flatness era una media SEMPLICE per-frame - un frame
    # quasi silenzioso (coda dopo il decadimento di strike/pluck/shaker) ha
    # mag->EPS uniforme su tutti i bin, quindi flatness->1.0 esatto (massima
    # rumorosita' per costruzione) e pesa quanto un frame pieno di segnale.
    # Pesata per mag_sum del frame (non power): la flatness stessa e' un
    # rapporto in dominio magnitudine (geo_mean(mag)/arith_mean(mag)), stessa
    # logica gia' usata per flux in analyzer.py (frame_mag_sum, non
    # frame_energy - vedi commento li'), qui allineato.
    flat_per_frame = torch.exp(torch.log(mag).mean(0)) / mag.mean(0)
    mag_sum_per_frame = mag.sum(0)
    flat = (flat_per_frame * mag_sum_per_frame).sum() / (mag_sum_per_frame.sum() + EPS)
    db = 10.0 * torch.log10(flat + EPS)
    # Fix 2026-09-09: clamp lineare hard -> gradiente ESATTAMENTE zero fuori
    # [LOW,HIGH] (stesso bug gia' corretto per flux, vedi commento sopra).
    # Isolato su strike: db bloccato saturo (SAT) all'estremo per l'intera run
    # su file collassati (Va-legno_batt-C5, Hn-slap-C#5), mai su file sani
    # (tt-tt_edge esce dal clamp entro it=60) - e con db saturo q_raw.grad e'
    # esattamente 0.000000 per tutta la run (verificato via istrumentazione
    # diretta su .grad.norm()), perche' harmonicity/noisiness e' l'unico canale
    # achieved sensibile a Q (bandwidth modale -> piattezza spettrale). Sigmoid
    # al posto del clamp lineare (stesso pattern di rolloff sopra): satura agli
    # stessi estremi (~0.018/~0.982 vs 0/1 esatti) ma non ha mai gradiente nullo.
    NOISINESS_SHARPNESS = 8.0 / (NOISINESS_DB_HIGH - NOISINESS_DB_LOW)
    noisiness = torch.sigmoid(NOISINESS_SHARPNESS * (db - (NOISINESS_DB_LOW + NOISINESS_DB_HIGH) / 2.0))
    harmonicity = 1.0 - noisiness

    power_pooled = power.sum(1)   # profilo spettrale dell'intero evento (somma sul tempo)
    frac = torch.cumsum(power_pooled, 0) / (power_pooled.sum() + EPS)
    # vedi nota sopra (fix 2026-08-29) sul perche' sigmoid e non softmax
    # sul picco: qui applicata una sola volta al profilo aggregato invece
    # che per-frame.
    sig = torch.sigmoid(300.0 * (0.90 - frac))
    bin_width = freqs[1, 0] - freqs[0, 0]
    rolloff = freqs[0, 0] + bin_width * sig.sum()

    # flux: incremento positivo di magnitudo frame-a-frame, aggregato
    # sull'intero segnale (stessa filosofia energy-weighted della fix
    # sopra, non media di rapporti per-frame). E' nel target (random_target
    # lo popola per tutti gli exciter type) e finisce in loss in
    # physical_agents_train.py via _weighted_loss_terms (qualunque chiave
    # presente sia in achieved che in target vi entra).
    #
    # clamp(min=0.0) (fix precedente): esatto ma con gradiente ESATTAMENTE
    # zero ovunque il segnale sia in puro decadimento (ogni frame <= il
    # precedente) - non un minimo locale morbido, una regione piatta vera.
    # Per pluck (decay_ms 0.5-8ms, molto piu' corto di un hop=1024
    # campioni/23ms) questo e' l'unico regime raggiungibile: flux osservato
    # bloccato a 0.0 ESATTO per 3 run consecutivi, invariato da qualunque
    # modifica all'exciter (attack, tilt nel tempo) perche' nessun
    # parametro riceve mai gradiente per smuoverlo.
    #
    # softplus(x*S)/S (primo tentativo) RITIRATO - bug di scala: softplus(0)
    # = ln(2), non 0, quindi OGNI coppia di frame silenzio-a-silenzio (la
    # stragrande maggioranza, es. pluck attivo per 1-2 frame su ~40 in un
    # buffer da 1s) contribuisce un residuo ~ln(2)/S costante invece di 0 -
    # sommato su ~1025 bin freq * decine di frame silenziosi esplode
    # (osservato: flux fino a 6.5, contro un range target 0.03-0.4,
    # inquinando la loss anche di strike/pluck che gia' andavano discretamente).
    # x*sigmoid(x*S): vale ESATTAMENTE 0 in x=0 (nessun residuo su coppie
    # silenzio-silenzio), tende a x per x>>0 (stessa pendenza di relu) e a 0
    # per x<<0 (non una costante), quindi non accumula mai sul grosso
    # background quasi-nullo - ma ha gradiente non nullo vicino a 0, l'unica
    # differenza che serve per uscire dalla regione piatta. Normalizzato
    # sulla magnitudine media del segnale (adimensionale) invece di una
    # soglia assoluta fissa, cosi' la sharpness non va ritarata per segnali
    # di livello diverso.
    FLUX_SHARPNESS = 8.0
    if mag.shape[1] > 1:
        diff = mag[:, 1:] - mag[:, :-1]
        scale = mag.mean() + EPS
        d = diff / scale
        soft = d * torch.sigmoid(d * FLUX_SHARPNESS)
        flux = (soft * scale).sum() / (mag.sum() + EPS)
        flux = flux.clamp(min=0.0)
    else:
        flux = torch.zeros((), device=x.device, dtype=mag.dtype)

    # centroide temporale: energia pesata sull'istante del frame (centro
    # frame, coerente con center=False), non sulla frequenza - un
    # decadimento rapido concentra l'energia nei primi frame (valore
    # basso), un ring lungo/tono sostenuto la sposta in avanti (valore
    # alto vicino a meta' buffer). Nessun descrittore esistente vede
    # questa dimensione (harmonicity/noisiness sono forma spettrale,
    # flux conta solo incrementi ed e' ~0 per un puro decadimento) -
    # serve come precondizione per scollegare T60 da Q nel banco modale
    # (punto 8): senza un descrittore che misuri il decadimento, un
    # parametro di T60 esplicito per modo non riceverebbe gradiente.
    n_frames = power.shape[1]
    t_frame = (torch.arange(n_frames, dtype=torch.float32, device=x.device) * hop
               + n_fft / 2.0) / sr
    power_per_frame = power.sum(0)
    t_centroid = (t_frame * power_per_frame).sum() / (power_per_frame.sum() + EPS)

    out = {"centroid": centroid, "flatness": flat, "harmonicity": harmonicity,
           "noisiness": noisiness, "rolloff": rolloff, "flux": flux,
           "t_centroid": t_centroid, "spread": spread}
    for (lo, hi), key in zip(bands, ("e_low", "e_mid", "e_high")):
        mask = (freqs >= lo) & (freqs < hi)
        out[key] = (power * mask).sum() / total_p_all
    # Fix 2026-09-09 (isolamento q_raw, seguito a DECAY_BANDS/decay_slope_db_per_band
    # sopra): harmonicity/noisiness pooled sull'INTERO spettro e' un solo scalare per
    # 12 modi - unico canale rimasto sensibile a Q (tau_raw ha gia' scollegato il
    # decadimento da Q, physical_agents_train.py "punto 8") ma senza risoluzione
    # per-modo, competere in _aggregate_loss/LogSumExp con gain/freq lo diluisce
    # (letteratura: identificabilita' da segnale aggregato, vedi anche Diaz & Hayes
    # ICASSP2023 su gradiente che sparisce per i modi meno energetici). Stessa
    # flatness/sigmoid di sopra ma per banda (stesse bande di e_low/e_mid/e_high):
    # 3 termini indipendenti invece di 1, ciascuno piu' vicino ai modi che ci
    # cadono dentro.
    freqs_flat = freqs.squeeze(1)
    for (lo, hi), key in zip(bands, ("low", "mid", "high")):
        band_mask = (freqs_flat >= lo) & (freqs_flat < hi)
        mag_b = mag[band_mask]
        if mag_b.shape[0] == 0:
            continue
        flat_pf_b = torch.exp(torch.log(mag_b).mean(0)) / mag_b.mean(0)
        mag_sum_pf_b = mag_b.sum(0)
        flat_b = (flat_pf_b * mag_sum_pf_b).sum() / (mag_sum_pf_b.sum() + EPS)
        db_b = 10.0 * torch.log10(flat_b + EPS)
        noisiness_b = torch.sigmoid(NOISINESS_SHARPNESS * (db_b - (NOISINESS_DB_LOW + NOISINESS_DB_HIGH) / 2.0))
        out[f"harmonicity_{key}"] = 1.0 - noisiness_b
        out[f"noisiness_{key}"] = noisiness_b
    return out


RATIO_KEYS = ("centroid", "rolloff", "f0", "spread", "t_centroid")

DECAY_BANDS = BANDS   # stessi bordi banda di e_low/e_mid/e_high (BANDS sopra)


def decay_slope_db_per_band(x, sr, n_fft=N_FFT_DEFAULT, hop=HOP_DEFAULT, bands=None):
    """Slope (dB/s) della Energy Decay Curve per banda (Schroeder 1965,
    integrazione ALL'INDIETRO dell'energia - flip/cumsum/flip, poi
    regressione lineare sulla curva in dB, tutto differenziabile).

    Fix 2026-09-09 (isolamento q_raw): i descrittori sopra (harmonicity via
    flatness, centroid, spread, ...) sono pooled sull'INTERO buffer in una
    manciata di scalari globali - segnale debole per Q/decadimento, che e'
    una proprieta' TEMPORALE (letteratura: Diaz & Hayes, "Rigid-Body Sound
    Synthesis with Differentiable Modal Resonators", ICASSP 2023 - stessa
    architettura banco-modale, stesso fenomeno di gradiente quasi nullo sul
    decadimento con loss spettrale globale; Mezza et al., DAFx24, fix con
    Energy Decay Curve per banda al posto/in aggiunta alla loss spettrale).
    Qui la EDC per banda da' un canale diretto, per-banda (non un solo
    scalare globale come t_centroid) verso Q, indipendente dal resto della
    pipeline descrittori/tanh/LogSumExp - stesso pattern architetturale di
    gain_profile_loss/freq_drift_loss (physical_agents_train.py): termine
    additivo esterno, non un altro descrittore dentro achieved."""
    bands = bands or DECAY_BANDS
    window = torch.hann_window(n_fft, periodic=True, device=x.device)
    if x.shape[-1] < n_fft:
        x = torch.nn.functional.pad(x, (0, n_fft - x.shape[-1]))
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window,
                       center=False, return_complex=True)
    power = spec.abs() ** 2 + EPS                     # (freq, n_frames)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(x.device)
    n_frames = power.shape[1]
    t_frame = (torch.arange(n_frames, dtype=torch.float32, device=x.device) * hop) / sr
    t_mean = t_frame.mean()
    t_c = t_frame - t_mean
    denom = (t_c * t_c).sum().clamp(min=EPS)

    slopes = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_power = power[mask].sum(0)                                    # (n_frames,)
        edc = torch.flip(torch.cumsum(torch.flip(band_power, [0]), 0), [0])  # Schroeder backward integration
        edc_db = 10.0 * torch.log10(edc + EPS)
        db_mean = edc_db.mean()
        slope = (t_c * (edc_db - db_mean)).sum() / denom   # dB/s (negativo per un decadimento normale)
        slopes.append(slope)
    return torch.stack(slopes)



# spread: su scala Hz come centroid, stesso errore log2-ratio.
# t_centroid: PROMOSSO da errore assoluto*peso200 (physical_agents_train.py,
# LOSS_WEIGHTS) a log2-ratio - il peso 200 era tarato sul range stretto di
# strike/pluck (0.015-0.065s): esteso il target a shaker (0.03-0.25s, vedi
# random_target) lo stesso peso assoluto ha fatto esplodere il termine
# (errori piu' grandi in valore assoluto -> *200 domina la loss, mascherando
# gli altri termini - osservato su run reale: shaker con loss totale 2-4
# mentre tutti gli altri termini restavano 0.0-0.06). Il rapporto log2 si
# auto-scala col target (come gia' per centroid/rolloff/spread), niente
# piu' peso ad-hoc per famiglia.


def _descriptor_loss_terms(achieved, target):
    terms = []
    for k, v in target.items():
        if k not in achieved:
            continue
        v = float(v)
        if k in RATIO_KEYS:
            terms.append(torch.log2((achieved[k] + EPS) / (v + EPS)) ** 2)
        else:
            terms.append((achieved[k] - v) ** 2)
    return terms


def _inv_softplus(y):
    y = max(float(y), 1e-4)
    return float(np.log(np.expm1(y)))


def _warm_start(engine, target):
    with torch.no_grad():
        if "harmonicity" in target:
            h = float(np.clip(target["harmonicity"], 0.02, 0.98))
            engine.w_sines_raw.copy_(torch.logit(torch.tensor(h)))
            engine.w_noise_raw.copy_(torch.logit(torch.tensor(1.0 - h)))
        for key, g in (("e_low", engine.noise.gain_low), ("e_mid", engine.noise.gain_mid),
                        ("e_high", engine.noise.gain_high)):
            if key in target:
                g.fill_(_inv_softplus(max(float(target[key]), 1e-3) * 3.0))


def fit_to_target(target, f0=220.0, n_partials=16, seconds=1.0, sr=SR_DEFAULT,
                   n_fft=N_FFT_DEFAULT, hop=HOP_DEFAULT, iters=500, lr=0.02,
                   log=None):
    engine = EventSynth(sr, n_fft, hop, f0=f0, n_partials=n_partials)
    _warm_start(engine, target)
    opt = torch.optim.Adam(engine.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    achieved, loss_val = {}, None
    for it in range(iters):
        opt.zero_grad()
        x = engine(seconds)
        achieved = descriptor_vector_torch(x, sr, n_fft, hop)
        terms = _descriptor_loss_terms(achieved, target)
        loss = torch.stack(terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 5.0)
        opt.step()
        sched.step()
        loss_val = float(loss.detach())
        if log and (it % max(1, iters // 10) == 0 or it == iters - 1):
            log(f"  it {it:4d}  loss={loss_val:.4f}")
    achieved_f = {k: float(v.detach()) for k, v in achieved.items()}
    return engine, achieved_f, loss_val


def render_numpy(engine, seconds):
    with torch.no_grad():
        engine.noise._noise = None
        x = engine(seconds).detach().cpu().numpy().astype(np.float32)
    return x


def save_wav(path, audio, sr=SR_DEFAULT):
    try:
        import soundfile as sf
        sf.write(path, audio, sr)
    except Exception:
        from scipy.io import wavfile
        wavfile.write(path, sr, (audio * 32767).astype(np.int16))


def analyze_reference(path, seconds=2.0, sr=SR_DEFAULT, n_fft=N_FFT_DEFAULT):
    """Allineata a descriptor_vector_torch (la metrica che train_agent
    ottimizza davvero - vedi analisi): centroid/rolloff/e_bands pesati per
    energia sull'intera registrazione via aggregate_energy_weighted
    (analyzer.py), non piu' una media semplice frame-per-frame - quella
    vecchia media diluiva il contributo dei suoni che decadono in fretta
    (strike/pluck/shaker) con i tanti frame quasi silenziosi dopo il
    decadimento, lo stesso bug gia' corretto in descriptor_vector_torch
    (fix 2026-08-29) ma rimasto qui fino ad ora.

    target ora include anche flux/t_centroid (prima assenti, silenziosamente
    ignorati da _descriptor_loss_terms) e target["extra"] con i descrittori
    di AnalyzerV3 non ancora usati in training (spread/skewness/kurtosis/
    slope/hfc/roughness/zcr/transientness/inharmonicity/jitter/band_crest)
    - preservati, non consumati da fit_to_target (chiave "extra" ignorata,
    non e' in achieved), disponibili per calibrazione/uso futuro."""
    from analyzer import AnalyzerV3, aggregate_energy_weighted
    audio = _load_audio(path, seconds, sr)
    analyzer = AnalyzerV3(sample_rate=sr, blocksize=n_fft)
    frames = [analyzer.process_frame(c, include_spectrum=True)
              for c in analyzer.ring_buffer.push(audio.astype(np.float32))]
    if not frames:
        raise ValueError("file troppo corto per l'analisi")
    target = aggregate_energy_weighted(frames, hop=analyzer.hop, sr=sr, n_fft=analyzer.n_fft)
    f0s = [f["sines"]["f0"] for f in frames if f["sines"]["pitch_confidence"] > 0.3]
    f0 = float(np.median(f0s)) if f0s else 220.0
    with torch.no_grad():
        audio_t = torch.from_numpy(audio.astype(np.float32))
        target["decay_slope"] = decay_slope_db_per_band(audio_t, sr)
        band_desc = descriptor_vector_torch(audio_t, sr)
        for k in ("harmonicity_low", "harmonicity_mid", "harmonicity_high",
                  "noisiness_low", "noisiness_mid", "noisiness_high"):
            target[k] = float(band_desc[k])
    return target, f0
