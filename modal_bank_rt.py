"""modal_bank_rt.py - banco modale riprogettato per l'USO A RUNTIME
(pseudo real-time: descrittori campionati ogni X ms, interpolazione tra
un campionamento e l'altro - vedi discussione in sessione).

Il ModalBank di physical_agents_train.py filtra l'INTERO buffer in un
colpo solo via FFT con coefficienti statici - efficiente e differenziabile
(necessario per il training a gradiente), ma incompatibile col real-time
per due motivi:
  1) non mantiene stato tra una chiamata e l'altra (il "ring" dei modi si
     azzera ad ogni forward()) - impossibile spezzare l'audio in blocchi
     consecutivi;
  2) i coefficienti sono fissi per tutto il buffer - nessun modo di far
     scivolare un parametro dal valore vecchio al nuovo senza un salto
     secco (click) tra un blocco e l'altro.

QUESTO modulo non tocca il training (resta physical_agents_train.py, FFT,
autograd, invariato) - e' l'engine usato in performance, DOPO che
resonator_predictor.py ha gia' mappato un target in parametri: qui non
serve il gradiente, solo DSP a bassa latenza con stato.

Ogni modo e' un risonatore a 2 poli in forma biquad RBJ ("BPF, constant
peak gain", Audio EQ Cookbook: https://webaudio.github.io/Audio-EQ-Cookbook/)
invece che nel dominio della frequenza: picco unitario esatto a f_i
indipendente da Q, poi scalato per gain*Q - stessa convenzione di "picco
= gain*Q" gia' usata nella loss di physical_agents_train.py. Non e' una
riproduzione bit-esatta della risposta H(f)=1/((1-(f/f0)^2)+j(f/f0)/Q)
usata offline (la forma del roll-off lontano dal picco differisce), ma
l'equivalente pratico per il picco risonante che conta a runtime.

Aggiornamento coefficienti ogni `coeff_chunk` campioni (default 64,
~1.5ms a 44.1kHz), non ad ogni campione: ricalcolare sin/cos per modo ad
ogni campione e' troppo costoso in puro Python. Compromesso standard in
DSP realtime, sufficiente ad eliminare i click. Per uso davvero
sample-accurate/a bassissima latenza in produzione questo loop andrebbe
portato a numba/C - qui e' pensato per pseudo real-time via Python.

Schema d'uso tipico:
    bank = ModalBankRT(sr=44100, n_modes=12)
    while streaming:
        if nuovo_target_disponibile:              # ogni X ms
            freq, gain, q = predittore(target)      # resonator_predictor.py, istantaneo
            bank.set_target(freq, gain, q, glide_samples=int(X_ms/1000*sr))
        out_block = bank.process_block(exciter_block)
        riproduci(out_block)
"""
import numpy as np

Q_MAX = 600.0


class ModalBankRT:
    def __init__(self, sr, n_modes=12, coeff_chunk=64):
        self.sr = sr
        self.n_modes = n_modes
        self.coeff_chunk = coeff_chunk

        # stato del biquad per modo (Direct Form I): sopravvive tra i blocchi.
        self.x1 = np.zeros(n_modes)
        self.x2 = np.zeros(n_modes)
        self.y1 = np.zeros(n_modes)
        self.y2 = np.zeros(n_modes)

        # valore corrente (interpolato) dei parametri
        self.freq = np.full(n_modes, 220.0)
        self.gain = np.zeros(n_modes)
        self.q = np.full(n_modes, 8.0)

        # punto di partenza/arrivo dell'interpolazione in corso
        self._freq0 = self.freq.copy(); self._freq1 = self.freq.copy()
        self._gain0 = self.gain.copy(); self._gain1 = self.gain.copy()
        self._q0 = self.q.copy();       self._q1 = self.q.copy()
        self._glide_total = 1
        self._glide_pos = 1  # gia' a target: nessuna interpolazione finche' non arriva un set_target

    def set_target(self, freq, gain, q, glide_samples):
        """Nuovo bersaglio da un campionamento del descrittore (uscita del
        predittore). Il valore CORRENTE (quello raggiunto finora
        dall'interpolazione) diventa il nuovo punto di partenza;
        glide_samples e' la distanza in campioni fino al prossimo
        campionamento (X-time convertito in campioni con sr) - i parametri
        raggiungono il bersaglio linearmente esattamente in quella
        finestra: e' cosi' che "il tempo come parametro" entra nel motore
        audio, non come idea astratta ma come interpolazione reale tra due
        campionamenti consecutivi dei descrittori."""
        self._freq0, self._gain0, self._q0 = self.freq.copy(), self.gain.copy(), self.q.copy()
        self._freq1 = np.clip(np.asarray(freq, dtype=np.float64), 1.0, self.sr * 0.49)
        self._gain1 = np.asarray(gain, dtype=np.float64)
        self._q1 = np.clip(np.asarray(q, dtype=np.float64), 0.5, Q_MAX)
        self._glide_total = max(1, int(glide_samples))
        self._glide_pos = 0

    def _biquad_coeffs(self, freq, q):
        w0 = 2.0 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2.0 * np.clip(q, 0.5, Q_MAX))
        b0, b2 = alpha, -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha
        return b0 / a0, b2 / a0, a1 / a0, a2 / a0

    def process_block(self, excitation):
        """excitation: array (n_samples,), la stessa eccitazione va in
        ingresso a tutti i modi (come in ModalBank.forward originale).
        Ritorna l'audio del blocco; lo stato resta nell'oggetto per il
        blocco successivo (continuita' tra chiamate, a differenza della
        versione FFT offline)."""
        excitation = np.asarray(excitation, dtype=np.float64)
        n = len(excitation)
        out = np.zeros(n)
        chunk = self.coeff_chunk
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            frac = min(1.0, self._glide_pos / self._glide_total)
            self.freq = self._freq0 + (self._freq1 - self._freq0) * frac
            self.gain = self._gain0 + (self._gain1 - self._gain0) * frac
            self.q = self._q0 + (self._q1 - self._q0) * frac
            self._glide_pos += (end - start)

            b0, b2, a1, a2 = self._biquad_coeffs(self.freq, self.q)
            peak_scale = self.gain * self.q   # stessa convenzione "picco=gain*Q" del training offline

            for i in range(start, end):
                xn = excitation[i]
                yn = b0 * xn + b2 * x2 - a1 * y1 - a2 * y2   # b1=0 nella forma BPF RBJ
                x2, x1 = x1, xn
                y2, y1 = y1, yn
                out[i] = float(np.sum(peak_scale * yn))

        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return out

    def reset(self):
        """Azzera stato e interpolazione (nuova nota/attacco da zero)."""
        self.x1[:] = 0.0; self.x2[:] = 0.0; self.y1[:] = 0.0; self.y2[:] = 0.0
        self._glide_pos = self._glide_total
