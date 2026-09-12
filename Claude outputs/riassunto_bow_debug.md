# Contesto progetto

Sintetizzatore fisico-modale differenziabile in PyTorch. Repo: `~/Desktop/ClaudeSynth` sul Mac dell'utente, raggiungibile SOLO via device bridge (`mcp__remote-devices__device_bash` ecc.) — MAI dal sandbox cloud (device_bash non ha torch: gli script torch vanno consegnati all'utente da eseguire nel suo terminale Mac, risultati incollati indietro). Istruzioni progetto: rispondere in italiano, sintetico, diretto, un solo punto di vista per risposta, mai proporre codice non richiesto, mai supporre ciecamente — sempre misurare prima di proporre un fix.

Obiettivo corrente: chiudere il gap di qualità harmonicity/noisiness di `bow` fino al livello di `shaker` (benchmark gap≈0.040, già raggiunto).

# Architettura rilevante

- `physical_agents_train.py`: agenti per-exciter-type su un `ModalBank` condiviso (bandpass a Q alto, Q_MAX=600).
- `FEEDBACK_TYPES={"bow","blow"}`: `freq_raw` fissato alle armoniche esatte di f0; allenabili solo `gain_raw` e i parametri propri dell'exciter.
- `_feedback_synthesize`: genera `tone` (armoniche deterministiche) e `noise_total` (rumore colorato) SEPARATI; il chiamante passa `tone` nel resonatore (`modal()`), poi somma il rumore — evita che il resonatore "pulisca" il rumore in energia tonale.
- Metrica `harmonicity`/`noisiness` in `synth_torch_before.py::descriptor_vector_torch`: NON un rapporto di energia tono/rumore, ma flatness spettrale (media geometrica/aritmetica magnitudo STFT) → dB → sigmoide. Satura velocemente: bastano tracce minime di rumore per spostarla molto (vedi "cliff" sotto).
- Target reali (`analyzer.py::noisiness_from_flatness`) usano invece un clip LINEARE, formula diversa dalla sigmoide achieved — mismatch noto, MAI risolto per scelta esplicita dell'utente (vedi sotto).

# Storia dei tentativi su bow (in ordine, tutti falsificati tranne l'ultimo)

1. **gain_raw isolato con LR*2** — nessun effetto misurato (tone/noise ratio non è il collo di bottiglia: la metrica non è un rapporto di energia).
2. **roughness_raw init abbassato a -4.5** — RITIRATO: su 2/3 file roughness risale comunque a 0.13-0.16 (gradiente reale, non saturato, va nella direzione "sbagliata"); su 1/3 file il training collassa in plateau morto. Ripristinato a -2.0.
3. **HARMONICITY_WEIGHT_BOW_MULT=3.0** (moltiplicatore di peso loss su harmonicity/noisiness) — nessun effetto (gradiente vicino a zero in quella zona, pesarlo di più non crea segnale). Lasciato nel codice, mai rimosso.
4. **noise_floor fisso non allenabile 0.00005** — pensato per uscire dal "cliff" (roughness=0→noisiness=0.018, roughness=0.00005→0.266: salto enorme per rumore trascurabile). Nessun effetto sul gap reale (roughness torna comunque a 0.10-0.15 durante training). Lasciato nel codice, innocuo.
5. **Mismatch formula sigmoide/lineare achieved-vs-target** — calcolato con precisione (la sigmoide ha una costante di pendenza 8.0 indipendente da LOW/HIGH, quindi ricalibrare le soglie non basta, serve toccare la costante condivisa da TUTTI i 7 exciter type, rischiando regressione su shaker già pronto). **L'utente ha esplicitamente rifiutato questa strada**, chiedendo di analizzare invece la MECCANICA di generazione del suono di BowExciter.
6. **hf_roughness** (seconda banda di rumore scorrelata da f0, tilt fisso verso l'alto) — aggiunta in una sessione PRECEDENTE (non in questa) per il problema `band_crest_3000-20000` (allora mean=23.9 max=136.5): il roughness esistente non può strutturalmente coprire 3-20kHz per f0 bassi. Ancora attiva.
7. **_jitter_proxy** — proxy differenziabile del jitter (dispersione spettrale dei parziali, teoria Cramer-Rao/Rife&Boorstyn), aggiunto in loss SOLO per bow in una sessione precedente. Nota: un tentativo precedente di jitter temporale (freq/ampiezza) fu RIMOSSO perché non collegato a nessun termine di loss e misurato peggiorare i gap.
8. **FIX VINCENTE — tremolo/AM post-resonatore** (questa sessione): il modulatore di ampiezza `mod_depth`/`mod_rate` (tremolo/vibrato, tecnica esecutiva intrinseca dell'arco, confermato dall'utente non essere un edge case) veniva applicato PRIMA del resonatore. Con Q fino a 600 (ring time ≈0.8s, molto più lento del tremolo a 6Hz≈167ms) il resonatore smussava quasi del tutto la modulazione. Fix: applicare l'inviluppo DOPO `modal(tone)`, e sostituire la sinusoide liscia con un inviluppo "seno raddrizzato a potenza 4" (`|sin(π·rate·t)|^4`), che imita i transienti di cambio d'arco reali. Misurato: flux 5x più alto post- vs pre-resonatore; l'inviluppo a potenza 4 da solo supera il target di flux senza aggiungere rumore.
   **Risultato su tutto il dataset (n=40, seed=0)**: gap harmonicity/noisiness 0.138→0.100 (-28% relativo). Confermato non essere un fluke da 3 file. Resta comunque 2.5x peggio di shaker (0.040).

# Stato attuale — punto esatto di ripartenza

Dopo il fix tremolo, il full stress test n=40 mostra alcuni descrittori MAI-in-loss (diagnostici, per costruzione non differenziabili: peak-picking/partial-tracking discreto) con outlier:
- `band_crest_3000-20000` abs_err: mean=18.959 max=107.889 — VERIFICATO non essere una regressione del fix tremolo: un run precedente (pre-hf_roughness) misurava mean=23.9 max=136.5, quindi il numero attuale è già un miglioramento su un problema preesistente noto.
- `n_partials_mean` rel_err: mean=3.954 max=59.633 — nessun baseline pre-fix documentato in codice, non ancora chiarito se è preesistente o introdotto dal fix.
- `jitter` rel_err: mean=0.849 max=18.972 — genuinamente mai-in-loss nella misura di stress test (AnalyzerV3 su audio reso), anche se un suo PROXY differenziabile è già in loss per bow — quindi informativo ma non detto causato dal fix.
- File peggiore su harmonicity: `Cb-ord-G2-ff-4c-N.wav` (tecnica "ord"/ordinario, NON tremolo) — mostrava un pattern di instabilità distinto nel log per-iterazione (loss che salta a 5.28, rolloff bloccato su un valore esatto per molte iterazioni) — non ancora diagnosticato.

**Script diagnostico preparato e già consegnato sul Mac** (`~/Desktop/ClaudeSynth/mod_depth_correlate.py`, non ancora eseguito dall'utente): riallena bow sullo stesso campione (seed=0, n=40), stampa per file `mod_depth`/`mod_rate`/`roughness`/`hf_roughness` finali insieme a `band_crest_3000-20000`/`n_partials_mean`/harmonicity gap, separando file tremolo da non-tremolo, e calcola la correlazione tra `mod_depth` appreso e questi due outlier. Ipotesi da verificare: l'inviluppo rettificato^4 (transienti ripidi in ampiezza) potrebbe iniettare splatter spettrale alto anche su file dove il tremolo non è la tecnica giusta (es. "ord"), se il gradiente spinge comunque `mod_depth` in alto perché aiuta localmente la metrica di noisiness.

**Prossimo passo immediato**: l'utente deve lanciare `python3 mod_depth_correlate.py` nel suo terminale Mac (dentro `~/Desktop/ClaudeSynth`) e incollare l'output. In base alla correlazione: se alta → il fix va reso scoped (mod_depth alto solo se necessario/target lo richiede chiaramente, non universale); se bassa → la causa degli outlier è altrove, il fix tremolo resta pulito così com'è, e vanno indagati `n_partials_mean`/`Cb-ord-G2-ff-4c-N.wav` separatamente.

# Ordine di lavoro dopo bow (stabilito dall'utente)

bow → blow → noise → strike/pluck → chaotic.
Note: `blow` condivide `_feedback_synthesize` ma NON ha `mod_rate`/`mod_depth` (il fix tremolo non si applica direttamente), mai ancora stress-testato. `noise` ha dataset n=20 (verificare sufficienza). `strike`/`pluck`: riallenare `resonator_predictor.py` sul dataset completo, ri-testare vs baseline 0.176/0.204. `chaotic`: escluso dal training del predictor, motivo non ancora indagato.

# Convenzioni operative da mantenere

- Modifiche a `physical_agents_train.py` via script patch Python con `assert src.count(old)==1`, trasferiti sul Mac con heredoc dentro `device_bash`, verificati con `ast.parse`, poi rimossi.
- Cancellazione file sul device già abilitata (permesso una tantum già concesso su `/Users/mattia/Desktop/ClaudeSynth`).
- Ogni fix va prima misurato (smoke test 3 file o stress test completo eseguito dall'utente), MAI dichiarato "risolto" solo per ragionamento teorico.
