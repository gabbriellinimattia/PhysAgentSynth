"""resonator_predictor.py - rete piccola che impara a predire i parametri
del banco modale + exciter da un vettore di descrittori target, invece di
risolvere un'ottimizzazione da zero per ogni target come fa oggi train_agent
(vedi analisi: stessa idea di base di DDSP/fader-network - un predittore
addestrato UNA VOLTA su molti esempi, non un fitting per-istanza - ma qui
il predittore NON genera audio, corregge il warm_start ANALITICO gia'
esistente in physical_agents_train.py con un DELTA (residual learning): la
fisica di base resta il prior, la rete impara solo dove l'euristica
sbaglia. Motivazione diretta: il mode-collapse di pluck (vedi log utente -
il training collassa su 2 modi vicini invece di distribuire l'energia come
richiesto dal target) e' un minimo locale del warm start+gradient descent
per-istanza, non risolvibile alzando iterazioni/restart all'infinito.

Loss primaria: L2 sui PARAMETRI verso le soluzioni convergenti di
train_agent_best (vedi predictor_dataset_gen.py) - distillazione, non
spectral matching: il predittore impara la distribuzione di soluzioni
FISICAMENTE PLAUSIBILI osservate (incluse quelle da sample reali,
Orchidea SOL compresa, anche fortemente inarmoniche - multifonici,
tam-tam graffiato), non insegue un target numerico esatto.
Loss secondaria (peso basso, opzionale): sui descrittori raggiunti dal
render risultante, per restare "vicino" al target senza spectral matching
diretto - vedi train_predictor().

Ambito: strike/pluck/shaker/noise/chaotic. bow/blow esclusi: n_modes varia
con f0 (_n_harm_feedback, 6-24) quindi non hanno una rappresentazione a
lunghezza fissa comoda per un MLP; inoltre il loro warm start (freq
congelata sulle armoniche esatte) converge gia' bene analiticamente (log
utente: loss~0.23-0.31, nessun mode-collapse osservato) - meno bisogno di
un predittore li'. Gli altri tipi hanno gia' freq_raw libero (non
congelato): sono gia' in grado di rappresentare spettri inarmonici, il
predittore li aiuta solo a TROVARLI senza cadere in un minimo locale.
chaotic (nuovo, vedi ChaoticExciter/ModalBank.use_coupling in
physical_agents_train.py) e' in ambito per costruzione (stesso layout a
lunghezza fissa) ma non ancora allenabile finche' predictor_dataset.json
non contiene esempi "chaotic" (predictor_dataset_gen.py, cartella
sample/inharmonic) - load_predictor torna None finche' il checkpoint non
esiste, nessun errore nel frattempo.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from physical_agents_train import EXCITERS, ModalBank, _limiter, _weighted_loss_terms, _aggregate_loss
from synth_torch_before import descriptor_vector_torch, SR_DEFAULT

SUPPORTED_TYPES = ("strike", "pluck", "shaker", "noise", "chaotic")

# ordine fisso per il vettore di input del predittore. inharmonicity/
# roughness (target solo per strike, vedi random_target in
# physical_agents_train.py) esclusi di proposito: servono al fine-tuning a
# valle (train_agent), non servono al predittore per un buon warm start,
# e non sono disponibili per gli esempi bow/blow (qui comunque fuori
# ambito) ne' per tutti i file reali (dipendono da AnalyzerV3).
TARGET_KEYS = ("harmonicity", "noisiness", "centroid", "rolloff", "e_low",
               "e_mid", "e_high", "flux", "spread", "t_centroid")


def _modal_param_names(modal):
    names = ["freq_raw", "gain_raw", "q_raw"]
    if modal.use_decay_envelope:
        names.append("tau_raw")
    if getattr(modal, "use_coupling", False):
        names += ["coupling_raw", "coupling_threshold_raw"]
    return names


def _new_modules(exciter_type, sr=SR_DEFAULT, n_modes=12):
    exciter = EXCITERS[exciter_type](sr)
    # use_legacy_filter allineato a train_agent (physical_agents_train.py):
    # SOLO noise/chaotic tengono il vecchio filtro H(0)=1 - vedi nota li'.
    # Senza questo allineamento il predittore imparerebbe una mappa
    # target->warm-start calibrata su una fisica diversa da quella usata
    # davvero in training per pluck/strike, invalidando il predittore.
    modal = ModalBank(sr, n_modes=n_modes, use_decay_envelope=(exciter_type in ("strike", "pluck")),
                       use_coupling=(exciter_type == "chaotic"),
                       use_legacy_filter=(exciter_type in ("noise", "chaotic")))
    return exciter, modal


def flatten_params(exciter, modal):
    """Concatena i parametri di exciter + modal in un unico vettore, in un
    ordine deterministico (named_parameters() di nn.Module e' stabile per
    costruzione, segue l'ordine di assegnazione in __init__) - stessa
    funzione usata sia per generare il dataset (predictor_dataset_gen.py)
    sia per applicare le predizioni (apply_delta sotto): un solo posto dove
    il layout puo' rompersi se si aggiunge un parametro a un exciter."""
    parts = [p.detach().reshape(-1) for _, p in exciter.named_parameters()]
    for name in _modal_param_names(modal):
        parts.append(getattr(modal, name).detach().reshape(-1))
    return torch.cat(parts)


def param_size(exciter_type, sr=SR_DEFAULT, n_modes=12):
    exciter, modal = _new_modules(exciter_type, sr, n_modes)
    return flatten_params(exciter, modal).numel()


def apply_delta(exciter, modal, delta):
    """Applica il delta predetto (stesso ordine di flatten_params) come
    CORREZIONE additiva ai parametri gia' inizializzati da warm_start -
    residual learning: usata a inferenza, PRIMA del fine-tuning con
    train_agent (vedi physical_agents_train.py). In-place/no_grad: qui
    serve solo a impostare un punto di partenza migliore, non a
    retropropagare (per quello vedi _functional_render sotto, usato in
    train_predictor)."""
    i = 0
    with torch.no_grad():
        for _, p in exciter.named_parameters():
            n = p.numel()
            p.add_(delta[i:i + n].reshape(p.shape))
            i += n
        for name in _modal_param_names(modal):
            p = getattr(modal, name)
            n = p.numel()
            p.add_(delta[i:i + n].reshape(p.shape))
            i += n


def _functional_render(exciter_type, exciter, modal, delta, seconds, sr):
    """Come apply_delta, ma DIFFERENZIABILE rispetto a delta (usata solo in
    train_predictor per la loss secondaria sui descrittori): invece di
    mutare i Parameter in-place (che spezzerebbe il grafo), usa
    torch.func.functional_call per eseguire exciter/modal con un dizionario
    di tensori sostitutivi (baseline warm-startata + delta), senza mai
    scrivere su exciter/modal stessi."""
    names_e = [n for n, _ in exciter.named_parameters()]
    base_e = {n: p.detach() for n, p in exciter.named_parameters()}
    names_m = _modal_param_names(modal)
    base_m = {n: getattr(modal, n).detach() for n in names_m}
    sizes = [base_e[n].numel() for n in names_e] + [base_m[n].numel() for n in names_m]
    parts = list(torch.split(delta, sizes))
    new_e = {n: (base_e[n] + parts[i].reshape(base_e[n].shape)) for i, n in enumerate(names_e)}
    off = len(names_e)
    new_m = {n: (base_m[n] + parts[off + i].reshape(base_m[n].shape)) for i, n in enumerate(names_m)}

    n_samp = int(round(seconds * sr))
    exc_out = functional_call(exciter, new_e, (n_samp,))
    y = functional_call(modal, new_m, (exc_out,))
    return _limiter(y)


def _target_vector(target, f0):
    vals = [float(target.get(k, 0.0)) for k in TARGET_KEYS]
    vals.append(float(torch.log(torch.tensor(max(float(f0), 1.0)))))   # log(f0): stessa scala log usata in freq_raw
    return torch.tensor(vals, dtype=torch.float32)


class ResonatorPredictor(nn.Module):
    """MLP piccola (2 hidden layer) - un'istanza per exciter_type (layout
    parametri diverso per tipo, vedi flatten_params): il dataset di
    distillazione (le dimensioni di sample/, non milioni di esempi) non
    giustifica niente di piu' grande."""
    def __init__(self, exciter_type, hidden=96):
        super().__init__()
        self.exciter_type = exciter_type
        in_dim = len(TARGET_KEYS) + 1
        out_dim = param_size(exciter_type)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        # init vicino a zero: prima di allenare il predittore, il delta e'
        # ~0 -> si comporta come il warm_start puro di oggi, nessuna
        # regressione finche' non converge.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, target, f0, mean=None, std=None):
        x = _target_vector(target, f0)
        if mean is not None:
            x = (x - mean) / (std + 1e-6)
        return self.net(x)


def train_predictor(exciter_type, dataset_path="predictor_dataset.json", epochs=300, lr=1e-3,
                     descriptor_weight=0.1, descriptor_subsample=8, seconds=1.0, sr=SR_DEFAULT,
                     out_path=None, log=print):
    """Allena il predittore per UN exciter_type sul dataset generato da
    predictor_dataset_gen.py. Loss = MSE sui parametri (primaria,
    distillazione) + descriptor_weight * loss sui descrittori raggiunti dal
    render (secondaria, sottocampionata a descriptor_subsample esempi per
    epoca - un render+STFT per esempio e' il passo costoso, non serve farlo
    su tutto il dataset ad ogni epoca per un termine di solo affinamento)."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)[exciter_type]
    if not data:
        raise ValueError(f"nessun esempio per {exciter_type} in {dataset_path}")

    targets = [d["target"] for d in data]
    f0s = [d["f0"] for d in data]
    final_params = torch.tensor([d["params"] for d in data], dtype=torch.float32)
    X = torch.stack([_target_vector(t, f0) for t, f0 in zip(targets, f0s)])
    x_mean, x_std = X.mean(0), X.std(0)

    # target di regressione = DELTA rispetto al warm_start, non i parametri
    # assoluti: apply_delta somma la predizione a un warm_start gia' fatto
    # (vedi predict_and_apply), quindi il predittore deve imparare quello
    # scarto, non il valore finale. BUG trovato in test: regredire
    # direttamente sui parametri assoluti fa dominare la loss a tau_raw
    # (init ~1000, MSE~1e5 solo da li') mentre freq/gain (scala 0.1-10)
    # restano invisibili nel gradiente. Calcolato una volta sola (warm_start
    # e' deterministico dato target/f0/exciter_type, nessun bisogno di
    # rifarlo ad ogni epoca).
    baselines = []
    for t, f0 in zip(targets, f0s):
        exciter, modal = _new_modules(exciter_type, sr)
        modal.warm_start(t, f0=f0, is_feedback=False, exciter_type=exciter_type)
        baselines.append(flatten_params(exciter, modal))
    baseline = torch.stack(baselines)
    delta_target = final_params - baseline
    # normalizzazione anche in USCITA (non solo in input): senza, anche dopo
    # il fix del delta, dimensioni con scala naturale diversa (es. tau_raw
    # se il target richiede un decadimento molto diverso dall'inerte,
    # rispetto a freq_raw che si muove di frazioni di ottava) continuerebbero
    # a dominare l'MSE sproporzionatamente.
    y_mean, y_std = delta_target.mean(0), delta_target.std(0)
    y_std = torch.where(y_std > 1e-6, y_std, torch.ones_like(y_std))
    delta_norm = (delta_target - y_mean) / y_std

    model = ResonatorPredictor(exciter_type)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(data)
    for ep in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(n)
        pred = torch.stack([model(targets[i], f0s[i], x_mean, x_std) for i in idx])
        loss_param = F.mse_loss(pred, delta_norm[idx])
        loss = loss_param

        if descriptor_weight > 0:
            sub = idx[:max(1, min(descriptor_subsample, n))]
            desc_terms = []
            for i in sub:
                exciter, modal = _new_modules(exciter_type, sr)
                modal.warm_start(targets[i], f0=f0s[i], is_feedback=False, exciter_type=exciter_type)
                delta = model(targets[i], f0s[i], x_mean, x_std) * y_std + y_mean   # de-normalizzato prima di applicarlo
                y = _functional_render(exciter_type, exciter, modal, delta, seconds, sr)
                achieved = descriptor_vector_torch(y, sr)
                desc_terms.append(_aggregate_loss(_weighted_loss_terms(achieved, targets[i])))
            loss = loss + descriptor_weight * torch.stack(desc_terms).mean()

        loss.backward()
        opt.step()
        if log and ep % max(1, epochs // 10) == 0:
            log(f"[{exciter_type}] epoch {ep:4d} loss_param={float(loss_param.detach()):.4f} loss_tot={float(loss.detach()):.4f}")

    out_path = out_path or f"resonator_predictor_{exciter_type}.pt"
    torch.save({"state_dict": model.state_dict(), "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std}, out_path)
    if log:
        log(f"Salvato {out_path} ({n} esempi)")
    return model


_predictor_cache = {}


def load_predictor(exciter_type, path=None):
    """Carica (una volta, cache in memoria) il predittore salvato per
    exciter_type, se esiste - stesso pattern opzionale di
    _load_calibration in physical_agents_train.py: assente -> None, chi
    chiama ricade sul warm_start puro senza errori."""
    if exciter_type in _predictor_cache:
        return _predictor_cache[exciter_type]
    path = path or f"resonator_predictor_{exciter_type}.pt"
    model = None
    try:
        ckpt = torch.load(path, weights_only=False)
        model = ResonatorPredictor(exciter_type)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model._x_mean, model._x_std = ckpt["x_mean"], ckpt["x_std"]
        model._y_mean, model._y_std = ckpt["y_mean"], ckpt["y_std"]
    except FileNotFoundError:
        model = None
    _predictor_cache[exciter_type] = model
    return model


def predict_and_apply(exciter_type, exciter, modal, target, f0):
    """Hook per train_agent (physical_agents_train.py): se un predittore
    allenato esiste per exciter_type, corregge il warm_start gia' eseguito
    su exciter/modal con il delta predetto (de-normalizzato con le
    statistiche salvate in checkpoint - vedi train_predictor). No-op
    silenzioso se il checkpoint non esiste (predittore non ancora
    allenato)."""
    if exciter_type not in SUPPORTED_TYPES:
        return False
    model = load_predictor(exciter_type)
    if model is None:
        return False
    with torch.no_grad():
        delta = model(target, f0, model._x_mean, model._x_std) * model._y_std + model._y_mean
    apply_delta(exciter, modal, delta)
    return True


if __name__ == "__main__":
    import sys
    etype = sys.argv[1] if len(sys.argv) > 1 else "strike"
    train_predictor(etype)
