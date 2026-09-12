"""Diagnosi one-shot: perche' alcuni file pluck restano bit-identici per
300 iterazioni anche con l'anchor su q_raw attivo (vedi batch n=40:
pluck gap medio 0.204, peggio di strike 0.176, con piu' file totalmente
congelati). Ipotesi: punto fisso a gradiente zero SIMULTANEO su tutti i
parametri a it=0 - non solo q_raw. Misura diretta (.grad.norm()), non
assunzioni, come per la diagnosi precedente su Va-legno/Hn-slap.

Uso: python3 probe_pluck_freeze.py <path_wav> [exciter_type]
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import physical_agents_train as T
from synth_torch_before import SR_DEFAULT, analyze_reference

path = sys.argv[1] if len(sys.argv) > 1 else "sample/pluck/Guitar/sul_ponticello/Gtr-pont-A3-mf-4c-N.wav"
exciter_type = sys.argv[2] if len(sys.argv) > 2 else "pluck"
seconds = 1.0

target, f0 = analyze_reference(path, seconds=seconds, sr=SR_DEFAULT)
print(f"file={path}  f0={f0:.1f}  target.harmonicity={target.get('harmonicity'):.3f}  target.noisiness={target.get('noisiness'):.3f}")

exciter = T.EXCITERS[exciter_type](SR_DEFAULT)
if hasattr(exciter, "warm_start"):
    exciter.warm_start(target)
n_modes = 12
use_decay_env = exciter_type in ("strike", "pluck", "shaker")
use_tail_noise = exciter_type in ("strike", "pluck")
modal = T.ModalBank(SR_DEFAULT, n_modes=n_modes, f0_init=f0, use_decay_envelope=use_decay_env,
                     use_coupling=False, use_legacy_filter=False, use_tail_noise=use_tail_noise)
tone_amp_exponent = None
modal.warm_start(target, f0=f0, is_feedback=False, exciter_type=exciter_type, tone_amp_exponent=tone_amp_exponent)
try:
    from resonator_predictor import predict_and_apply
    predict_and_apply(exciter_type, exciter, modal, target, f0)
    print("predictor: applicato")
except Exception as e:
    print(f"predictor: non applicato ({e})")

n = int(round(seconds * SR_DEFAULT))

with torch.no_grad():
    y0 = T._limiter(modal(exciter(n)))
print(f"warm-start audio: peak={float(y0.abs().max()):.4f}  rms={float((y0**2).mean().sqrt()):.4f}")

achieved0 = T.descriptor_vector_torch(y0, SR_DEFAULT)
print("\n-- raw error pre-tanh (soglia saturazione tanh ~ TERM_CLAMP*7 = %.0f) --" % (T.TERM_CLAMP * 7))
n_sat = 0
for k, v in target.items():
    if k not in achieved0:
        continue
    vv = float(v)
    w = T.LOSS_WEIGHTS.get(k, 1.0)
    if k == "flux":
        raw = w * ((float(achieved0[k]) - vv) / (vv + T.FLUX_NORM_FLOOR)) ** 2
    elif k in T.RATIO_KEYS:
        raw = w * (torch.log2((achieved0[k].detach() + T.EPS) / (vv + T.EPS)) ** 2).item()
    else:
        raw = w * (float(achieved0[k]) - vv) ** 2
    sat = raw > T.TERM_CLAMP * 7
    n_sat += int(sat)
    av = achieved0[k]
    av = float(av) if not torch.is_tensor(av) else float(av)
    print(f"  {k:16s} achieved={av:.4f}  target={vv:.4f}  raw_err={raw:.2f}{'  <-- SATURO' if sat else ''}")
print(f"termini saturi: {n_sat}")

for name, p in modal.named_parameters():
    p.requires_grad_(True)
for name, p in exciter.named_parameters():
    p.requires_grad_(True)

y = T._limiter(modal(exciter(n)))
achieved = T.descriptor_vector_torch(y, SR_DEFAULT)
terms = T._weighted_loss_terms(achieved, target)
loss = T._aggregate_loss(terms)
gp_loss = T._gain_profile_loss(modal)
fd_loss = T._freq_drift_loss(modal)
qa_loss = T._q_anchor_loss(modal)
mr_loss = T._mode_repulsion_loss(modal) if exciter_type in ("strike", "pluck") else None
print(f"\nloss_descrittori={float(loss):.4f}  gp_loss={None if gp_loss is None else float(gp_loss):.6f}"
      f"  fd_loss={None if fd_loss is None else float(fd_loss):.6f}"
      f"  q_anchor_loss={None if qa_loss is None else float(qa_loss):.6f}"
      f"  mr_loss={None if mr_loss is None else float(mr_loss):.6f}")
loss_total = loss
if gp_loss is not None:
    loss_total = loss_total + T.GAIN_PROFILE_WEIGHT * gp_loss
if fd_loss is not None:
    loss_total = loss_total + T.FREQ_DRIFT_WEIGHT * fd_loss
if qa_loss is not None:
    loss_total = loss_total + T.Q_ANCHOR_WEIGHT * qa_loss
if mr_loss is not None:
    loss_total = loss_total + T.MODE_REPULSION_WEIGHT * mr_loss
loss_total.backward()

print("\n-- gradienti (norma) per parametro, SENZA alcun freeze di warmup --")
for name, p in list(modal.named_parameters()) + [(f"exciter.{n_}", p_) for n_, p_ in exciter.named_parameters()]:
    g = p.grad
    if g is None:
        print(f"  {name:20s} grad=None")
    else:
        print(f"  {name:20s} norm={float(g.norm()):.3e}  max_abs={float(g.abs().max()):.3e}")
