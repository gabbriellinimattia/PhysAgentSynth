"""Step 2 diagnosi pluck-freeze: q_raw/freq_raw a it=0 hanno gradiente
non-nullo (probe_pluck_freeze.py: q_raw grad norm=4.8e-05, freq_raw
grad norm=2.9e-03) - NON e' la stessa patologia (dead-gradient ~1e-9/1e-11)
gia' risolta per strike. Eppure il batch log riporta valori identici per
tutta la run su questo file. Verifica diretta: i parametri si MUOVONO
davvero iterazione per iterazione? Replica fedele del loop di train_agent
(stessi import/costanti/ordine) ma con stampa per-iterazione dei VALORI
(non solo i gradienti) sulle prime iterazioni.

Uso: python3 probe_pluck_steps.py <path_wav> [exciter_type] [iters]
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import physical_agents_train as T
from synth_torch_before import SR_DEFAULT, analyze_reference

path = sys.argv[1] if len(sys.argv) > 1 else "sample/pluck/Guitar/sul_ponticello/Gtr-pont-A3-mf-4c-N.wav"
exciter_type = sys.argv[2] if len(sys.argv) > 2 else "pluck"
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
seconds = 1.0
sr = SR_DEFAULT
lr = 0.06
noise_samples = 3
torch.manual_seed(0)

target, f0 = analyze_reference(path, seconds=seconds, sr=sr)
print(f"file={path}  f0={f0:.1f}  iters={iters}")

exciter = T.EXCITERS[exciter_type](sr)
if hasattr(exciter, "warm_start"):
    exciter.warm_start(target)
is_feedback = exciter_type in T.FEEDBACK_TYPES
n_modes = 12
use_decay_env = exciter_type in ("strike", "pluck", "shaker")
use_tail_noise = exciter_type in ("strike", "pluck")
modal = T.ModalBank(sr, n_modes=n_modes, f0_init=f0, use_decay_envelope=use_decay_env,
                     use_coupling=False, use_legacy_filter=False, use_tail_noise=use_tail_noise)
modal.warm_start(target, f0=f0, is_feedback=False, exciter_type=exciter_type, tone_amp_exponent=None)
try:
    from resonator_predictor import predict_and_apply
    predict_and_apply(exciter_type, exciter, modal, target, f0)
except Exception:
    pass

n = int(round(seconds * sr))
env_params = [p for name, p in exciter.named_parameters() if name in ("decay_raw", "attack_raw")]
env_param_ids = {id(p) for p in env_params}
exciter_base_params = [p for p in exciter.parameters() if id(p) not in env_param_ids]
other_params = exciter_base_params + [modal.gain_raw]
freq_params = [modal.freq_raw]
tau_params = [modal.tau_raw] if exciter_type == "pluck" else []
if exciter_type != "pluck" and use_decay_env:
    other_params = other_params + [modal.tau_raw]
tail_noise_params = [modal.tail_noise_raw] if use_tail_noise else []

opt = torch.optim.Adam([
    {"params": other_params, "lr": lr},
    {"params": freq_params, "lr": lr},
    {"params": [modal.q_raw], "lr": lr * 2},
    {"params": env_params, "lr": lr * 2},
    {"params": tau_params, "lr": lr * 2},
    {"params": tail_noise_params, "lr": lr},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)  # T_max reale (300), non iters ridotto del probe

q0 = float(F.softplus(modal.q_raw).mean())
print(f"it=-1 (pre)  q_raw.mean(softplus)={q0:.5f}")

for it in range(iters):
    opt.zero_grad()
    loss_acc, achieved = 0.0, {}
    for _ in range(noise_samples):
        y = T._limiter(modal(exciter(n)))
        achieved = T.descriptor_vector_torch(y, sr)
        terms = T._weighted_loss_terms(achieved, target)
        loss_acc = loss_acc + T._aggregate_loss(terms)
        if exciter_type in T.DECAY_SLOPE_TYPES and "decay_slope" in target:
            achieved_slope = T.decay_slope_db_per_band(y, sr)
            ds_err = ((achieved_slope - target["decay_slope"]) / T.DECAY_SLOPE_NORM) ** 2
            loss_acc = loss_acc + T.DECAY_SLOPE_WEIGHT * ds_err.mean()
    loss = loss_acc / noise_samples
    warmup_it = int(300 * T.GAIN_FREQ_WARMUP_FRAC)
    gp_loss = T._gain_profile_loss(modal)
    if gp_loss is not None:
        gp_mult = (T.GAIN_PROFILE_WEIGHT_PLUCK_MULT if exciter_type == "pluck" else
                   T.GAIN_PROFILE_WEIGHT_STRIKE_MULT if exciter_type == "strike" else 1.0)
        loss = loss + T.GAIN_PROFILE_WEIGHT * gp_mult * gp_loss
    fd_loss = T._freq_drift_loss(modal) if not is_feedback else None
    if fd_loss is not None:
        loss = loss + T.FREQ_DRIFT_WEIGHT * fd_loss
    q_anchor_loss = T._q_anchor_loss(modal) if exciter_type in T.DECAY_SLOPE_TYPES else None
    if q_anchor_loss is not None:
        loss = loss + T.Q_ANCHOR_WEIGHT * q_anchor_loss
    mr_loss = T._mode_repulsion_loss(modal) if exciter_type in ("strike", "pluck") else None
    if mr_loss is not None:
        loss = loss + T.MODE_REPULSION_WEIGHT * mr_loss
    loss.backward()
    g = modal.q_raw.grad
    q_grad_str = "None" if g is None else f"{float(g.norm()):.3e}"
    if it < warmup_it:
        modal.gain_raw.grad = None
        if not is_feedback:
            modal.freq_raw.grad = None
    params_all = other_params + freq_params + [modal.q_raw] + env_params + tau_params + tail_noise_params
    total_norm_before = torch.nn.utils.clip_grad_norm_(params_all, 5.0)
    opt.step()
    sched.step()
    q_now = float(F.softplus(modal.q_raw).mean())
    freq_now = float(torch.exp(modal.freq_raw).mean())
    h_now = float(achieved["harmonicity"])
    print(f"it={it:3d}  loss={float(loss):.6f}  h_ach={h_now:.6f}  q_mean={q_now:.6f}  "
          f"freq_mean={freq_now:.3f}  q_grad_norm={q_grad_str}  clip_total_norm={float(total_norm_before):.3e}  "
          f"lr_now={sched.get_last_lr()[0]:.5f}")
