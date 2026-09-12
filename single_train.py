"""
single_train.py - addestra UN solo agente exciter alla volta, invece del
loop completo in physical_agents_train.py su tutti e 7 i tipi: tempi di
iterazione piu' bassi quando si lavora su una singola famiglia (vedi note
di tuning per-exciter accumulate in physical_agents_train.py).

Uso:
    python3 single_train.py strike
    python3 single_train.py pluck --restarts 5
    python3 single_train.py shaker --iters 500 --seed 1
    python3 single_train.py --list

Riproducibilita' del target: random_target() consuma un numero di draw
dall'rng diverso per ogni famiglia (vedi physical_agents_train.py), quindi
allenare "shaker" da solo con un rng fresco NON darebbe lo stesso target
che shaker riceve nel loop completo (li' l'rng e' gia' stato avanzato da
strike/pluck/bow/blow). Per restare confrontabile con gli esiti gia'
raccolti, qui l'rng viene "riavvolto" chiamando random_target() (senza
addestrare nulla) per ogni tipo che in EXCITERS precede quello richiesto,
nello stesso ordine - a parita' di --seed (default 0, lo stesso del loop
completo in physical_agents_train.py) il target ottenuto e' identico.
"""
import argparse

import numpy as np
import torch

from physical_agents_train import (
    EXCITERS, SR_DEFAULT, random_target, train_agent_best,
    _band_isolation_probe, _diagnose_run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exciter_type", nargs="?", choices=list(EXCITERS.keys()),
                         help="strike | pluck | bow | blow | shaker | noise | chaotic")
    parser.add_argument("--seed", type=int, default=0,
                         help="seed torch/numpy (default 0, come il loop completo)")
    parser.add_argument("--restarts", type=int, default=10,
                         help="restart di train_agent_best, ognuno con seed diverso (default 10 - "
                              "la varianza tra restart e' spesso alta, vedi diagnosi finale: piu' "
                              "restart = stima piu' affidabile del best-of-N, non solo piu' tentativi)")
    parser.add_argument("--iters", type=int, default=300,
                         help="iterazioni per restart (default 300, come il loop completo)")
    parser.add_argument("--list", action="store_true", help="elenca i tipi disponibili ed esce")
    args = parser.parse_args()

    if args.list or not args.exciter_type:
        print("Tipi disponibili:", ", ".join(EXCITERS.keys()))
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # riavvolge l'rng sui tipi che precedono quello richiesto, per riprodurre
    # lo stesso target del loop completo a parita' di seed (vedi docstring).
    for etype in EXCITERS:
        if etype == args.exciter_type:
            break
        random_target(rng, exciter_type=etype)

    target, f0 = random_target(rng, exciter_type=args.exciter_type)
    restart_losses = []
    exciter, modal, achieved, loss = train_agent_best(
        args.exciter_type, target, f0=f0, restarts=args.restarts,
        log=print, iters=args.iters, seed=args.seed,
        on_restart=lambda r, l: restart_losses.append(l),
    )
    print(f"  target={ {k: round(v, 3) for k, v in target.items()} }")
    print(f"  achieved={ {k: round(v, 3) for k, v in achieved.items()} }")
    print(f"  loss finale={loss:.4f}")

    _diagnose_run(args.exciter_type, achieved, target, modal, f0,
                  restart_losses=restart_losses, log=print)
    _band_isolation_probe(args.exciter_type, exciter, modal, f0, seconds=1.0, sr=SR_DEFAULT, target=target)


if __name__ == "__main__":
    main()
