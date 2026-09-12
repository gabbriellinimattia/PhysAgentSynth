"""train_all_predictors.py - allena resonator_predictor.py per tutti i tipi
supportati TRANNE "chaotic" (instabile anche dopo le patch tentate in
sessione a physical_agents_train.py, rimandato - vedi note li').

Usa il predictor_dataset.json gia' generato (predictor_dataset_gen.py).
Un checkpoint per tipo: resonator_predictor_<tipo>.pt (load_predictor in
resonator_predictor.py lo trova gia' con questo nome di default).

Uso: python3 train_all_predictors.py [--epochs 300] [--dataset predictor_dataset.json]
"""
import argparse

from resonator_predictor import train_predictor, SUPPORTED_TYPES

SKIP = {"chaotic"}


def main(dataset_path="predictor_dataset.json", epochs=300, lr=1e-3, descriptor_weight=0.1):
    for etype in SUPPORTED_TYPES:
        if etype in SKIP:
            print(f"-- salto {etype} (vedi nota in testa al file) --")
            continue
        print(f"=== allenamento predittore: {etype} ===")
        train_predictor(etype, dataset_path=dataset_path, epochs=epochs, lr=lr,
                         descriptor_weight=descriptor_weight,
                         out_path=f"resonator_predictor_{etype}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="predictor_dataset.json")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--descriptor-weight", type=float, default=0.1)
    a = p.parse_args()
    main(dataset_path=a.dataset, epochs=a.epochs, lr=a.lr, descriptor_weight=a.descriptor_weight)
