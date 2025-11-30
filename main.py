import os
import argparse

import numpy as np
import pandas as pd

from data_loader import load_and_preprocess, create_clients, create_clients_dirichlet
from fl_core import dp_federated_training, evaluate_state_metrics
from plots import plot_perf_vs_priv, plot_metrics_bar
from dp_utils import DELTA

# Default paths / constants
DATA_PATH = os.getenv("SGCC_DATA_PATH", "C:/Users/HP/SmartmeterPrivcy_full_final_grid/data set.csv")
OUT_DIR = "C:/Users/HP/SmartmeterPrivcy_full_final_grid/opacus_results"
os.makedirs(OUT_DIR, exist_ok=True)


def main(args):
    # ---------------------
    # Load data & partition
    # ---------------------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess(DATA_PATH)

    if args.partition == "non-iid":
        print(f"[INFO] Using non-IID Dirichlet partitioning, alpha={args.alpha}")
        clients = create_clients_dirichlet(
            X_train,
            y_train,
            num_clients=args.num_clients,
            alpha=args.alpha,
            cap=None,
        )
    else:
        print("[INFO] Using IID partitioning.")
        clients = create_clients(
            X_train, y_train, num_clients=args.num_clients, cap=None
        )

    input_dim = X_train.shape[1]
    num_classes = int(y_train.max() + 1)

    # Parse sigma sweep
    sigma_values = [float(s) for s in args.sigmas.split(",")]
    sigma_values = [s for s in sigma_values if s >= 0.0]

    print(
        f"[INFO] algo={args.algo}, dp_mode={args.dp}, "
        f"partition={args.partition}, sigmas={sigma_values}"
    )

    records = []

    # ---------------------
    # Sweep over σ values
    # ---------------------
    for sigma in sigma_values:
        print("=" * 60)
        print(f"[INFO] Training with sigma={sigma}")

        train_out = dp_federated_training(
            input_dim=input_dim,
            num_classes=num_classes,
            clients=clients,
            rounds=args.rounds,
            clients_per_round=args.clients_per_round,
            lr=args.lr,
            local_epochs=args.local_epochs,
            batch_size=args.local_batch,
            clipping_norm=args.clip,
            noise_multiplier=sigma,
            delta=DELTA,
            dp_mode=args.dp,
            algo=args.algo,
            mu=args.mu,
        )

        global_model = train_out["global_model"]
        eps_final = train_out["epsilon"]
        comm_bits = train_out["total_comm_bits"]

        state_dict = global_model.state_dict()

        # Evaluation
        X_train_small = X_train[: min(2000, X_train.shape[0])]
        y_train_small = y_train[: min(2000, y_train.shape[0])]

        train_metrics = evaluate_state_metrics(state_dict, X_train_small, y_train_small)
        val_metrics = evaluate_state_metrics(state_dict, X_val, y_val)
        test_metrics = evaluate_state_metrics(state_dict, X_test, y_test)

        print(f"[RESULT] sigma={sigma}, eps_final={eps_final:.4f}")
        print(f"[RESULT] test metrics: {test_metrics}")
        print(f"[RESULT] communication: {comm_bits / (1024 ** 2):.3f} MB")

        record = {
            "sigma": sigma,
            "eps_final": eps_final,
            "comm_bytes": comm_bits / 8.0,
            "algo": args.algo,
            "dp_mode": args.dp,
            "partition": args.partition,
            "alpha": args.alpha,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "clients_per_round": args.clients_per_round,
            "lr": args.lr,
            "local_epochs": args.local_epochs,
            "local_batch": args.local_batch,
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "test_acc": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        }
        records.append(record)

    # ---------------------
    # Save merged CSV
    # ---------------------
    df_metrics = pd.DataFrame(records)
    csv_path = os.path.join(
        OUT_DIR,
        f"results_{args.algo}_{args.dp}_{args.partition}.csv",
    )
    df_metrics.to_csv(csv_path, index=False)
    print(f"[INFO] Saved metrics (all sigmas) to {csv_path}")

    # ---------------------
    # Plots over sweep
    # ---------------------
    # Privacy–utility trade-off: eps vs test_acc
    plot_perf_vs_priv(df_metrics, os.path.join(OUT_DIR, "perf_vs_priv.png"))

    # Bar plot for metrics of the *best* sigma (e.g., highest test_acc)
    best_idx = df_metrics["test_acc"].idxmax()
    best_row = df_metrics.loc[best_idx]
    best_metrics = {
        "loss": best_row["test_loss"],
        "accuracy": best_row["test_acc"],
        "precision": best_row["test_precision"],
        "recall": best_row["test_recall"],
        "f1": best_row["test_f1"],
    }
    plot_metrics_bar(best_metrics, os.path.join(OUT_DIR, "best_test_metrics_bar.png"))

    print("\nBest configuration over sigmas:")
    print(best_row)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DP-FL on SGCC with multiple accountants & FL algos (sigma sweep)"
    )

    # Partition
    parser.add_argument(
        "--partition",
        choices=["iid", "non-iid"],
        default="iid",
        help="Client data partitioning",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha for non-iid partition",
    )

    # Federated setup
    parser.add_argument(
        "--num_clients", type=int, default=10, help="Total number of clients"
    )
    parser.add_argument(
        "--rounds", type=int, default=20, help="Number of communication rounds"
    )
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=3,
        help="Clients sampled per round",
    )

    # Local training
    parser.add_argument("--lr", type=float, default=0.01, help="Local learning rate")
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local epochs per round",
    )
    parser.add_argument(
        "--local_batch", type=int, default=64, help="Local batch size per client"
    )

    # FL algorithm
    parser.add_argument(
        "--algo",
        choices=["fedavg", "fedprox", "scaffold"],
        default="fedavg",
        help="Federated optimization algorithm",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx proximal term coefficient (only used if algo=fedprox)",
    )

    # DP settings
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="L2 clipping norm for server-side DP",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default="0.0,0.5,1.0,2.0,4.0",
        help='Comma-separated list of noise multipliers, e.g. "0.5,1.0,2.0"',
    )
    parser.add_argument(
        "--dp",
        choices=["dp", "rdp", "zcdp", "ma"],
        default="rdp",
        help="Privacy accountant mode",
    )

    args = parser.parse_args()
    main(args)
