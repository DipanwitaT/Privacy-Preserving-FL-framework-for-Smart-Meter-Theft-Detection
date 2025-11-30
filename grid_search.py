import os
import argparse

import numpy as np
import pandas as pd

from data_loader import load_and_preprocess, create_clients, create_clients_dirichlet
from fl_core import dp_federated_training, evaluate_state_metrics
from dp_utils import DELTA
from plots import (
    plot_perf_vs_priv,
    plot_metrics_bar,
    plot_priv_vs_comm,
    plot_util_vs_comm,
    plot_priv_vs_f1,
)

# Default paths / constants (adapt paths as you like)
DATA_PATH = os.getenv("SGCC_DATA_PATH", "C:/Users/HP/SmartmeterPrivcy_full_final_grid/data set.csv")
OUT_DIR = os.getenv("SGCC_OUT_DIR", "C:/Users/HP/SmartmeterPrivcy_full_final_grid/opacus_results_grid")
os.makedirs(OUT_DIR, exist_ok=True)


def run_grid_search(args):
    # ---------------------
    # Load data once
    # ---------------------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess(DATA_PATH)
    input_dim = X_train.shape[1]
    num_classes = int(y_train.max() + 1)

    # ---------------------
    # Build clients once (depends only on partition, alpha, num_clients)
    # ---------------------
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

    # ---------------------
    # Parse lists
    # ---------------------
    sigma_values = [float(s) for s in args.sigmas.split(",") if s.strip() != ""]
    sigma_values = [s for s in sigma_values if s >= 0.0]

    algo_list = [a.strip().lower() for a in args.algos.split(",") if a.strip() != ""]
    dp_list = [d.strip().lower() for d in args.dps.split(",") if d.strip() != ""]

    print(f"[INFO] Partition={args.partition}, alpha={args.alpha}")
    print(f"[INFO] Algorithms={algo_list}")
    print(f"[INFO] DP modes={dp_list}")
    print(f"[INFO] Sigmas={sigma_values}")

    records = []

    # Small train subset for quick metric check
    X_train_small = X_train[: min(2000, X_train.shape[0])]
    y_train_small = y_train[: min(2000, y_train.shape[0])]

    # ---------------------
    # Grid search loops: algo × dp × sigma
    # ---------------------
    for algo in algo_list:
        for dp_mode in dp_list:
            for sigma in sigma_values:
                print("=" * 70)
                print(
                    f"[GRID] algo={algo}, dp_mode={dp_mode}, "
                    f"sigma={sigma}, partition={args.partition}"
                )

                # Run federated training with these settings
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
                    dp_mode=dp_mode,
                    algo=algo,
                    mu=args.mu,  # only used if algo='fedprox'
                )

                global_model = train_out["global_model"]
                eps_final = train_out["epsilon"]
                comm_bits = train_out["total_comm_bits"]

                state_dict = global_model.state_dict()

                # Evaluate metrics
                train_metrics = evaluate_state_metrics(
                    state_dict, X_train_small, y_train_small
                )
                val_metrics = evaluate_state_metrics(state_dict, X_val, y_val)
                test_metrics = evaluate_state_metrics(state_dict, X_test, y_test)

                print(
                    f"[RESULT] algo={algo}, dp_mode={dp_mode}, sigma={sigma}, "
                    f"eps_final={eps_final:.4f}"
                )
                print(f"[RESULT] test metrics: {test_metrics}")
                print(
                    f"[RESULT] communication: {comm_bits / (1024 ** 2):.3f} MB"
                )

                # Collect a record for master CSV
                record = {
                    "algo": algo,
                    "dp_mode": dp_mode,
                    "sigma": sigma,
                    "eps_final": eps_final,
                    "comm_bytes": comm_bits / 8.0,  # bits → bytes
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
    # Master CSV
    # ---------------------
    df_master = pd.DataFrame(records)

    # One big CSV for everything in this run
    master_path = os.path.join(
        OUT_DIR,
        f"master_results_partition-{args.partition}_alpha-{args.alpha}.csv",
    )
    df_master.to_csv(master_path, index=False)
    print(f"\n[INFO] Saved master results to {master_path}")

    # ---------------------
    # Plots for the whole grid (all algos & dps)
    # ---------------------
    # Privacy–Utility (ε vs test_acc)
    plot_perf_vs_priv(
        df_master, os.path.join(OUT_DIR, "master_privacy_utility.png")
    )
    # Privacy–Communication (ε vs comm)
    plot_priv_vs_comm(
        df_master, os.path.join(OUT_DIR, "master_privacy_communication.png")
    )
    # Utility–Communication (test_acc vs comm)
    plot_util_vs_comm(
        df_master, os.path.join(OUT_DIR, "master_utility_communication.png")
    )
    # Privacy–F1
    plot_priv_vs_f1(
        df_master, os.path.join(OUT_DIR, "master_privacy_f1.png")
    )

    # ---------------------
    # Also, for convenience, save per (algo, dp_mode) CSVs & plots
    # ---------------------
    for algo in algo_list:
        for dp_mode in dp_list:
            mask = (df_master["algo"] == algo) & (df_master["dp_mode"] == dp_mode)
            df_sub = df_master[mask].copy()
            if df_sub.empty:
                continue

            fname = f"results_{algo}_{dp_mode}_{args.partition}.csv"
            sub_path = os.path.join(OUT_DIR, fname)
            df_sub.to_csv(sub_path, index=False)
            print(f"[INFO] Saved subset results to {sub_path}")

            # Per-(algo, dp) plots
            prefix = f"{algo}_{dp_mode}_{args.partition}"
            plot_perf_vs_priv(
                df_sub,
                os.path.join(OUT_DIR, f"privacy_utility_{prefix}.png"),
            )
            plot_priv_vs_comm(
                df_sub,
                os.path.join(OUT_DIR, f"privacy_communication_{prefix}.png"),
            )
            plot_util_vs_comm(
                df_sub,
                os.path.join(OUT_DIR, f"utility_communication_{prefix}.png"),
            )
            plot_priv_vs_f1(
                df_sub,
                os.path.join(OUT_DIR, f"privacy_f1_{prefix}.png"),
            )

            # Bar plot for the best config in this subset (max test_acc)
            best_idx = df_sub["test_acc"].idxmax()
            best_row = df_sub.loc[best_idx]
            best_metrics = {
                "loss": best_row["test_loss"],
                "accuracy": best_row["test_acc"],
                "precision": best_row["test_precision"],
                "recall": best_row["test_recall"],
                "f1": best_row["test_f1"],
            }
            plot_metrics_bar(
                best_metrics,
                os.path.join(OUT_DIR, f"best_metrics_{prefix}.png"),
            )

    print("\n[INFO] Grid search completed.")
    print(df_master.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search over {algo} × {dp} × {sigma} for DP-FL on SGCC"
    )

    # Partition settings (IID vs Non-IID Dirichlet)
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

    # FL algorithm list
    parser.add_argument(
        "--algos",
        type=str,
        default="fedavg,fedprox,scaffold",
        help='Comma-separated list of FL algorithms (subset of "fedavg,fedprox,scaffold")',
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx proximal term coefficient (used when algo includes fedprox)",
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
        help='Comma-separated list of noise multipliers, e.g. "0.0,0.5,1.0,2.0"',
    )
    parser.add_argument(
        "--dps",
        type=str,
        default="dp,rdp,zcdp,ma",
        help='Comma-separated list of DP modes (subset of "dp,rdp,zcdp,ma")',
    )

    args = parser.parse_args()
    run_grid_search(args)
