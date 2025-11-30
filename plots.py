import matplotlib.pyplot as plt
import os


def plot_eps_vs_rounds(eps_traj, out_path):
    plt.figure()
    plt.plot(range(1, len(eps_traj) + 1), eps_traj, marker='o')
    plt.xlabel('round')
    plt.ylabel('epsilon')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_perf_vs_priv(df_metrics, out_path):
    """
    Plot utility vs privacy: test_acc vs eps_final.
    Expects columns: 'eps_final', 'test_acc'.
    """
    if "eps_final" not in df_metrics or "test_acc" not in df_metrics:
        return
    plt.figure()
    plt.plot(df_metrics['eps_final'], df_metrics['test_acc'], marker='o')
    plt.xlabel('epsilon')
    plt.ylabel('test_acc')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metrics_bar(metrics_dict, out_path):
    """
    Bar plot for classification metrics: loss, accuracy, precision, recall, f1.
    """
    # Only include these keys if present
    keys = ["loss", "accuracy", "precision", "recall", "f1"]
    names = []
    values = []
    for k in keys:
        if k in metrics_dict:
            names.append(k)
            values.append(metrics_dict[k])

    if not names:
        return

    plt.figure()
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names)
    plt.ylabel("value")
    plt.title("Test metrics")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------------
# New plots for grid-search trade-offs
# ------------------------------------------------------------------

def plot_priv_vs_comm(df_metrics, out_path):
    """
    Privacy vs Communication:
    eps_final (x-axis) vs comm_bytes in MB (y-axis).
    Expects columns: 'eps_final', 'comm_bytes'.
    """
    if "eps_final" not in df_metrics or "comm_bytes" not in df_metrics:
        return

    comm_mb = df_metrics["comm_bytes"] / (1024.0**2)
    plt.figure()
    plt.scatter(df_metrics["eps_final"], comm_mb, marker='o')
    for _, row in df_metrics.iterrows():
        if "sigma" in row:
            label = f"σ={row['sigma']}"
        else:
            label = ""
        plt.annotate(label, (row["eps_final"], row["comm_bytes"] / (1024.0**2)))
    plt.xlabel("epsilon")
    plt.ylabel("communication (MB)")
    plt.title("Privacy–Communication Trade-off")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_util_vs_comm(df_metrics, out_path):
    """
    Utility vs Communication:
    test_acc (y-axis) vs comm_bytes in MB (x-axis).
    Expects columns: 'test_acc', 'comm_bytes'.
    """
    if "test_acc" not in df_metrics or "comm_bytes" not in df_metrics:
        return

    comm_mb = df_metrics["comm_bytes"] / (1024.0**2)
    plt.figure()
    plt.scatter(comm_mb, df_metrics["test_acc"], marker='o')
    for _, row in df_metrics.iterrows():
        if "sigma" in row:
            label = f"σ={row['sigma']}"
        else:
            label = ""
        plt.annotate(label, (row["comm_bytes"] / (1024.0**2), row["test_acc"]))
    plt.xlabel("communication (MB)")
    plt.ylabel("test_acc")
    plt.title("Utility–Communication Trade-off")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_priv_vs_f1(df_metrics, out_path):
    """
    Privacy vs F1:
    f1 (y-axis) vs eps_final (x-axis).
    Expects columns: 'eps_final', 'test_f1'.
    """
    if "eps_final" not in df_metrics or "test_f1" not in df_metrics:
        return

    plt.figure()
    plt.plot(df_metrics["eps_final"], df_metrics["test_f1"], marker='o')
    plt.xlabel("epsilon")
    plt.ylabel("test_f1")
    plt.title("Privacy–F1 Trade-off")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

