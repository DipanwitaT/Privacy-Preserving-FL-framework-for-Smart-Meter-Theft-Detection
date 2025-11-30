import copy
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import get_model
from dp_utils import compute_epsilon_privacy
from sklearn.metrics import precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Helper: flatten / unflatten
# ------------------------------
def get_model_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_model_parameters(model: torch.nn.Module, flat_params: torch.Tensor):
    pointer = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(flat_params[pointer : pointer + num].view_as(p))
        pointer += num


# ------------------------------
# Local training: FedAvg / FedProx
# ------------------------------
def local_train(
    global_model: torch.nn.Module,
    Xc: np.ndarray,
    yc: np.ndarray,
    lr: float,
    local_epochs: int,
    batch_size: int,
    algo: str = "fedavg",
    mu: float = 0.0,
    device=DEVICE,
) -> Tuple[torch.Tensor, int]:
    """
    Local training for a single client.

    - algo='fedavg': standard ERM on local data.
    - algo='fedprox': ERM + proximal term around the initial global model.
    """
    model = copy.deepcopy(global_model).to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dataset = TensorDataset(
        torch.tensor(Xc, dtype=torch.float32),
        torch.tensor(yc, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # For FedProx: capture the initial model parameters
    if algo == "fedprox":
        init_params = [p.detach().clone() for p in model.parameters()]
    else:
        init_params = None

    num_samples = 0

    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            if algo == "fedprox" and init_params is not None:
                prox_term = 0.0
                for p, p0 in zip(model.parameters(), init_params):
                    prox_term = prox_term + ((p - p0.to(device)) ** 2).sum()
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            num_samples += xb.size(0)

    flat_params = get_model_parameters(model).detach().cpu()
    return flat_params, num_samples


# ------------------------------
# Local training: SCAFFOLD
# ------------------------------
def local_train_scaffold(
    global_model: torch.nn.Module,
    Xc: np.ndarray,
    yc: np.ndarray,
    lr: float,
    local_epochs: int,
    batch_size: int,
    c_global: List[torch.Tensor],
    c_client: List[torch.Tensor],
    device=DEVICE,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """
    Local training for SCAFFOLD with simple control variates.
    """
    model = copy.deepcopy(global_model).to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dataset = TensorDataset(
        torch.tensor(Xc, dtype=torch.float32),
        torch.tensor(yc, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_samples = 0
    num_steps = 0

    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # Gradient correction: g' = g - (c_i - c)
            for p, cg, cc in zip(model.parameters(), c_global, c_client):
                if p.grad is not None:
                    p.grad = p.grad - (cc.to(device) - cg.to(device))

            optimizer.step()
            num_samples += xb.size(0)
            num_steps += 1

    # Update local control variate
    w_global = get_model_parameters(global_model).detach()
    w_local = get_model_parameters(model).detach()
    delta_w = (w_global - w_local) / max(num_steps * lr, 1e-8)

    new_c_client: List[torch.Tensor] = []
    pointer = 0
    for cg, cc, p in zip(c_global, c_client, model.parameters()):
        num = p.numel()
        delta = delta_w[pointer : pointer + num].view_as(p.cpu())
        pointer += num
        new_c_client.append(cc + delta)

    flat_params = w_local.cpu()
    return flat_params, num_samples, new_c_client


# ------------------------------
# Federated training with DP
# ------------------------------
def dp_federated_training(
    input_dim: int,
    num_classes: int,
    clients: List[Tuple[np.ndarray, np.ndarray]],
    rounds: int,
    clients_per_round: int,
    lr: float,
    local_epochs: int,
    batch_size: int,
    clipping_norm: float,
    noise_multiplier: float,
    delta: float,
    dp_mode: str,
    algo: str,
    mu: float,
    device=DEVICE,
) -> Dict:
    """
    Federated learning with:
      - FedAvg / FedProx / SCAFFOLD
      - Server-side Gaussian DP
      - Multiple privacy accountants via dp_mode
    """
    num_clients = len(clients)
    client_indices = list(range(num_clients))

    # Initialize global model
    global_model = get_model(input_dim, 128, num_classes, device=device)
    param_dim = sum(p.numel() for p in global_model.parameters())
    bits_per_param = 32

    # For SCAFFOLD
    c_global = None
    c_clients = None
    if algo == "scaffold":
        c_global = [torch.zeros_like(p.data) for p in global_model.parameters()]
        c_clients = [
            [torch.zeros_like(p.data) for p in global_model.parameters()]
            for _ in range(num_clients)
        ]

    total_comm_bits = 0
    total_steps = 0

    sampling_rate = clients_per_round / num_clients

    for r in range(rounds):
        selected = np.random.choice(
            client_indices, size=clients_per_round, replace=False
        )

        # Downlink comm: send global model to each selected client
        downlink_bits = param_dim * bits_per_param * len(selected)

        client_flats = []
        client_weights = []

        for cid in selected:
            Xc, yc = clients[cid]
            if algo == "scaffold":
                flat, n_samples, new_c = local_train_scaffold(
                    global_model,
                    Xc,
                    yc,
                    lr=lr,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    c_global=c_global,
                    c_client=c_clients[cid],
                    device=device,
                )
                c_clients[cid] = new_c
            else:
                flat, n_samples = local_train(
                    global_model,
                    Xc,
                    yc,
                    lr=lr,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    algo=algo,
                    mu=mu,
                    device=device,
                )
            client_flats.append(flat)
            client_weights.append(n_samples)

        # Update global control variate for SCAFFOLD
        if algo == "scaffold" and c_clients is not None:
            for j in range(len(c_global)):
                c_global[j] = sum(c_clients[i][j] for i in range(num_clients)) / num_clients

        client_weights = np.array(client_weights, dtype=np.float64)
        client_weights = client_weights / client_weights.sum()

        stacked = torch.stack(client_flats)  # [clients, param_dim]
        weighted = (stacked * torch.from_numpy(client_weights).view(-1, 1)).sum(dim=0)

        # DP: clip aggregated update and add Gaussian noise
        l2 = torch.norm(weighted)
        if l2 > clipping_norm:
            weighted = weighted * (clipping_norm / (l2 + 1e-12))

        if noise_multiplier > 0:
            noise_std = noise_multiplier * clipping_norm
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=weighted.shape,
            )
            weighted = weighted + noise

        # Set new global model parameters
        set_model_parameters(global_model, weighted)

        # Uplink comm: clients send updates to server
        uplink_bits = param_dim * bits_per_param * len(selected)
        total_comm_bits += (downlink_bits + uplink_bits)

        total_steps += 1

    # Compute epsilon for chosen accountant
    epsilon = compute_epsilon_privacy(
        mode=dp_mode,
        sigma=noise_multiplier,
        q=sampling_rate,
        rounds=total_steps,
        delta=delta,
    )

    return {
        "global_model": global_model,
        "epsilon": float(epsilon),
        "total_comm_bits": float(total_comm_bits),
        "total_steps": int(total_steps),
    }


# ------------------------------
# Evaluation helpers
# ------------------------------
def evaluate_state_metrics(state_dict, X_eval, y_eval, device=DEVICE):
    """
    Evaluate a state_dict on (X_eval, y_eval) and return:
    loss, accuracy, precision, recall, f1 (macro).
    """
    num_classes = int(y_eval.max() + 1)
    model = get_model(X_eval.shape[1], 128, num_classes, device=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(X_eval, dtype=torch.float32).to(device)
        yb = torch.tensor(y_eval, dtype=torch.long).to(device)

        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb).item()
        preds = logits.argmax(dim=1)

        acc = (preds == yb).float().mean().item()

    y_true = yb.cpu().numpy()
    y_pred = preds.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
