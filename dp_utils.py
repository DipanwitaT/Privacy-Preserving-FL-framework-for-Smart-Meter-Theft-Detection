import math
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
import numpy as np

DELTA = 1e-5


def compute_epsilon_opacus_old(sigma, q, rounds, delta: float = DELTA) -> float:
    """
    Legacy wrapper kept for backward-compatibility.
    Uses Opacus' RDP analysis (moments accountant).
    """
    orders = list(range(2, 1000))
    rdp = compute_rdp(
        q=q,
        noise_multiplier=sigma,
        steps=rounds,
        orders=orders,
    )
    eps, best_alpha = get_privacy_spent(
        orders=orders,
        rdp=rdp,
        delta=delta,
    )
    return float(eps)


def compute_epsilon_opacus(sigma, q, rounds, delta: float = DELTA) -> float:
    """
    RDP-based epsilon using Opacus (Moments Accountant style).
    This is equivalent to using the 'rdp'/'ma' mode in compute_epsilon_privacy.
    """
    orders = list(range(2, 1000))
    rdp = compute_rdp(
        q=q,
        noise_multiplier=sigma,
        steps=rounds,
        orders=orders,
    )
    eps, best_alpha = get_privacy_spent(
        orders=orders,
        rdp=rdp,
        delta=delta,
    )
    return float(eps)


def compute_epsilon_privacy(
    mode: str,
    sigma: float,
    q: float,
    rounds: int,
    delta: float = DELTA,
) -> float:
    """
    Unified privacy accountant for different notions of DP.

    Args:
        mode: 'dp', 'rdp', 'zcdp', or 'ma'
        sigma: noise multiplier
        q: sampling rate (e.g., clients_per_round / num_clients)
        rounds: number of rounds / compositions
        delta: target delta

    Returns:
        epsilon (float)
    """
    if sigma == 0:
        return math.inf

    mode = mode.lower()

    # Shared orders for RDP-based analyses
    orders = list(range(2, 128))

    if mode in ("rdp", "ma"):
        # Use Opacus RDP analysis (Moments Accountant)
        rdp = compute_rdp(
            q=q,
            noise_multiplier=sigma,
            steps=rounds,
            orders=orders,
        )
        eps, best_alpha = get_privacy_spent(
            orders=orders,
            rdp=rdp,
            delta=delta,
        )
        return float(eps)

    elif mode == "zcdp":
        # Approximate zCDP for subsampled Gaussian:
        #   rho ≈ rounds * q^2 / (2 σ^2)
        rho = rounds * (q ** 2) / (2.0 * sigma ** 2)
        # Convert zCDP (rho) to (ε, δ)-DP:
        #   ε = rho + 2 sqrt{rho * ln(1/δ)}
        eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        return float(eps)

    elif mode == "dp":
        # Very simple (and loose) strong composition for Gaussian:
        # Each round ~ Gaussian(σ), treated as ε_round ≈ 1 / (2 σ^2)
        # Total ε ≈ rounds * ε_round
        eps_round = 1.0 / (2.0 * sigma ** 2)
        eps = rounds * eps_round
        return float(eps)

    else:
        raise ValueError(f"Unknown privacy mode: {mode}")
