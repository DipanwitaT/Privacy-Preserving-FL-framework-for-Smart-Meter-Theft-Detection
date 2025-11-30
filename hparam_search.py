from dp_utils import compute_epsilon_opacus

def find_min_sigma(target_eps, clients_per_round, rounds, num_clients, sigma_lo=0.5, sigma_hi=5.0, tol=0.01, max_iters=20):
    q = clients_per_round / float(num_clients)
    eps_hi = compute_epsilon_opacus(sigma_hi, q, rounds)
    if eps_hi > target_eps:
        return None
    eps_lo = compute_epsilon_opacus(sigma_lo, q, rounds)
    if eps_lo <= target_eps:
        return sigma_lo
    lo = sigma_lo
    hi = sigma_hi
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        eps_mid = compute_epsilon_opacus(mid, q, rounds)
        if eps_mid <= target_eps:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi
