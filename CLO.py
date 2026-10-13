import time

import numpy as np


def clip_bounds(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


# Clouded Leopard Optimization (CLO)
def CLO(X, obj_fun, lb, ub, T):
    N, dim = X.shape[0], X.shape[1]
    minimize = True
    seed = None
    rng = np.random.default_rng(seed)
    best_fit = float('inf')

    sign = 1 if minimize else -1
    X = clip_bounds(X, lb, ub)

    F = np.array([obj_fun(x) for x in X])
    F_internal = sign * F

    best_idx = np.argmin(F_internal)
    best_pos = X[best_idx].copy()
    best_fit_internal = F_internal[best_idx]
    Convergence_curve = np.zeros((T, 1))

    ct = time.time()
    # Main loop
    for t in range(T):
        for i in range(N):
            xi = X[i].copy()
            fi_internal = F_internal[i]

            # --- Phase 1 ---
            prey_idx = rng.integers(0, N - 1)
            if prey_idx >= i:
                prey_idx += 1
            prey = X[prey_idx]
            f_prey_int = F_internal[prey_idx]

            r_ij = rng.random(dim)
            I_ij = rng.integers(1, 3, size=dim)

            if f_prey_int < fi_internal:
                x1 = xi + r_ij * (prey - I_ij * xi)
            else:
                x1 = xi + r_ij * (xi - I_ij * prey)

            x1 = clip_bounds(x1, lb, ub)
            f1 = obj_fun(x1)
            f1_int = sign * f1
            if f1_int < fi_internal:
                X[i] = x1
                F[i] = f1
                F_internal[i] = f1_int
                fi_internal = f1_int
                xi = x1

            # --- Phase 2 ---
            r_ij2 = rng.random(dim)
            base = lb + r_ij2 * (ub - lb)
            delta = (base / max(1, t)) * (2 * r_ij2 - 1)
            x2 = xi + delta

            x2 = clip_bounds(x2, lb, ub)
            f2 = obj_fun(x2)
            f2_int = sign * f2
            if f2_int < fi_internal:
                X[i] = x2
                F[i] = f2
                F_internal[i] = f2_int
                fi_internal = f2_int

        # Track best
        current_best_idx = np.argmin(F_internal)
        if F_internal[current_best_idx] < best_fit_internal:
            best_fit_internal = F_internal[current_best_idx]
            best_pos = X[current_best_idx].copy()
        Convergence_curve[t] = best_fit
    ct = time.time() - ct
    best_fit = sign * best_fit_internal
    return best_fit, Convergence_curve, best_pos, ct
