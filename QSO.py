import time

import numpy as np


def QSO(X, fobj, lb, ub, max_iter):
    pop_size, dim = X.shape[0], X.shape[1]
    T_init = 0.3
    H_init = 0.4
    N_init = 0.9
    alpha = 0.6
    beta = 0.2
    gamma = 0.2
    rng = np.random.default_rng(5)
    D = rng.random(pop_size)  # drought factors
    w = rng.random(pop_size)  # weights
    fitness = np.array([fobj(x) for x in X])  # fitness values

    Convergence_curve = np.zeros((max_iter, 1))
    ct = time.time()
    for t in range(max_iter):

        # Select Leader (best fitness)
        leader_idx = np.argmin(fitness)
        X_leader = X[leader_idx]
        D_leader = D[leader_idx]
        w_leader = w[leader_idx]

        # --- For each quokka: update using Eq.(1) and Eq.(2)
        for i in range(pop_size):

            rand = rng.random()
            w = w_leader - w[i]
            X = X_leader - X[i]
            norm_X = np.linalg.norm(X)
            D = D_leader - D[i]

            # Eq.(1): Drought update
            term_env = (T_init + H_init) / 0.8 * rand
            term_w = w * rand
            term_pos = (norm_X / (1 + norm_X)) * np.sign(D) if norm_X != 0 else 0.0
            D_new = alpha * term_env + beta * term_w + gamma * term_pos
            D_new = np.clip(D_new, 0.0, 1.0)

            # D[i] = D_new

            # Eq.(2): Position update
            X_new = X[i] + D_new * N_init * (X_leader - X[i])
            X[i] = np.clip(X_new, lb[0, :], ub[0, :])

            # Weight update (small drift)
            w[i] += 0.01 * (rng.random() - 0.5)
            if w[i] < 0: w[i] = 0.0

        fitness = np.array([fobj(x) for x in X])
        best_fit = np.min(fitness)
        best_idx = np.argmin(fitness)
        Convergence_curve[t] = best_fit
    best_fit = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return best_fit, Convergence_curve, best_idx, ct
