import numpy as np
import time


def exploration_phase(X, F, N, m):
    """
    Phase 1: Selecting, Attacking, and Pecking at Seabirds Carrying Prey (Exploration)
    Eq. (4), (5), (6)
    """
    new_X = np.copy(X)
    for i in range(N):
        CSi = [X[k] for k in range(N) if F[k] < F[i]]  # Eq. (4)
        if CSi:
            SSi = CSi[np.random.randint(len(CSi))]
            r = np.random.rand(m)
            I = np.random.choice([1, 2], size=m)
            new_X[i] = X[i] + (1 - 2 * r) * (SSi - I * X[i])  # Eq. (5)
    return new_X


def exploitation_phase(X, best_X, t, N, m):
    """
    Phase 2: Diving Towards Dropped Prey (Exploitation)
    Eq. (7), (8)
    """
    new_X = np.copy(X)
    r = np.random.rand(N, m)
    new_X += (1 - 2 * r) * (best_X - X) / t  # Eq. (7)
    return new_X


def MFO(X, objective_function, VRmin, VRmax, T):
    """
    Magnificent Frigatebird Optimization algorithm
    """

    N, m = X.shape[0], X.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    F = objective_function(X[:])
    best_idx = np.argmin(F)
    best_X = X[best_idx]
    Convergence_curve = np.zeros((T, 1))
    best_value = float('inf')
    best_solution = np.zeros((m, 1))

    t = 0
    ct = time.time()
    for t in range(1, T + 1):
        X_new = exploration_phase(X, F, N, m)
        X_new = np.clip(X_new, lb, ub)
        F_new = objective_function(X[:])
        X[F_new < F] = X_new[F_new < F]  # Eq. (6)
        F[F_new < F] = F_new[F_new < F]

        X_new = exploitation_phase(X, best_X, t, N, m)
        X_new = np.clip(X_new, lb, ub)
        F_new = objective_function(X[:])
        X[F_new < F] = X_new[F_new < F]  # Eq. (8)
        F[F_new < F] = F_new[F_new < F]

        best_idx = np.argmin(F)
        best_X = X[best_idx]

        Convergence_curve[t] = best_X[t]
    best_value = Convergence_curve[T - 1][0]
    ct = time.time() - ct
    return best_value, Convergence_curve, best_solution, ct
