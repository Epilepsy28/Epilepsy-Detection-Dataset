import time
import numpy as np


def clip_bounds(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


# Hybrid of Northern Goshawk Optimization (NGO) and Clouded Leopard Optimization (CLO) Starting Line No.50
def Proposed(Search_Agents, objective, Lowerbound, Upperbound, Max_iterations):
    N, dimensions = Search_Agents.shape[0], Search_Agents.shape[1]
    X = Search_Agents
    fit = np.zeros(Search_Agents)  # Fitness values
    fit_new = np.zeros(Search_Agents)  # New fitness values
    NGO_curve = np.zeros(Max_iterations)  # To store the curve of the best score

    # Initial fitness evaluation
    for i in range(Search_Agents):
        fit[i] = objective(X[i, :])

    # Main loop for iterations
    ct = time.time()
    for t in range(Max_iterations):
        # Find the best solution so far
        best = np.min(fit)
        blocation = np.argmin(fit)

        if t == 1:
            xbest = X[blocation, :]
            fbest = best
        elif best < fbest:
            fbest = best
            xbest = X[blocation, :]

        # Update positions based on the two phases
        for i in range(Search_Agents.shape[0]):
            # Phase 1: Exploration
            I = np.round(1 + np.random.rand()).astype(int)
            k = np.random.randint(0, Search_Agents)
            P = X[k, :]  # Eq. (3)
            F_P = fit[k]

            if fit[i] > F_P:
                X_new = X[i, :] + np.random.rand(1, dimensions) * (P - I * X[i, :])  # Eq. (4)
            else:
                X_new = X[i, :] + np.random.rand(1, dimensions) * (X[i, :] - P)  # Eq. (4)

            X_new = np.clip(X_new, Lowerbound, Upperbound)  # Enforce bounds
            CurrentFit = fit[i]
            MeanFit = np.mean(fit)
            if CurrentFit > MeanFit:
                # Update position based on Eq (5)
                fit_new[i] = objective(X_new)
                if fit_new[i] < fit[i]:
                    X[i, :] = X_new
                    fit[i] = fit_new[i]
            else:
                # CLO Update position
                rng = np.random.default_rng(5)
                r_ij = rng.random(dimensions)
                I_ij = rng.integers(1, 3, size=dimensions)

                if fit[i] < F_P:
                    x1 = X + r_ij * (P - I_ij * X)
                else:
                    x1 = X + r_ij * (X - I_ij * P)

                x1 = clip_bounds(x1, Lowerbound, Upperbound)
                f1 = objective(x1)
                if X[i] < fit[i]:
                    X[i] = x1
                    fit[i] = f1

                # Phase 2: Exploitation
                R = 0.02 * (1 - t / Max_iterations)  # Eq. (6)
                X_new = X[i, :] + (-R + 2 * R * np.random.rand(1, dimensions)) * X[i, :]  # Eq. (7)

                X_new = np.clip(X_new, Lowerbound, Upperbound)  # Enforce bounds

                # Update position based on Eq (8)
                fit_new[i] = objective(X_new)
                if fit_new[i] < fit[i]:
                    X[i, :] = X_new
                    fit[i] = fit_new[i]

        # Save best score
        NGO_curve[t - 1] = fbest
        Score = fbest
        Best_pos = xbest
    ct = time.time() - ct
    return Score, NGO_curve, Best_pos, ct
