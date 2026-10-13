import time

import numpy as np


# NGO Algorithm
def NGO(Search_Agents, objective, Lowerbound, Upperbound, Max_iterations):
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
    for t in range(1, Max_iterations + 1):
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
        for i in range(Search_Agents):
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

            # Update position based on Eq (5)
            fit_new[i] = objective(X_new)
            if fit_new[i] < fit[i]:
                X[i, :] = X_new
                fit[i] = fit_new[i]

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
