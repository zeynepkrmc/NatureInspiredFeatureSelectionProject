import math
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .models import make_model

@dataclass
class DOAParams:
    pop_size: int = 20
    Tmax: int = 25
    u: float = 0.9
    seed: int = 42
    cv_folds: int = 5

def doa_select_features_fixed_k(X, y, k, clf, params=DOAParams()):
    rng = np.random.default_rng(params.seed)
    N = params.pop_size
    Dim = X.shape[1]

    xl, xu = 0.0, 1.0
    Tmax = params.Tmax
    Td = int(round(0.9 * Tmax))  # Td = 9/10 * Tmax

    cv = StratifiedKFold(n_splits=params.cv_folds, shuffle=True, random_state=params.seed)
    model = make_model(clf)

    cache = {}

    def fitness(vec):
        idx = tuple(np.argsort(vec)[-k:])  # top-k
        if idx in cache:
            return cache[idx]
        Xk = X[:, idx]
        score = cross_val_score(model, Xk, y, cv=cv, scoring="accuracy").mean()
        val = 1.0 - score  # minimize
        cache[idx] = val
        return val

    pop = rng.uniform(xl, xu, size=(N, Dim))
    fit = np.array([fitness(pop[i]) for i in range(N)])

    best_i = int(np.argmin(fit))
    best = pop[best_i].copy()
    best_fit = float(fit[best_i])

    g = 5
    group_sizes = [N // g] * g
    for i in range(N % g):
        group_sizes[i] += 1
    bounds = np.cumsum([0] + group_sizes)

    for t in range(Tmax):
        if t < Td:
            for q in range(g):
                s, e = bounds[q], bounds[q + 1]
                if s == e:
                    continue

                g_pop = pop[s:e]
                g_fit = fit[s:e]
                gbest = g_pop[int(np.argmin(g_fit))].copy()

                lo = int(np.ceil(Dim / (8 * (q + 1))))
                hi = int(max(2, np.ceil(Dim / (3 * (q + 1)))))
                lo = max(lo, 1)
                hi = max(hi, lo)
                kq = int(rng.integers(lo, hi + 1))

                for i_ind in range(s, e):
                    pop[i_ind] = gbest.copy()
                    K = rng.choice(Dim, size=min(kq, Dim), replace=False)

                    if rng.random() < params.u:
                        randv = rng.random(len(K))
                        cos_factor = 0.5 * (math.cos(math.pi * (t + Tmax - Td) / Tmax) + 1.0)
                        pop[i_ind, K] = gbest[K] + (xl + randv * (xu - xl)) * cos_factor
                    else:
                        for d in K:
                            m = int(rng.integers(0, N))
                            pop[i_ind, d] = pop[m, d]

                    pop[i_ind] = np.clip(pop[i_ind], xl, xu)

        else:
            lo, hi = 2, int(max(2, np.ceil(Dim / 3)))
            kr = int(rng.integers(lo, hi + 1))

            for i_ind in range(N):
                pop[i_ind] = best.copy()
                K = rng.choice(Dim, size=min(kr, Dim), replace=False)
                randv = rng.random(len(K))
                cos_factor = 0.5 * (math.cos(math.pi * t / Tmax) + 1.0)
                pop[i_ind, K] = best[K] + (xl + randv * (xu - xl)) * cos_factor
                pop[i_ind] = np.clip(pop[i_ind], xl, xu)

        fit = np.array([fitness(pop[i]) for i in range(N)])
        bi = int(np.argmin(fit))
        if float(fit[bi]) < best_fit:
            best_fit = float(fit[bi])
            best = pop[bi].copy()

    selected = np.sort(np.argsort(best)[-k:])
    return selected
