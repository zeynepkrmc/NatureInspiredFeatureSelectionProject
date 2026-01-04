import numpy as np
import pandas as pd
import math
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import mutual_info_classif, chi2

import dataload

# -------------------------
# 1) Data load
# -------------------------
train = pd.read_excel("data/train.xlsx")
test  = pd.read_excel("data/test.xlsx")

TARGET = "Diagnosis"
X_train = train.drop(columns=[TARGET]).values
y_train_raw = train[TARGET].values

X_test  = test.drop(columns=[TARGET]).values
y_test_raw  = test[TARGET].values

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

feature_names = train.drop(columns=[TARGET]).columns.to_list()
Dim = X_train.shape[1]

# -------------------------
# 2) Filter scorers: ReliefF, Chi2, Information Gain
# -------------------------
def reliefF_scores(X, y, n_neighbors=10):
    """
    Basit ReliefF skorlayıcı.
    Skor büyükse daha önemli kabul edip top-k seçiyoruz.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    m, n = X.shape

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X, return_distance=True)
    idx = idx[:, 1:]  # self hariç

    rng = X.max(axis=0) - X.min(axis=0)
    rng[rng == 0] = 1.0

    scores = np.zeros(n)
    for i in range(m):
        for j in idx[i]:
            diff = np.abs(X[i] - X[j]) / rng
            if y[i] == y[j]:
                scores -= diff / (m * n_neighbors)
            else:
                scores += diff / (m * n_neighbors)
    return scores


def select_k_by_scores(scores, k):
    scores = np.asarray(scores)
    idx = np.argsort(scores)[-k:]
    return np.sort(idx)


def select_filter_method(X, y, method, k):
    if k is None:
        return np.arange(X.shape[1])

    if method == "relief":
        s = reliefF_scores(X, y, n_neighbors=10)
        return select_k_by_scores(s, k)

    if method == "chi2":
        # chi2 non-negative ister -> MinMax ölçekle
        X_pos = MinMaxScaler().fit_transform(X)
        s, _ = chi2(X_pos, y)
        return select_k_by_scores(s, k)

    if method == "infogain":
        s = mutual_info_classif(X, y, random_state=42)
        return select_k_by_scores(s, k)

    raise ValueError("Unknown filter method")


# -------------------------
# 3) DOA (wrapper) for fixed-k feature selection
# -------------------------
@dataclass
class DOAParams:
    pop_size: int = 20
    Tmax: int = 25
    u: float = 0.9
    seed: int = 42
    cv_folds: int = 5


def make_model(clf):
    # KNN ve GNB ölçekten faydalanır; RF genelde gerekmez
    if isinstance(clf, (KNeighborsClassifier, GaussianNB)):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return clf


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


# -------------------------
# 4) Evaluation helper
# -------------------------
def evaluate_on_test(Xtr, ytr, Xte, yte, clf):
    model = make_model(clf)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "acc": accuracy_score(yte, pred),
        "f1_macro": f1_score(yte, pred, average="macro")
    }


# -------------------------
# 5) Experiment grid
# -------------------------
classifiers = {
    "RF":  RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),  # ✅ K=5 burada
    "GNB": GaussianNB()
}

methods = ["none", "relief", "chi2", "infogain", "doa"]
ks = [None, 5, 10, 15, 20]  # None -> selectmeden (All)

rows = []

for clf_name, clf in classifiers.items():
    for method in methods:
        for k in ks:
            if method == "none" and k is not None:
                continue
            if method != "none" and k is None:
                continue

            if method == "none":
                idx = np.arange(Dim)
                k_label = "All"
                method_label = "none"
            elif method in ("relief", "chi2", "infogain"):
                idx = select_filter_method(X_train, y_train, method=method, k=k)
                k_label = str(k)
                method_label = method
            elif method == "doa":
                idx = doa_select_features_fixed_k(X_train, y_train, k=k, clf=clf, params=DOAParams())
                k_label = str(k)
                method_label = "doa"

            res = evaluate_on_test(X_train[:, idx], y_train, X_test[:, idx], y_test, clf)

            # rows.append({
            #     "classifier": clf_name,
            #     "method": method_label,
            #     "k": k_label,
            #     "n_features": len(idx),
            #     "test_acc": res["acc"],
            #     "test_f1_macro": res["f1_macro"],
            #     "features": ", ".join([feature_names[i] for i in idx])
            # })
            rows.append({
                "classifier": clf_name,
                "method": method_label,
                "n_selected": k_label,              # 🔁 k yerine n_selected
                "n_features": len(idx),
                "knn_k": 5 if clf_name == "KNN" else None,  # 🔹 sadece KNN için
                "test_acc": res["acc"],
                "test_f1_macro": res["f1_macro"],
                "features": ", ".join([feature_names[i] for i in idx])
            })


results = pd.DataFrame(rows).sort_values(["classifier", "method", "n_features"])
# print(results[["classifier","method","k","n_features","test_acc","test_f1_macro"]])
print(results[[
    "classifier",
    "method",
    "n_selected",
    "n_features",
    "knn_k",
    "test_acc",
    "test_f1_macro"
]])

results.to_excel("feature_selection_results1.xlsx", index=False)
print("\nSaved: feature_selection_results1.xlsx")
