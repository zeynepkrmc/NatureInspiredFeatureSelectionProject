import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, chi2

def reliefF_scores(X, y, n_neighbors=10):
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

    raise ValueError(f"Unknown filter method: {method}")
