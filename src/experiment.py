import numpy as np
import pandas as pd

from .filters import select_filter_method
from .doa import doa_select_features_fixed_k, DOAParams
from .evaluation import evaluate_on_test

def run_experiment(
    X_train, y_train, X_test, y_test, feature_names,
    classifiers, methods, ks,
    knn_k_value=5,
    doa_params=None
):
    Dim = X_train.shape[1]
    doa_params = doa_params or DOAParams()

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
                    n_selected_label = "All"
                    method_label = "none"

                elif method in ("relief", "chi2", "infogain"):
                    idx = select_filter_method(X_train, y_train, method=method, k=k)
                    n_selected_label = str(k)
                    method_label = method

                elif method == "doa":
                    idx = doa_select_features_fixed_k(X_train, y_train, k=k, clf=clf, params=doa_params)
                    n_selected_label = str(k)
                    method_label = "doa"

                else:
                    raise ValueError(f"Unknown method: {method}")

                res = evaluate_on_test(X_train[:, idx], y_train, X_test[:, idx], y_test, clf)

                rows.append({
                    "classifier": clf_name,
                    "method": method_label,
                    "n_selected": n_selected_label,                 # ✅ k yerine
                    "n_features": len(idx),
                    "knn_k": knn_k_value if clf_name == "KNN" else None,  # ✅ sadece KNN
                    "test_acc": res["acc"],
                    "test_f1_macro": res["f1_macro"],
                    "features": ", ".join([feature_names[i] for i in idx]),
                })

    results = pd.DataFrame(rows).sort_values(["classifier", "method", "n_features"])
    return results
