from src.config import Config
from src.data_io import load_train_test
from src.models import build_classifiers
from src.experiment import run_experiment
from src.doa import DOAParams

def main():
    cfg = Config()

    X_train, y_train, X_test, y_test, feature_names = load_train_test(
        cfg.train_path, cfg.test_path, cfg.target_col
    )

    knn_k_value = 5
    classifiers = build_classifiers(knn_k=knn_k_value, rf_estimators=200, seed=42)

    doa_params = DOAParams(pop_size=20, Tmax=25, u=0.9, seed=42, cv_folds=5)

    results = run_experiment(
        X_train, y_train, X_test, y_test, feature_names,
        classifiers=classifiers,
        methods=cfg.methods,
        ks=cfg.ks,
        knn_k_value=knn_k_value,
        doa_params=doa_params,
    )

    print(results[[
        "classifier", "method", "n_selected", "n_features", "knn_k", "test_acc", "test_f1_macro"
    ]])

    results.to_excel(cfg.output_xlsx, index=False)
    print(f"\nSaved: {cfg.output_xlsx}")

if __name__ == "__main__":
    main()
