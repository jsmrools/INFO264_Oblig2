# src/pcaa_analysis.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from src.model_builder import buildmodel  # uses your existing builder

def load_data(path):
    with np.load(path) as f:
        X, y = f["X"], f["y"]
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False)

def make_pca_svm(best_svm_params: dict, *, n_components=0.95, svd='randomized'):
    """
    Returns a Pipeline: StandardScaler -> PCA -> SVM,
    where best_svm_params are the hyperparams you already found for SVM.
    n_components: float=variance target or int=#components.
    """
    svm = buildmodel("SVM", **best_svm_params)
    pca = PCA(n_components=n_components, svd_solver=svd, random_state=best_svm_params.get("random_state"))
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca",    pca),
        ("svm",    svm),
    ])

def refit_on_trainval(model, X_train, y_train, X_val=None, y_val=None):
    if X_val is not None and y_val is not None:
        X_train = np.vstack([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
    return model.fit(X_train, y_train)

def test_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return float(accuracy_score(y_test, y_pred))

def pick_k_by_val(candidates, best_svm_params, X_train, y_train, X_val, y_val):
    """
    Small helper: try each value in `candidates` for PCA n_components,
    fit on TRAIN, score on VAL, and return the best one.
    `candidates` can contain ints (k) and/or floats (variance targets like 0.95).
    """
    results = []
    for k in candidates:
        pipe = make_pca_svm(best_svm_params, n_components=k)
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_val, y_val)
        results.append((k, float(acc)))
    # best first
    results.sort(key=lambda t: t[1], reverse=True)
    best_k = results[0][0]
    return best_k, results
