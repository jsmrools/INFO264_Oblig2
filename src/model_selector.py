# evaluates and compares different models and hyperparameters using cross validation

def evaluate_models():
    pass
def select_best_model():
    pass





















from src.model_builder import buildmodel
from src.model_trainer import train_model_cv
from itertools import product # https://docs.python.org/3/library/itertools.html#itertools.product

import pandas as pd

def run_model_search(model_name, param_grid, X, y, cv=5, seed=7):
    """
    Runs cross-validation for every combination of hyperparameters for one model type.
    Returns a DataFrame with results for transparency.
    """
    results = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = buildmodel(model_name, randomseed=seed, **params)
        mean_score = train_model_cv(model, X, y, cv=cv, seed=seed)
        results.append({
            'model': model_name,
            **params,
            'cv_mean_accuracy': mean_score
        })

    return pd.DataFrame(results)

