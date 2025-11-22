import torch
import numpy as np
import os
from itertools import combinations
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils import clip_grad_norm_
import pandas as pd

import optuna
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import optuna.visualization as vis


from ../src/datasets import BaseAntibodyDataset, WarmupDataset

from ../src/training_funcs import train_one_epoch, evaluate, warmup_train, cross_validate_with_pretrained

from ../src/models import AntibodyModel2, AntibodyModel3, AntibodyModel4, AntibodyModel5


# Config
num_folds = 5
num_models = 5
num_tasks = 5
meta_folds = 5  # how many folds for the meta-ensemble training

# Load predictions
val_preds_dict, test_preds_dict = defaultdict(list), defaultdict(list)

for fold in range(num_folds):
    for model_ in range(2, num_models + 1):
        for task in range(num_tasks):
            val_preds_dict[task].append(np.load(f'../model_weights/preds_val_{fold}_model{model_}_{task}.npy'))
            test_preds_dict[task].append(np.load(f'../model_weights/preds_test_{fold}_model{model_}_{task}.npy'))

def CV_ens(task, params):
    y_val = np.load("../data/all_data.npy")[:, 0]
    y_mask = ~np.isnan(y_val)
    y_val = y_val[y_mask]

    val_preds = np.stack(val_preds_dict[0])   # (num_models_total, N_val)
    test_preds = np.stack(test_preds_dict[0]) # (num_models_total, N_test)

    X_val = val_preds.T[y_mask]
    X_test = test_preds.T

    kf = KFold(n_splits=meta_folds, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y_val))
    weights_all = []

    for train_idx, valid_idx in kf.split(X_val):
        X_tr, X_va = X_val[train_idx], X_val[valid_idx]
        y_tr, y_va = y_val[train_idx], y_val[valid_idx]

        model = make_pipeline(StandardScaler(), Ridge(**params))
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        meta_oof[valid_idx] = preds
        weights_all.append(model.named_steps["ridge"].coef_)

    r2 = r2_score(y_val, meta_oof)
    rmse = mean_squared_error(y_val, meta_oof)
    print(f"Meta-model CV R²: {r2:.4f}, RMSE: {rmse:.4f}")
    return rmse


def objective_task(trial, task):
    model_type = trial.suggest_categorical("model_type", ["ridge", "elasticnet"])
    alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    if model_type == "ridge":
        solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
        params = {"model_type": model_type, "alpha": alpha, "fit_intercept": fit_intercept, "solver": solver}
    else:
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        params = {"model_type": model_type, "alpha": alpha, "fit_intercept": fit_intercept, "l1_ratio": l1_ratio}

    rmse = CV_ens(task=task, params=params)
    return rmse


task_best_params = {}

for task in range(num_tasks):
    print(f"\nStarting Optuna for Task {task}")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_task(trial, task), n_trials=50, show_progress_bar=False)

    print(f"✅ Task {task} Best RMSE: {study.best_value}")
    print(f"🏆 Task {task} Best Params:", study.best_params)
    task_best_params[task] = study.best_params

