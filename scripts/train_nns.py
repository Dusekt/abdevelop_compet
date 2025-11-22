import optuna
import torch
import numpy as np
import os
import uuid
from itertools import combinations
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils import clip_grad_norm_
from concurrent.futures import ProcessPoolExecutor, as_completed

from ../src/datasets import BaseAntibodyDataset, WarmupDataset

from ../src/training_funcs import train_one_epoch, evaluate, warmup_train, cross_validate_with_pretrained

from ../src/models import AntibodyModel2, AntibodyModel3, AntibodyModel4, AntibodyModel5

# -------------------------
# Global warmup combo choices (avoid Optuna persistent-storage tuple warning)
# -------------------------
all_tasks = [0, 1, 2, 3]
warmup_combos = []
for r in range(1, len(all_tasks) + 1):
    warmup_combos.extend(list(combinations(all_tasks, r)))
# Represent combos as comma-joined strings for Optuna categorical storage
warmup_choices = [",".join(map(str, combo)) for combo in warmup_combos]

def objective(trial, finetune_task, model_class):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Warmup choice (use stringified combos to avoid Optuna warning) ---
    choice = trial.suggest_categorical('warmup_tasks', warmup_choices)
    warmup_tasks = list(map(int, choice.split(',')))
    n_warmup = len(warmup_tasks)

    # --- Hyperparameters ---
    warmup_lr = trial.suggest_loguniform("warmup_lr", 1e-6, 1e-3)
    warmup_wd = trial.suggest_loguniform("warmup_wd", 1e-6, 1e-3)
    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 60)

    finetune_lr = trial.suggest_loguniform("finetune_lr", 1e-6, 1e-3)
    finetune_wd = trial.suggest_loguniform("finetune_wd", 1e-6, 1e-3)
    patience = trial.suggest_int("patience", 20, 60)
    frz_a = trial.suggest_int("freeze_epochsa", 10, 40)
    frz_d = trial.suggest_int("freeze_epochsb", 10, 40)

    # Unique warmup checkpoint filename per process/trial to avoid collisions
    warmup_ckpt = f"warmup_trial_{finetune_task}_{model_class.__name__}_{os.getpid()}_{uuid.uuid4().hex}.pt"

    # --- Warmup Stage ---
    warmup_data = WarmupDataset(
        "../data/thera_11_embeddings2.npy",
        "../data/thera_11_DSP.npy",
        "../data/thera_TAP.npy",
        which=warmup_tasks
    )
    warmup_train(
        warmup_data,
        model_class=lambda: model_class(output_dim=n_warmup),
        epochs=warmup_epochs,
        batch_size=128,
        lr=warmup_lr,
        weight_decay=warmup_wd,
        device=device,
        save_path=warmup_ckpt,
        patience=10
    )

    # --- Finetune Stage ---
    dataset = BaseAntibodyDataset(
        "../data/train_embeddings2.npy",
        "../data/train_DSP_out.npy",
        "../data/all_data.npy",
        which=[finetune_task]  # always single-task finetune here
    )

    val_losses, r2_scores = cross_validate_with_pretrained(
        dataset=dataset,
        model_class=lambda: model_class(output_dim=n_warmup),
        pretrained_path=warmup_ckpt,
        fa="../data/fold_array.npy",
        epochs=200,
        batch_size=32,
        device=device,
        lr=finetune_lr,
        weight_decay=finetune_wd,
        patience=patience,
        frz_a=frz_a,
        frz_d=frz_d,
        output_dim=1,
        save=False
    )

    # Cleanup
    if os.path.exists(warmup_ckpt):
        os.remove(warmup_ckpt)

    # Report validation loss and optionally R2
    val_loss = np.mean(val_losses) if len(val_losses) > 0 else float('inf')
    r2_mean = np.mean(r2_scores) if len(r2_scores) > 0 else float('nan')
    print(f"Trial {trial.number}: Warmup {warmup_tasks}, Finetune {finetune_task}, Val Loss={val_loss:.4f}, R2={r2_mean:.4f}")
    
    trial.set_user_attr("mean_r2", r2_mean)

    return float(val_loss)


def train_best(finetune_task, params, model_class, model_name='model0', out_dir ='../outputs'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # In the objective function
    warmup_tasks = params["warmup_tasks"] if isinstance(params.get("warmup_tasks"), list) else list(map(int, params["warmup_tasks"].split(',')))
    n_warmup = len(warmup_tasks)

    # --- Hyperparameters ---
    warmup_lr = params["warmup_lr"]
    warmup_wd = params["warmup_wd"]
    warmup_epochs = params["warmup_epochs"]

    finetune_lr = params["finetune_lr"]
    finetune_wd = params["finetune_wd"]
    patience = params["patience"]
    frz_a = params["freeze_epochsa"]
    frz_b = params["freeze_epochsb"]

    # unique warmup filename for this run
    warmup_ckpt = f"warmup_trial_{finetune_task}_{model_name}_{os.getpid()}_{uuid.uuid4().hex}.pt"

    # --- Warmup Stage ---
    warmup_data = WarmupDataset(
        "../data/thera_11_embeddings2.npy",
        "../data/thera_11_DSP.npy",
        "../data/thera_TAP.npy",
        which=warmup_tasks
    )
    warmup_train(
        warmup_data,
        model_class=lambda: model_class(output_dim=n_warmup),
        epochs=warmup_epochs,
        batch_size=128,
        lr=warmup_lr,
        weight_decay=warmup_wd,
        device=device,
        save_path=warmup_ckpt,
        patience=10
    )

    # --- Finetune Stage ---
    dataset = BaseAntibodyDataset(
        "../data/train_embeddings2.npy",
        "../data/train_DSP_out.npy",
        "../data/all_data.npy",
        which=[finetune_task]  # always single-task finetune here
    )

    val_losses, r2_scores = cross_validate_with_pretrained(
        dataset=dataset,
        model_class=lambda: model_class(output_dim=n_warmup),
        pretrained_path=warmup_ckpt,
        fa="../data/fold_array.npy",
        epochs=200,
        batch_size=32,
        device=device,
        lr=finetune_lr,
        weight_decay=finetune_wd,
        patience=patience,
        frz_a=frz_a,
        frz_d=frz_b,
        output_dim=1,
        save=True,
        save_name=f'{out_dir}/{finetune_task}_{model_name}'
        
    )

    # Cleanup
    if os.path.exists(warmup_ckpt):
        os.remove(warmup_ckpt)

    # Report validation loss and optionally R2
    val_loss = np.mean(val_losses) if len(val_losses) > 0 else float('inf')
    r2_mean = np.mean(r2_scores) if len(r2_scores) > 0 else float('nan')
    
    print(f'Val_loss: {val_loss}')
    print(f'R2: {r2_mean}')


    return float(val_loss)


def run_study(model_class, fold, model_name):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, fold, model_class), n_trials=50)

    print(f"\n?? Best trial for {model_name}")
    print(study.best_trial.params)
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Corresponding mean R2: {study.best_trial.user_attrs.get('mean_r2', float('nan')):.3f}")

    train_best(fold, study.best_trial.params, model_class, model_name=model_name)

    return (model_name, fold, study.best_value)


if __name__ == '__main__':
    # Models and names (paired)
    model_classes = [AntibodyModel2, AntibodyModel3, AntibodyModel4, AntibodyModel5]
    model_names = ['model2', 'model3', 'model4', 'model5']
    folds = [0, 1, 2, 3, 4]

    # Correct pairing model_classes <-> model_names
    tasks = [(m, f, name) for m, name in zip(model_classes, model_names) for f in folds]

    num_tasks = len(tasks)
    max_workers = min(num_tasks, os.cpu_count() or 1)
    print(f"Launching {num_tasks} tasks with up to {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_study, m, f, n) for m, f, n in tasks]

        for future in as_completed(futures):
            try:
                model_name, fold, best_val = future.result()
                print(f"? Completed {model_name} task {fold} with best loss {best_val:.4f}")
            except Exception as e:
                print(f"? Task failed: {e}")
