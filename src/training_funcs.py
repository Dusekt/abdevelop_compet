import torch
import numpy as np
import os
from itertools import combinations
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils import clip_grad_norm_
import pandas as pd



# -----------------------------
# Train & Evaluate
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, grad_clip=1):
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0
    for embeds, descs, labels in loader:
        embeds, descs, labels = embeds.to(device), descs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(embeds, descs).squeeze(-1)
        loss = criterion(preds, labels.squeeze(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item() * embeds.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    """
    Evaluate model performance on a DataLoader, computing per-target RÂ².
    Returns:
        avg_loss: float, mean loss over dataset
        r2_mean: float, mean RÂ² across all targets
        r2_per_target: list of floats, RÂ² per target
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds_list, labels_list = [], []

    criterion = nn.MSELoss(reduction='sum')  # sum for averaging later

    with torch.no_grad():
        for embeds, descs, labels in loader:
            embeds = embeds.to(device)
            descs = descs.to(device)
            labels = labels.to(device)

            outputs = model(embeds, descs)  # shape: (B, num_targets)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_samples += embeds.size(0)

            preds_list.append(outputs.cpu())
            labels_list.append(labels.cpu())

    if total_samples == 0:
        return float("nan"), float("nan"), []

    preds_all = torch.cat(preds_list, dim=0)  # (N, num_targets)
    labels_all = torch.cat(labels_list, dim=0)

    # Compute per-target RÂ²
    r2_per_target = []
    for i in range(labels_all.shape[1]):
        ss_res = torch.sum((labels_all[:, i] - preds_all[:, i]) ** 2)
        ss_tot = torch.sum((labels_all[:, i] - labels_all[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_per_target.append(r2.item())

    r2_mean = np.mean(r2_per_target)
    avg_loss = total_loss / total_samples

    return avg_loss, r2_mean, r2_per_target

def warmup_train(dataset, model_class, epochs=50, batch_size=16, lr=1e-4,
                 weight_decay=1e-5, device='cuda', save_path='warmup.pt', patience=5, con=False, output_dim=1):
    """
    Pretrain the model on a large auxiliary dataset (e.g., TAP dataset)
    with early stopping based on validation loss.
    """

    # Split for validation
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model_class().to(device)
    model.freeze_ablang(False)  # train all during warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = np.inf
    best_epoch = 0
    patience_counter = 0
    if con and os.path.exists(save_path):
        state_dict = torch.load(save_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.replace_head(output_dim=output_dim)

    print(f"?? Starting warmup: {len(train_ds)} train, {len(val_ds)} val samples")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, r2_mean, r2_per_target = evaluate(model, val_loader, device)

        print(f"Warmup Epoch {epoch+1}] "
              f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"R2_mean={r2_mean:.3f}, R2_per_target={r2_per_target}")
        # Check for improvement
        if val_loss < best_val - 1e-6:  # small threshold to avoid noise
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"? Early stopping triggered at epoch {epoch+1} "
                  f"(best epoch {best_epoch+1}, val_loss={best_val:.4f})")
            break

    print(f"? Warmup finished. Best model (epoch {best_epoch+1}) saved to {save_path}")
    return model


def cross_validate_with_pretrained(dataset, model_class, pretrained_path=None,
                                   fa="fold_array.npy", epochs=500, batch_size=8,
                                   device="cuda", lr=1e-4, weight_decay=1e-3,
                                   patience=50, frz_a=20, frz_d=20, output_dim=5, save=False, save_name=1):

    fold_array = np.load(fa)
    if len(fold_array) != len(dataset):
        fold_array = fold_array[:len(dataset)]

    k = int(fold_array.max()) + 1
    val_losses, val_r2s = [], []

    for fold in range(k):
        val_idx = np.where(fold_array == fold)[0]
        train_idx = np.where(fold_array != fold)[0]

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        model = model_class().to(device)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.replace_head(output_dim=output_dim)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=lr, weight_decay=weight_decay)

        best_val = float("inf")
        best_r2 = -float("inf")
        wait = 0
        best_state = None
        unfrozen_a = False
        unfrozen_d = False
        model.freeze_ablang(True)

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, r2_mean, r2_targets = evaluate(model, val_loader, device)
            print(f"[Fold {fold+1} | Epoch {epoch+1}] Train={train_loss:.4f}, Val={val_loss:.4f}, R2={r2_mean:.3f}")

            if val_loss < best_val:
                best_val = val_loss
                best_r2 = r2_mean
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (fold {fold+1})")
                break
                
            if wait >= frz_a and not unfrozen_a:
                model.freeze_ablang(False)
                unfrozen_a = True
                
            if wait >= frz_d and not unfrozen_d:
                model.freeze_dsp(False)
                unfrozen_d = True

        if best_state and save:
            torch.save(best_state, f"cv_fold_{save_name}_{fold}.pt")
        val_losses.append(best_val)
        val_r2s.append(best_r2)

    print("CV results:", val_losses)
    print("Mean loss:", np.mean(val_losses))
    print("Mean R2:", np.mean(val_r2s))

    return val_losses, val_r2s