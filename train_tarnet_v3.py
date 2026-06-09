"""
Self-contained TARNet-inspired training script for v3 pregnancy PPG windows.

This file does not use or clone the official TARNet repository. It implements a
small Transformer time-series classifier with an auxiliary masked reconstruction
loss, which follows the main idea of TARNet-style task-aware reconstruction:
learn a representation that is useful for classification while also preserving
time-series structure.

Input:
    dataset_v3_tarnet.npz from preprocess_v3_tarnet.py

Model input:
    X: (N, 500)  # 5-second PPG windows at 100 Hz
    Optional F: (N, 7) auxiliary handcrafted features

Usage:
    python3 preprocess_v3_tarnet.py
    python3 train_tarnet_v3.py --epochs 80 --use-features
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset_v3_tarnet.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "trained_models" / "self_tarnet_v3"


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install torch before running this script."
        ) from exc
    return torch, nn, F, DataLoader, Dataset


def base_subject_id(sid: object) -> str:
    text = str(sid)
    return text[:-4] if text.endswith("_aug") else text


def split_train_val_by_subject(subject_ids, val_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted({base_subject_id(s) for s in subject_ids}))
    rng.shuffle(subjects)
    n_val = max(1, round(len(subjects) * val_size))
    val_subjects = set(subjects[:n_val])
    train_mask = np.array([base_subject_id(s) not in val_subjects for s in subject_ids])
    val_mask = ~train_mask
    return train_mask, val_mask


def standardize_features(F_train, F_val, F_test):
    mean = F_train.mean(axis=0, keepdims=True)
    std = F_train.std(axis=0, keepdims=True) + 1e-6
    return (F_train - mean) / std, (F_val - mean) / std, (F_test - mean) / std, mean, std


def load_dataset(path: Path, val_size: float, seed: int, use_features: bool):
    data = np.load(path, allow_pickle=True)

    X_train_full = data["X_train"].astype(np.float32)
    y_train_full = data["y_train"].astype(np.int64)
    sid_train_full = data["subject_ids_train"]

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    sid_test = data["subject_ids_test"]

    train_subjects = {base_subject_id(s) for s in sid_train_full}
    test_subjects = {base_subject_id(s) for s in sid_test}
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise ValueError(f"Subject leakage detected between train/test: {overlap}")

    train_mask, val_mask = split_train_val_by_subject(sid_train_full, val_size, seed)

    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]
    sid_train = sid_train_full[train_mask]

    X_val = X_train_full[val_mask]
    y_val = y_train_full[val_mask]
    sid_val = sid_train_full[val_mask]

    feature_payload = None
    if use_features:
        F_train_full = data["F_train"].astype(np.float32)
        F_test = data["F_test"].astype(np.float32)
        F_train = F_train_full[train_mask]
        F_val = F_train_full[val_mask]
        F_train, F_val, F_test, feat_mean, feat_std = standardize_features(F_train, F_val, F_test)
        feature_payload = {
            "F_train": F_train.astype(np.float32),
            "F_val": F_val.astype(np.float32),
            "F_test": F_test.astype(np.float32),
            "feature_names": data.get("feature_names", np.array([])),
            "feature_mean": feat_mean.astype(np.float32),
            "feature_std": feat_std.astype(np.float32),
        }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "sid_train": sid_train,
        "X_val": X_val,
        "y_val": y_val,
        "sid_val": sid_val,
        "X_test": X_test,
        "y_test": y_test,
        "sid_test": sid_test,
        "features": feature_payload,
    }


def make_components():
    torch, nn, torch_F, DataLoader, Dataset = require_torch()

    class PPGDataset(Dataset):
        def __init__(self, X, y, features=None):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.long)
            self.features = None if features is None else torch.as_tensor(features, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            if self.features is None:
                return self.X[idx], self.y[idx], torch.empty(0, dtype=torch.float32)
            return self.X[idx], self.y[idx], self.features[idx]

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class SelfTARNet(nn.Module):
        def __init__(
            self,
            seq_len=500,
            d_model=64,
            nhead=4,
            nlayers=2,
            dropout=0.1,
            nclasses=2,
            feature_dim=0,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.input_proj = nn.Linear(1, d_model)
            self.pos = PositionalEncoding(d_model, max_len=seq_len + 1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
            self.reconstruct_head = nn.Linear(d_model, 1)
            self.feature_proj = None
            classifier_in = d_model
            if feature_dim > 0:
                self.feature_proj = nn.Sequential(
                    nn.Linear(feature_dim, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                classifier_in += d_model // 2
            self.classifier = nn.Sequential(
                nn.LayerNorm(classifier_in),
                nn.Linear(classifier_in, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, nclasses),
            )

        def forward(self, x, features=None, mask_ratio=0.0):
            original = x
            if x.ndim == 2:
                x = x.unsqueeze(-1)

            mask = None
            if self.training and mask_ratio > 0:
                mask = torch.rand(x.shape[:2], device=x.device) < mask_ratio
                x = x.clone()
                x[mask] = 0.0

            h = self.input_proj(x)
            h = self.pos(h)
            h = self.encoder(h)

            pooled = h.mean(dim=1)
            if self.feature_proj is not None and features is not None and features.numel() > 0:
                pooled = torch.cat([pooled, self.feature_proj(features)], dim=1)

            logits = self.classifier(pooled)
            recon = self.reconstruct_head(h).squeeze(-1)
            return logits, recon, original, mask

    return torch, nn, torch_F, DataLoader, PPGDataset, SelfTARNet


def classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    return {
        "accuracy": acc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def subject_level_metrics(subject_ids, y_true, y_prob):
    rows = []
    subject_ids = np.asarray(subject_ids)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    for sid in np.unique(subject_ids):
        idx = np.where(subject_ids == sid)[0]
        true_label = int(np.bincount(y_true[idx].astype(int)).argmax())
        vote_counts = np.bincount(y_pred[idx].astype(int), minlength=2)
        pred_label = int(vote_counts.argmax())
        rows.append(
            {
                "subject_id": str(sid),
                "true_label": true_label,
                "pred_label": pred_label,
                "pregnant_prob_mean": float(y_prob[idx, 1].mean()),
                "total_windows": int(len(idx)),
            }
        )
    metrics = classification_metrics(
        [r["true_label"] for r in rows],
        [r["pred_label"] for r in rows],
    )
    return metrics, rows


def run_epoch(model, loader, optimizer, device, mask_ratio, recon_rate, train=True):
    torch, nn, torch_F, _, _, _ = make_components()
    model.train(train)
    total_loss = 0.0
    total_count = 0
    all_y = []
    all_prob = []

    for x, y, features in loader:
        x = x.to(device)
        y = y.to(device)
        features = features.to(device)

        with torch.set_grad_enabled(train):
            logits, recon, original, mask = model(x, features, mask_ratio=mask_ratio if train else 0.0)
            cls_loss = torch_F.cross_entropy(logits, y)
            if train and recon_rate > 0 and mask is not None and mask.any():
                recon_loss = torch_F.mse_loss(recon[mask], original[mask])
            else:
                recon_loss = torch.zeros((), device=device)
            loss = cls_loss + recon_rate * recon_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = len(y)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        all_y.append(y.detach().cpu().numpy())
        all_prob.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_prob)
    y_pred = y_prob.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics, y_true, y_prob


def train(args):
    torch, nn, torch_F, DataLoader, PPGDataset, SelfTARNet = make_components()

    payload = load_dataset(args.dataset, args.val_size, args.seed, args.use_features)
    features = payload["features"]
    feature_dim = 0
    if features is not None:
        feature_dim = features["F_train"].shape[1]

    train_ds = PPGDataset(
        payload["X_train"],
        payload["y_train"],
        None if features is None else features["F_train"],
    )
    val_ds = PPGDataset(
        payload["X_val"],
        payload["y_val"],
        None if features is None else features["F_val"],
    )
    test_ds = PPGDataset(
        payload["X_test"],
        payload["y_test"],
        None if features is None else features["F_test"],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SelfTARNet(
        seq_len=payload["X_train"].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout,
        feature_dim=feature_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_path = args.out_dir / "best_model.pt"
    history = []
    stale = 0

    print("Device:", device)
    print("Train:", payload["X_train"].shape, np.unique(payload["y_train"], return_counts=True))
    print("Val:  ", payload["X_val"].shape, np.unique(payload["y_val"], return_counts=True))
    print("Test: ", payload["X_test"].shape, np.unique(payload["y_test"], return_counts=True))
    print("Use features:", args.use_features, "feature_dim:", feature_dim)

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mask_ratio=args.mask_ratio,
            recon_rate=args.recon_rate,
            train=True,
        )
        val_metrics, _, _ = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            mask_ratio=0.0,
            recon_rate=0.0,
            train=False,
        )
        scheduler.step()

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "feature_dim": feature_dim,
                    "best_val_accuracy": best_val,
                },
                best_path,
            )
        else:
            stale += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
                f"sen {val_metrics['sensitivity']:.4f} spe {val_metrics['specificity']:.4f}"
            )

        if args.patience > 0 and stale >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_y, test_prob = run_epoch(
        model,
        test_loader,
        optimizer,
        device,
        mask_ratio=0.0,
        recon_rate=0.0,
        train=False,
    )
    subject_metrics, subject_rows = subject_level_metrics(payload["sid_test"], test_y, test_prob)

    result = {
        "best_val_accuracy": float(best_val),
        "window_level_test": test_metrics,
        "subject_level_test": subject_metrics,
        "subject_rows": subject_rows,
        "history": history,
    }
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nBest val accuracy:", best_val)
    print("Window-level test:", test_metrics)
    print("Subject-level test:", subject_metrics)
    print("Saved model:", best_path)
    print("Saved results:", args.out_dir / "results.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.10)
    parser.add_argument("--recon-rate", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-features", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
