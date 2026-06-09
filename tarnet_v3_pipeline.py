"""
Corrected TARNet training pipeline for the pregnancy PPG project.

This script replaces the fragile Colab notebook workflow with a repeatable
local/Colab-compatible pipeline:

1. Use the v3 5-second window dataset produced by preprocess_v3_tarnet.py.
2. Export it to the folder layout expected by TARNet:
      TARNet/data/PREG_5SEC_SUBJECT_CLEAN/
3. Keep subject-level train/test split only. No wave-level leakage.
4. Keep train/test subject IDs for subject-level voting.
5. If a self-written/local TARNet-compatible project exists, patch it safely:
      - fix only the deprecated np.float token, not np.float32
      - save best model
      - run subject-level voting on test_subject_ids.npy
      - optional learning-rate scheduler and stage logging
6. Optionally launch that local TARNet-compatible training script.

For this class project, do not use --clone-tarnet: the model code should be
written in this repository. Use train_tarnet_v3.py for the self-contained
implementation.

Important:
    This script assumes dataset_v3_tarnet.npz already exists. Run:
        python3 preprocess_v3_tarnet.py
    before running this pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_NPZ = PROJECT_ROOT / "dataset_v3_tarnet.npz"
DEFAULT_TARNET_DIR = PROJECT_ROOT / "TARNet"
DEFAULT_DATASET_NAME = "PREG_5SEC_SUBJECT_CLEAN"


def normalize_subject_id(sid: object) -> str:
    text = str(sid)
    if text.endswith("_aug"):
        return text[:-4]
    return text


def load_v3_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Cannot find {npz_path}. Run `python3 preprocess_v3_tarnet.py` first."
        )

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "subject_ids_train",
        "subject_ids_test",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing keys: {missing}")

    return {key: data[key] for key in data.files}


def validate_subject_split(train_sids: np.ndarray, test_sids: np.ndarray) -> None:
    train_base = {normalize_subject_id(sid) for sid in train_sids}
    test_base = {normalize_subject_id(sid) for sid in test_sids}
    overlap = sorted(train_base & test_base)
    if overlap:
        raise ValueError(f"Subject leakage detected: {overlap}")


def export_dataset_for_tarnet(
    npz_path: Path,
    tarnet_dir: Path,
    dataset_name: str,
    add_channel_dim: bool = True,
    overwrite: bool = True,
) -> Path:
    data = load_v3_dataset(npz_path)

    X_train = data["X_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    train_sids = data["subject_ids_train"]
    test_sids = data["subject_ids_test"]

    validate_subject_split(train_sids, test_sids)

    if add_channel_dim:
        if X_train.ndim == 2:
            X_train = X_train[..., None]
        if X_test.ndim == 2:
            X_test = X_test[..., None]

    dataset_dir = tarnet_dir / "data" / dataset_name
    if dataset_dir.exists() and overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.save(dataset_dir / "X_train.npy", X_train)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "X_test.npy", X_test)
    np.save(dataset_dir / "y_test.npy", y_test)
    np.save(dataset_dir / "train_subject_ids.npy", train_sids)
    np.save(dataset_dir / "test_subject_ids.npy", test_sids)

    # Keep aliases too; different scripts use different names.
    np.save(dataset_dir / "subject_ids_train.npy", train_sids)
    np.save(dataset_dir / "subject_ids_test.npy", test_sids)

    if "F_train" in data:
        np.save(dataset_dir / "F_train.npy", data["F_train"].astype(np.float32))
    if "F_test" in data:
        np.save(dataset_dir / "F_test.npy", data["F_test"].astype(np.float32))
    if "feature_names" in data:
        np.save(dataset_dir / "feature_names.npy", data["feature_names"])

    metadata = {
        "dataset_name": dataset_name,
        "source": str(npz_path),
        "x_train_shape": list(X_train.shape),
        "x_test_shape": list(X_test.shape),
        "train_labels": {
            str(int(k)): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
        },
        "test_labels": {
            str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))
        },
        "subject_overlap": 0,
        "label_mapping": {"0": "non-pregnant", "1": "pregnant"},
        "preprocessing": {
            "raw_sampling_rate_hz": 500,
            "downsample_ratio": 5,
            "model_sampling_rate_hz": 100,
            "window_seconds": 5,
            "window_points": 500,
            "split": "subject-level",
            "augmentation": "train-only conservative jitter/scaling from preprocess_v3_tarnet.py",
        },
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Prepared TARNet dataset:", dataset_dir)
    print("X_train:", X_train.shape, "labels:", np.unique(y_train, return_counts=True))
    print("X_test :", X_test.shape, "labels:", np.unique(y_test, return_counts=True))
    print("Subject overlap: 0")
    return dataset_dir


def clone_tarnet_if_needed(tarnet_dir: Path) -> None:
    raise SystemExit(
        "This project should not pull the official TARNet repository. "
        "Use the self-contained `train_tarnet_v3.py` instead."
    )
    if (tarnet_dir / "script.py").exists() and (tarnet_dir / "utils.py").exists():
        return

    if tarnet_dir.exists():
        tarnet_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = tarnet_dir.parent / "_TARNet_clone_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ranakroychowdhury/TARNet.git",
            str(tmp_dir),
        ]
        print("Cloning TARNet source into temporary directory:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        for item in tmp_dir.iterdir():
            dest = tarnet_dir / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        shutil.rmtree(tmp_dir)
        return

    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/ranakroychowdhury/TARNet.git",
        str(tarnet_dir),
    ]
    print("Cloning TARNet:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_fix_np_float(py_path: Path) -> None:
    text = py_path.read_text(encoding="utf-8", errors="ignore")
    fixed = re.sub(r"(?<![\w.])np\.float(?!\w)", "float", text)
    if fixed != text:
        py_path.write_text(fixed, encoding="utf-8")


def patch_np_float_tokens(tarnet_dir: Path) -> None:
    for py_path in tarnet_dir.rglob("*.py"):
        safe_fix_np_float(py_path)


HELPER_MARKER = "# === PREG_PPG_SUBJECT_EVAL_PATCH_V3 ==="
SCHEDULER_MARKER = "# === PREG_PPG_LR_SCHEDULER_PATCH_V3 ==="
STAGE_MARKER = "# === PREG_PPG_STAGE_OUTPUT_PATCH_V3 ==="


HELPER_CODE = r'''
# === PREG_PPG_SUBJECT_EVAL_PATCH_V3 ===

def preg_ppg_save_best_model(best_model, prop):
    import json
    import os
    import torch

    save_dir = os.path.join("./trained_models", prop["dataset"])
    os.makedirs(save_dir, exist_ok=True)

    torch.save(best_model, os.path.join(save_dir, "best_model_full.pt"))
    torch.save(best_model.state_dict(), os.path.join(save_dir, "best_model_state_dict.pt"))

    prop_json = {}
    for key, value in prop.items():
        try:
            json.dumps(value)
            prop_json[key] = value
        except TypeError:
            prop_json[key] = str(value)
    prop_json["label_mapping"] = {"0": "non-pregnant", "1": "pregnant"}

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(prop_json, f, indent=2, ensure_ascii=False)

    print("\n[Model Saved]", save_dir)


def preg_ppg_subject_level_eval(model, X, y, prop):
    import os
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    dataset_dir = os.path.join("./data", prop["dataset"])
    subject_path = os.path.join(dataset_dir, "test_subject_ids.npy")
    if not os.path.exists(subject_path):
        print("\n[Subject-level] test_subject_ids.npy not found; skipped.")
        return None

    subject_ids = np.load(subject_path, allow_pickle=True)
    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)

    n_valid = min(len(subject_ids), len(y_np), X.shape[0])
    subject_ids = subject_ids[:n_valid]
    y_np = y_np[:n_valid]

    batch_size = int(prop["batch"])
    device = prop["device"]
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)
            num_inst = end - start

            if hasattr(X, "detach"):
                x_batch = X[start:end].detach().clone().float().to(device)
            else:
                x_batch = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)

            if num_inst < batch_size:
                pad_shape = (batch_size - num_inst,) + tuple(x_batch.shape[1:])
                pad = torch.zeros(pad_shape, dtype=x_batch.dtype, device=device)
                x_batch = torch.cat([x_batch, pad], dim=0)

            logits = model(x_batch, prop["task_type"])[0][:num_inst]
            outputs.append(logits)

    logits = torch.cat(outputs, dim=0)
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
    pred = np.argmax(prob, axis=1)

    rows = []
    for sid in np.unique(subject_ids):
        idx = np.where(subject_ids == sid)[0]
        true_label = int(np.bincount(y_np[idx].astype(int)).argmax())
        vote_counts = np.bincount(pred[idx].astype(int), minlength=prop["nclasses"])
        pred_label = int(np.argmax(vote_counts))
        rows.append({
            "subject_id": sid,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": int(true_label == pred_label),
            "non_pregnant_votes": int(vote_counts[0]),
            "pregnant_votes": int(vote_counts[1]) if len(vote_counts) > 1 else 0,
            "total_segments": int(len(idx)),
            "majority_confidence": float(vote_counts[pred_label] / len(idx)),
            "pregnant_prob_mean": float(prob[idx, 1].mean()) if prob.shape[1] > 1 else float("nan"),
        })

    df = pd.DataFrame(rows)
    subject_acc = accuracy_score(df["true_label"], df["pred_label"])

    print("\n================ Subject-level Evaluation ================")
    print(df.to_string(index=False))
    print("\nSubject-level Accuracy:", subject_acc)
    print("\nConfusion Matrix, labels=[0 non-pregnant, 1 pregnant]:")
    print(confusion_matrix(df["true_label"], df["pred_label"], labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(
        df["true_label"],
        df["pred_label"],
        labels=[0, 1],
        target_names=["non-pregnant", "pregnant"],
        zero_division=0,
    ))

    save_path = os.path.join(dataset_dir, "subject_level_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("Subject-level results saved to:", save_path)
    print("==========================================================\n")
    return subject_acc
'''


SCHEDULER_CODE = r'''

# === PREG_PPG_LR_SCHEDULER_PATCH_V3 ===

def preg_ppg_lr_schedule_step(optimizer, epoch, prop):
    import math
    import os

    mode = os.environ.get("CUSTOM_LR_SCHEDULE", "none").lower()
    if mode in ["none", ""]:
        return

    epoch_num = int(epoch)
    total_epochs = int(prop.get("epochs", 50))

    base_lr = float(os.environ.get("CUSTOM_BASE_LR", prop.get("lr", 0.0001)))
    mid_lr = float(os.environ.get("CUSTOM_MID_LR", base_lr * 0.5))
    min_lr = float(os.environ.get("CUSTOM_MIN_LR", base_lr * 0.1))
    step1 = int(os.environ.get("CUSTOM_STEP1", "15"))
    step2 = int(os.environ.get("CUSTOM_STEP2", "35"))

    if mode == "step":
        if epoch_num <= step1:
            lr = base_lr
        elif epoch_num <= step2:
            lr = mid_lr
        else:
            lr = min_lr
    elif mode == "cosine":
        t = 0.0 if total_epochs <= 1 else (epoch_num - 1) / (total_epochs - 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    else:
        return

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if epoch_num == 1 or epoch_num % 10 == 0 or epoch_num in [step1, step1 + 1, step2, step2 + 1, total_epochs]:
        print("[Custom LR Scheduler] Epoch:", epoch_num, "LR:", lr)
'''


def append_helper_once(utils_path: Path, marker: str, code: str) -> None:
    text = utils_path.read_text(encoding="utf-8", errors="ignore")
    if marker not in text:
        utils_path.write_text(text.rstrip() + "\n\n" + code.strip() + "\n", encoding="utf-8")


def insert_before_first_del_model(utils_path: Path) -> None:
    text = utils_path.read_text(encoding="utf-8", errors="ignore")
    call_marker = "preg_ppg_save_best_model(best_model, prop)"
    if call_marker in text:
        return

    lines = text.splitlines()
    new_lines = []
    inserted = False
    for line in lines:
        stripped = line.lstrip()
        if not inserted and stripped.startswith("del model"):
            indent = line[: len(line) - len(stripped)]
            new_lines.extend(
                [
                    indent + "try:",
                    indent + "    preg_ppg_save_best_model(best_model, prop)",
                    indent + "except Exception as e:",
                    indent + "    print('[Model Save Error]', e)",
                    indent + "if prop['task_type'] == 'classification':",
                    indent + "    try:",
                    indent + "        preg_ppg_subject_level_eval(best_model, X_test, y_test, prop)",
                    indent + "    except Exception as e:",
                    indent + "        print('[Subject-level Eval Error]', e)",
                ]
            )
            inserted = True
        new_lines.append(line)

    if not inserted:
        print("Warning: could not find `del model` in utils.py; model save/eval call not inserted.")
        return

    utils_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def insert_scheduler_call(utils_path: Path) -> None:
    text = utils_path.read_text(encoding="utf-8", errors="ignore")
    call = "        preg_ppg_lr_schedule_step(optimizer, epoch, prop)\n"
    if call in text:
        return
    target = "    for epoch in range(1, prop['epochs'] + 1):\n"
    if target not in text:
        print("Warning: training loop not found; LR scheduler call not inserted.")
        return
    utils_path.write_text(text.replace(target, target + call, 1), encoding="utf-8")


def insert_stage_output(utils_path: Path) -> None:
    text = utils_path.read_text(encoding="utf-8", errors="ignore")
    if STAGE_MARKER in text:
        return

    target = """        if prop['task_type'] == 'classification' and test_metrics[1] > acc:
            acc = test_metrics[1]
        elif prop['task_type'] == 'regression' and test_metrics[0] < rmse:
            rmse = test_metrics[0]
            mae = test_metrics[1]
"""
    if target not in text:
        print("Warning: stage-output insertion point not found; skipped.")
        return

    insert = target + f"""
        {STAGE_MARKER}
        stage_every = int(os.environ.get("CUSTOM_STAGE_EVERY", "0"))
        if stage_every > 0 and (epoch % stage_every == 0 or epoch == prop['epochs']):
            current_metrics = test(
                model, X_test, y_test,
                prop['batch'], prop['nclasses'],
                criterion_task, prop['task_type'],
                prop['device'], prop['avg']
            )
            print("\\n========== Stage Result ==========")
            print("Epoch:", epoch)
            print("Dataset:", prop['dataset'])
            if prop['task_type'] == 'classification':
                print("Current pulse/window-level test loss:", current_metrics[0])
                print("Current pulse/window-level test acc :", current_metrics[1])
                print("Best pulse/window-level test acc    :", acc)
            else:
                print("Current model RMSE:", current_metrics[0], "MAE:", current_metrics[1])
            print("Training TASK Loss:", task_loss)
            print("==================================\\n")
"""
    utils_path.write_text(text.replace(target, insert, 1), encoding="utf-8")


def patch_tarnet(tarnet_dir: Path) -> None:
    if not tarnet_dir.exists():
        raise FileNotFoundError(f"TARNet directory not found: {tarnet_dir}")

    utils_path = tarnet_dir / "utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Cannot find TARNet utils.py: {utils_path}")

    patch_np_float_tokens(tarnet_dir)
    append_helper_once(utils_path, HELPER_MARKER, HELPER_CODE)
    append_helper_once(utils_path, SCHEDULER_MARKER, SCHEDULER_CODE)
    insert_before_first_del_model(utils_path)
    insert_scheduler_call(utils_path)
    insert_stage_output(utils_path)
    print("Patched TARNet:", tarnet_dir)


def run_training(args: argparse.Namespace) -> None:
    script_path = args.tarnet_dir / "script.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find TARNet script.py: {script_path}")

    env = os.environ.copy()
    env["CUSTOM_STAGE_EVERY"] = str(args.stage_every)
    env["CUSTOM_LR_SCHEDULE"] = args.lr_schedule
    env["CUSTOM_BASE_LR"] = str(args.lr)
    env["CUSTOM_MID_LR"] = str(args.mid_lr)
    env["CUSTOM_MIN_LR"] = str(args.min_lr)
    env["CUSTOM_STEP1"] = str(args.step1)
    env["CUSTOM_STEP2"] = str(args.step2)

    cmd = [
        sys.executable,
        "script.py",
        "--dataset",
        args.dataset_name,
        "--task_type",
        "classification",
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
        "--nlayers",
        str(args.nlayers),
        "--emb_size",
        str(args.emb_size),
        "--nhead",
        str(args.nhead),
        "--task_rate",
        str(args.task_rate),
        "--masking_ratio",
        str(args.masking_ratio),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=args.tarnet_dir, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--tarnet-dir", type=Path, default=DEFAULT_TARNET_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--clone-tarnet", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--no-channel-dim", action="store_true")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--mid-lr", type=float, default=0.00005)
    parser.add_argument("--min-lr", type=float, default=0.00001)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--task-rate", type=float, default=0.01)
    parser.add_argument("--masking-ratio", type=float, default=0.01)
    parser.add_argument("--lr-schedule", choices=["none", "step", "cosine"], default="step")
    parser.add_argument("--step1", type=int, default=15)
    parser.add_argument("--step2", type=int, default=35)
    parser.add_argument("--stage-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tarnet_dir = args.tarnet_dir.resolve()
    args.dataset_npz = args.dataset_npz.resolve()

    if args.clone_tarnet:
        clone_tarnet_if_needed(args.tarnet_dir)

    export_dataset_for_tarnet(
        npz_path=args.dataset_npz,
        tarnet_dir=args.tarnet_dir,
        dataset_name=args.dataset_name,
        add_channel_dim=not args.no_channel_dim,
        overwrite=not args.no_overwrite,
    )

    if args.prepare_only:
        return

    patch_tarnet(args.tarnet_dir)

    if args.patch_only:
        return

    run_training(args)


if __name__ == "__main__":
    main()
