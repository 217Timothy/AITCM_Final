"""
PPG preprocessing v3 for TARNet-style 5-second windows.

Paper alignment:
1. Read 500 Hz PPG recordings from both hands.
2. Downsample with a 5:1 ratio to 100 Hz.
3. Segment each recording into 5-second sliding windows.
4. Use peakposition and IrrHBPosition files for quality filtering.
5. Keep augmentation conservative: paper-style jittering/scaling only, and only
   when the training labels need balancing.

Outputs:
    dataset_v3_tarnet.npz
    TARNet_data_v3/{X,y,F,subject_ids}_{train,test}.npy

X shape:
    (N, 500), because 5 seconds * 100 Hz = 500 points.

F shape:
    (N, 7), optional auxiliary features:
    rr_mean, rr_std, heart_rate, ri_mean, ri_std, auc_mean, clean_beat_count.
"""

import glob
import os

import numpy as np


DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
PREGNANT_DIR = os.path.join(DATA_ROOT, "孕婦組")
CONTROL_DIR = os.path.join(DATA_ROOT, "對照組")
OUTPUT_PATH = os.path.join(DATA_ROOT, "dataset_v3_tarnet.npz")
OUTPUT_DIR = os.path.join(DATA_ROOT, "TARNet_data_v3")

RAW_SAMPLING_RATE = 500
DOWNSAMPLE_FACTOR = 5
MODEL_SAMPLING_RATE = RAW_SAMPLING_RATE // DOWNSAMPLE_FACTOR

WINDOW_SECONDS = 5
STRIDE_SECONDS = 1
WINDOW_LEN = WINDOW_SECONDS * MODEL_SAMPLING_RATE
STRIDE_LEN = STRIDE_SECONDS * MODEL_SAMPLING_RATE

RR_MIN = int(0.30 * RAW_SAMPLING_RATE)
RR_MAX = int(1.50 * RAW_SAMPLING_RATE)
MIN_BEATS_PER_WINDOW = 4
MIN_CLEAN_BEAT_RATIO = 0.80

TEST_SIZE = 0.2
RANDOM_SEED = 42

BALANCE_TRAIN_CLASSES = True
USE_AUGMENTATION = True
JITTER_SIGMA = 0.01
SCALE_RANGE = (0.95, 1.05)

LABEL_PREGNANT = 1
LABEL_CONTROL = 0


def read_rawdata(filepath):
    values = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            try:
                values.append(float(line))
            except ValueError:
                pass
    return np.array(values, dtype=np.float32)


def read_positions(filepath):
    positions = set()
    if not os.path.exists(filepath):
        return positions
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                positions.add(int(line))
    return positions


def find_companion_file(raw_path, suffix):
    exact = raw_path.replace(".wave.rawdata.txt", suffix)
    if os.path.exists(exact):
        return exact

    folder = os.path.dirname(raw_path)
    raw_stem = os.path.basename(raw_path).replace(".wave.rawdata.txt", "")
    candidates = sorted(glob.glob(os.path.join(folder, f"*{suffix}")))
    if not candidates:
        return exact

    scored = []
    for candidate in candidates:
        cand_stem = os.path.basename(candidate).replace(suffix, "")
        if raw_stem.startswith(cand_stem) or cand_stem.startswith(raw_stem):
            score = max(len(raw_stem), len(cand_stem))
        else:
            score = len(os.path.commonprefix([raw_stem, cand_stem]))
        scored.append((score, candidate))

    best_score, best_path = max(scored, key=lambda item: item[0])
    if best_score >= 8:
        return best_path
    return exact


def downsample_mean(raw, factor=DOWNSAMPLE_FACTOR):
    usable_len = (len(raw) // factor) * factor
    if usable_len == 0:
        return np.array([], dtype=np.float32)
    return raw[:usable_len].reshape(-1, factor).mean(axis=1).astype(np.float32)


def linear_detrend(signal):
    if len(signal) < 2:
        return signal
    trend = np.linspace(signal[0], signal[-1], len(signal), dtype=np.float32)
    return signal - trend


def normalize_window(window):
    window = linear_detrend(window.astype(np.float32))
    mn = float(window.min())
    mx = float(window.max())
    if mx - mn < 1e-6:
        return None
    return ((window - mn) / (mx - mn)).astype(np.float32)


def estimate_ri_and_auc(raw, start, end):
    beat = raw[start:end].astype(np.float32)
    if len(beat) < RR_MIN:
        return None

    beat = linear_detrend(beat)
    mn = float(beat.min())
    mx = float(beat.max())
    if mx - mn < 1e-6:
        return None
    beat = (beat - mn) / (mx - mn)

    p1_region_end = max(3, int(0.20 * len(beat)))
    p1 = float(beat[:p1_region_end].max())
    search_start = int(0.25 * len(beat))
    search_end = max(search_start + 1, int(0.85 * len(beat)))
    p2 = float(beat[search_start:search_end].max())
    if p1 < 1e-6:
        return None

    ri = p2 / p1
    auc = float(np.trapezoid(beat) / len(beat))
    return ri, auc


def get_beat_records(raw, peaks, irr_positions):
    records = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        rr = end - start
        is_clean = RR_MIN <= rr <= RR_MAX and start not in irr_positions and end not in irr_positions

        ri = np.nan
        auc = np.nan
        if is_clean:
            ri_auc = estimate_ri_and_auc(raw, start, end)
            if ri_auc is not None:
                ri, auc = ri_auc
            else:
                is_clean = False

        records.append(
            {
                "start": start,
                "end": end,
                "rr": rr,
                "is_clean": is_clean,
                "ri": ri,
                "auc": auc,
            }
        )
    return records


def summarize_window_features(beat_records, win_start_raw, win_end_raw):
    inside = [
        b
        for b in beat_records
        if b["start"] >= win_start_raw and b["end"] <= win_end_raw
    ]
    if len(inside) < MIN_BEATS_PER_WINDOW:
        return None, "too_few_beats"

    clean = [b for b in inside if b["is_clean"]]
    clean_ratio = len(clean) / len(inside)
    if clean_ratio < MIN_CLEAN_BEAT_RATIO or len(clean) < MIN_BEATS_PER_WINDOW:
        return None, "low_quality"

    rr_sec = np.array([b["rr"] / RAW_SAMPLING_RATE for b in clean], dtype=np.float32)
    ri_vals = np.array([b["ri"] for b in clean if not np.isnan(b["ri"])], dtype=np.float32)
    auc_vals = np.array([b["auc"] for b in clean if not np.isnan(b["auc"])], dtype=np.float32)

    if len(ri_vals) == 0 or len(auc_vals) == 0:
        return None, "missing_features"

    features = np.array(
        [
            rr_sec.mean(),
            rr_sec.std(),
            60.0 / rr_sec.mean(),
            ri_vals.mean(),
            ri_vals.std(),
            auc_vals.mean(),
            float(len(clean)),
        ],
        dtype=np.float32,
    )
    return features, "ok"


def process_recording(raw_path):
    peak_path = find_companion_file(raw_path, ".peakposition.txt")
    irr_path = find_companion_file(raw_path, ".IrrHBPosition.txt")

    if not os.path.exists(peak_path):
        return [], [], {"missing_peak": 1}

    raw = read_rawdata(raw_path)
    peaks = sorted(read_positions(peak_path))
    irr_positions = read_positions(irr_path)
    if len(raw) == 0 or len(peaks) < 2:
        return [], [], {"empty_recording": 1}

    downsampled = downsample_mean(raw)
    beat_records = get_beat_records(raw, peaks, irr_positions)

    windows = []
    features = []
    stats = {"ok": 0, "too_few_beats": 0, "low_quality": 0, "missing_features": 0}

    for start_ds in range(0, len(downsampled) - WINDOW_LEN + 1, STRIDE_LEN):
        end_ds = start_ds + WINDOW_LEN
        win_start_raw = start_ds * DOWNSAMPLE_FACTOR
        win_end_raw = end_ds * DOWNSAMPLE_FACTOR

        feat, reason = summarize_window_features(beat_records, win_start_raw, win_end_raw)
        stats[reason] = stats.get(reason, 0) + 1
        if feat is None:
            continue

        window = normalize_window(downsampled[start_ds:end_ds])
        if window is None:
            stats["flat_window"] = stats.get("flat_window", 0) + 1
            continue

        windows.append(window)
        features.append(feat)
        stats["ok"] += 1

    return windows, features, stats


def process_subject(subject_dir):
    raw_files = sorted(glob.glob(os.path.join(subject_dir, "*.wave.rawdata.txt")))
    all_windows = []
    all_features = []
    stats = {}

    for raw_path in raw_files:
        windows, features, rec_stats = process_recording(raw_path)
        all_windows.extend(windows)
        all_features.extend(features)
        for key, value in rec_stats.items():
            stats[key] = stats.get(key, 0) + value

    return all_windows, all_features, stats


def load_group(group_dir):
    subjects = []
    total_stats = {}

    for name in sorted(os.listdir(group_dir)):
        subj_dir = os.path.join(group_dir, name)
        if not os.path.isdir(subj_dir):
            continue

        windows, features, stats = process_subject(subj_dir)
        for key, value in stats.items():
            total_stats[key] = total_stats.get(key, 0) + value

        if len(windows) == 0:
            print(f"  [warning] {name}: no valid windows, skipped")
            continue

        print(f"  {name}: {len(windows)} windows")
        subjects.append((name, windows, features))

    print(f"  quality stats: {total_stats}")
    return subjects


def subject_split(n_subjects, test_size=TEST_SIZE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    ids = np.arange(n_subjects)
    rng.shuffle(ids)
    n_test = max(1, round(n_subjects * test_size))
    return ids[n_test:], ids[:n_test]


def collect(subjects, ids, label):
    X = []
    F = []
    subject_ids = []
    for idx in ids:
        subject_name, windows, features = subjects[idx]
        X.extend(windows)
        F.extend(features)
        subject_ids.extend([subject_name] * len(windows))

    X = np.array(X, dtype=np.float32)
    F = np.array(F, dtype=np.float32)
    y = np.full(len(X), label, dtype=np.int32)
    subject_ids = np.array(subject_ids)
    return X, y, F, subject_ids


def augment_safe(window, feature):
    augmented = window.copy()
    if np.random.random() < 0.5:
        augmented = augmented + np.random.normal(0, JITTER_SIGMA, size=augmented.shape).astype(np.float32)
    if np.random.random() < 0.5:
        scale = np.random.uniform(*SCALE_RANGE)
        augmented = augmented * scale
    augmented = np.clip(augmented, 0.0, 1.0).astype(np.float32)
    return augmented, feature.copy()


def augment_to_balance(X_minority, F_minority, target_count):
    if target_count <= 0 or len(X_minority) == 0:
        return (
            np.empty((0, WINDOW_LEN), dtype=np.float32),
            np.empty((0, F_minority.shape[1]), dtype=np.float32),
        )

    rng = np.random.default_rng(RANDOM_SEED)
    X_aug = []
    F_aug = []
    while len(X_aug) < target_count:
        for idx in rng.permutation(len(X_minority)):
            if len(X_aug) >= target_count:
                break
            aug_x, aug_f = augment_safe(X_minority[idx], F_minority[idx])
            X_aug.append(aug_x)
            F_aug.append(aug_f)

    return np.array(X_aug, dtype=np.float32), np.array(F_aug, dtype=np.float32)


def maybe_balance_train(X_p, y_p, F_p, sid_p, X_c, y_c, F_c, sid_c):
    if not BALANCE_TRAIN_CLASSES or not USE_AUGMENTATION:
        return X_p, y_p, F_p, sid_p, X_c, y_c, F_c, sid_c

    if len(X_p) == len(X_c):
        return X_p, y_p, F_p, sid_p, X_c, y_c, F_c, sid_c

    if len(X_p) < len(X_c):
        target = len(X_c) - len(X_p)
        print(f"Augment pregnant windows to balance: +{target}")
        X_aug, F_aug = augment_to_balance(X_p, F_p, target)
        y_aug = np.full(len(X_aug), LABEL_PREGNANT, dtype=np.int32)
        sid_aug = np.array([f"{sid_p[i % len(sid_p)]}_aug" for i in range(len(X_aug))])
        X_p = np.concatenate([X_p, X_aug])
        y_p = np.concatenate([y_p, y_aug])
        F_p = np.concatenate([F_p, F_aug])
        sid_p = np.concatenate([sid_p, sid_aug])
    else:
        target = len(X_p) - len(X_c)
        print(f"Augment control windows to balance: +{target}")
        X_aug, F_aug = augment_to_balance(X_c, F_c, target)
        y_aug = np.full(len(X_aug), LABEL_CONTROL, dtype=np.int32)
        sid_aug = np.array([f"{sid_c[i % len(sid_c)]}_aug" for i in range(len(X_aug))])
        X_c = np.concatenate([X_c, X_aug])
        y_c = np.concatenate([y_c, y_aug])
        F_c = np.concatenate([F_c, F_aug])
        sid_c = np.concatenate([sid_c, sid_aug])

    return X_p, y_p, F_p, sid_p, X_c, y_c, F_c, sid_c


def save_outputs(X_train, y_train, F_train, sid_train, X_test, y_test, F_test, sid_test):
    np.savez(
        OUTPUT_PATH,
        X_train=X_train,
        y_train=y_train,
        F_train=F_train,
        subject_ids_train=sid_train,
        X_test=X_test,
        y_test=y_test,
        F_test=F_test,
        subject_ids_test=sid_test,
        feature_names=np.array(
            ["rr_mean", "rr_std", "heart_rate", "ri_mean", "ri_std", "auc_mean", "clean_beat_count"]
        ),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "F_train.npy"), F_train)
    np.save(os.path.join(OUTPUT_DIR, "subject_ids_train.npy"), sid_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUTPUT_DIR, "F_test.npy"), F_test)
    np.save(os.path.join(OUTPUT_DIR, "subject_ids_test.npy"), sid_test)


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Loading pregnant group...")
    preg = load_group(PREGNANT_DIR)

    print("\nLoading control group...")
    ctrl = load_group(CONTROL_DIR)

    print("\n" + "=" * 60)
    print("Subject-level split (8:2)...")
    p_train_ids, p_test_ids = subject_split(len(preg), seed=RANDOM_SEED)
    c_train_ids, c_test_ids = subject_split(len(ctrl), seed=RANDOM_SEED)

    X_p_train, y_p_train, F_p_train, sid_p_train = collect(preg, p_train_ids, LABEL_PREGNANT)
    X_p_test, y_p_test, F_p_test, sid_p_test = collect(preg, p_test_ids, LABEL_PREGNANT)
    X_c_train, y_c_train, F_c_train, sid_c_train = collect(ctrl, c_train_ids, LABEL_CONTROL)
    X_c_test, y_c_test, F_c_test, sid_c_test = collect(ctrl, c_test_ids, LABEL_CONTROL)

    X_p_train, y_p_train, F_p_train, sid_p_train, X_c_train, y_c_train, F_c_train, sid_c_train = (
        maybe_balance_train(
            X_p_train,
            y_p_train,
            F_p_train,
            sid_p_train,
            X_c_train,
            y_c_train,
            F_c_train,
            sid_c_train,
        )
    )

    X_train = np.concatenate([X_p_train, X_c_train])
    y_train = np.concatenate([y_p_train, y_c_train])
    F_train = np.concatenate([F_p_train, F_c_train])
    sid_train = np.concatenate([sid_p_train, sid_c_train])

    X_test = np.concatenate([X_p_test, X_c_test])
    y_test = np.concatenate([y_p_test, y_c_test])
    F_test = np.concatenate([F_p_test, F_c_test])
    sid_test = np.concatenate([sid_p_test, sid_c_test])

    rng = np.random.default_rng(RANDOM_SEED)
    train_perm = rng.permutation(len(X_train))
    test_perm = rng.permutation(len(X_test))
    X_train, y_train, F_train, sid_train = (
        X_train[train_perm],
        y_train[train_perm],
        F_train[train_perm],
        sid_train[train_perm],
    )
    X_test, y_test, F_test, sid_test = (
        X_test[test_perm],
        y_test[test_perm],
        F_test[test_perm],
        sid_test[test_perm],
    )

    save_outputs(X_train, y_train, F_train, sid_train, X_test, y_test, F_test, sid_test)

    print("\n" + "=" * 60)
    print("Done: v3 TARNet preprocessing")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"F_train: {F_train.shape}, F_test: {F_test.shape}")
    print(f"Train labels: pregnant={(y_train == 1).sum()} control={(y_train == 0).sum()}")
    print(f"Test labels:  pregnant={(y_test == 1).sum()} control={(y_test == 0).sum()}")
    print(f"Window: {WINDOW_SECONDS}s at {MODEL_SAMPLING_RATE} Hz = {WINDOW_LEN} points")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved npy files: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
