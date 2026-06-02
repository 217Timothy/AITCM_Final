"""
PPG 脈搏資料前處理腳本
Paper: Pregnancy detection on a smartphone using deep learning (2025)

前處理流程：
1. 讀取所有受試者的 wave.rawdata.txt 和 peakposition.txt
2. 用 peak-to-peak 切出單波並做 min-max 正規化
3. Subject-level train/test split（依人切，避免 data leakage）
4. 對 train set 孕婦波形做 data augmentation（jittering + scaling）
5. 儲存成 .npz 格式
"""

import os
import glob
import numpy as np


def train_test_split(ids, test_size=0.2, random_state=42):
    """簡單的 subject-level split，不需要 sklearn"""
    rng = np.random.default_rng(random_state)
    ids = list(ids)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_size))
    return ids[n_test:], ids[:n_test]

# ── 設定 ────────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.dirname(os.path.abspath(__file__))
PREGNANT_DIR   = os.path.join(DATA_ROOT, "孕婦組")
CONTROL_DIR    = os.path.join(DATA_ROOT, "對照組")
OUTPUT_PATH    = os.path.join(DATA_ROOT, "dataset.npz")

SAMPLING_RATE  = 500          # Hz
TARGET_LEN     = 250          # 每個波形插值到固定長度（約 0.5 秒的波形）
MIN_WAVE_LEN   = 50           # 過濾太短的異常波（< 0.1 秒）
MAX_WAVE_LEN   = 1000         # 過濾太長的異常波（> 2 秒）
TEST_SIZE      = 0.2          # 20% 的人做 test
RANDOM_SEED    = 42

# Data augmentation 參數（只用在 train 的孕婦資料）
AUG_TIMES      = 4            # 複製 4 倍
JITTER_SIGMA   = 0.02         # jittering 雜訊強度
SCALE_RANGE    = (0.9, 1.1)   # scaling 範圍

LABEL_PREGNANT = 1
LABEL_CONTROL  = 0


# ── 工具函式 ─────────────────────────────────────────────────────────────────
def read_rawdata(filepath):
    """讀取 wave.rawdata.txt，回傳 numpy array（跳過第一行 sampling rate）"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # 第一行可能是 "Sampling rate = 500 points/sec"，跳過非數字行
    values = []
    for line in lines:
        line = line.strip()
        if line.lstrip("-").isdigit():
            values.append(int(line))
    return np.array(values, dtype=np.float32)


def read_peakposition(filepath):
    """讀取 peakposition.txt，回傳 peak index list"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    peaks = []
    for line in lines:
        line = line.strip()
        if line.isdigit():
            peaks.append(int(line))
    return peaks


def cut_waves(raw, peaks):
    """
    用 peak-to-peak 切出單波。
    回傳 list of 1D numpy arrays（每個波形長度不同）。
    """
    waves = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end   = peaks[i + 1]
        wave  = raw[start:end]
        wave_len = len(wave)
        if MIN_WAVE_LEN <= wave_len <= MAX_WAVE_LEN:
            waves.append(wave)
    return waves


def normalize_wave(wave):
    """Min-max 正規化到 [0, 1]"""
    mn, mx = wave.min(), wave.max()
    if mx - mn < 1e-6:
        return np.zeros_like(wave)
    return (wave - mn) / (mx - mn)


def interpolate_wave(wave, target_len=TARGET_LEN):
    """把每個波形插值到固定長度"""
    x_old = np.linspace(0, 1, len(wave))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, wave).astype(np.float32)


def process_subject(subject_dir):
    """
    處理單一受試者資料夾，回傳該受試者的所有波形 list。
    一個人可能有左右手兩筆資料，全部合併。
    """
    rawdata_files = glob.glob(os.path.join(subject_dir, "*.wave.rawdata.txt"))
    # 有些命名方式不同（如 右.wave.rawdata.txt），用更寬鬆的匹配
    if not rawdata_files:
        rawdata_files = glob.glob(os.path.join(subject_dir, "*rawdata*"))

    all_waves = []
    for raw_path in rawdata_files:
        # 找對應的 peakposition 檔（同前綴，或同資料夾）
        basename = raw_path.replace(".wave.rawdata.txt", "").replace("rawdata.txt", "")
        peak_path = basename + ".peakposition.txt"

        if not os.path.exists(peak_path):
            # 嘗試在同資料夾找 peakposition
            peak_candidates = glob.glob(os.path.join(subject_dir, "*.peakposition.txt"))
            if peak_candidates:
                peak_path = peak_candidates[0]
            else:
                print(f"  [警告] 找不到 peakposition：{raw_path}，跳過")
                continue

        raw   = read_rawdata(raw_path)
        peaks = read_peakposition(peak_path)

        if len(raw) == 0 or len(peaks) < 2:
            continue

        waves = cut_waves(raw, peaks)
        # 正規化 + 插值到固定長度
        processed = [interpolate_wave(normalize_wave(w)) for w in waves]
        all_waves.extend(processed)

    return all_waves


# ── Data Augmentation ────────────────────────────────────────────────────────
def augment_wave(wave):
    """對單一波形產生一個 augmented 版本（jittering + scaling）"""
    noise = np.random.normal(0, JITTER_SIGMA, size=wave.shape).astype(np.float32)
    scale = np.random.uniform(*SCALE_RANGE)
    augmented = wave * scale + noise
    # clip 回 [0, 1]
    return np.clip(augmented, 0, 1).astype(np.float32)


def augment_waves(waves, times=AUG_TIMES):
    """對一組波形做 augmentation，回傳 original + augmented"""
    augmented = []
    for wave in waves:
        for _ in range(times):
            augmented.append(augment_wave(wave))
    return augmented


# ── 主流程 ──────────────────────────────────────────────────────────────────
def load_all_subjects(group_dir, label):
    """
    讀取整個群組（孕婦/對照）的所有受試者。
    回傳：
        subjects: list of (subject_id, waves_list)
    """
    subjects = []
    for subj_name in sorted(os.listdir(group_dir)):
        subj_dir = os.path.join(group_dir, subj_name)
        if not os.path.isdir(subj_dir):
            continue
        waves = process_subject(subj_dir)
        if len(waves) == 0:
            print(f"  [警告] {subj_name} 沒有有效波形，跳過")
            continue
        print(f"  {subj_name}: {len(waves)} 個波形")
        subjects.append((subj_name, waves))
    return subjects


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 50)
    print("讀取孕婦組...")
    pregnant_subjects = load_all_subjects(PREGNANT_DIR, LABEL_PREGNANT)

    print("\n讀取對照組...")
    control_subjects  = load_all_subjects(CONTROL_DIR, LABEL_CONTROL)

    # ── Subject-level split ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("做 Subject-level train/test split (8:2)...")

    p_ids = list(range(len(pregnant_subjects)))
    c_ids = list(range(len(control_subjects)))

    p_train_ids, p_test_ids = train_test_split(p_ids, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    c_train_ids, c_test_ids = train_test_split(c_ids, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    def collect(subjects, ids, label):
        waves = []
        for i in ids:
            waves.extend(subjects[i][1])
        X = np.array(waves)
        y = np.full(len(X), label, dtype=np.int32)
        return X, y

    X_p_train, y_p_train = collect(pregnant_subjects, p_train_ids, LABEL_PREGNANT)
    X_p_test,  y_p_test  = collect(pregnant_subjects, p_test_ids,  LABEL_PREGNANT)
    X_c_train, y_c_train = collect(control_subjects,  c_train_ids, LABEL_CONTROL)
    X_c_test,  y_c_test  = collect(control_subjects,  c_test_ids,  LABEL_CONTROL)

    # ── Data Augmentation（只對 train 的孕婦）────────────────────────────────
    print("\n" + "=" * 50)
    print(f"對 train 孕婦波形做 augmentation（x{AUG_TIMES}）...")
    aug_waves = augment_waves(list(X_p_train), times=AUG_TIMES)
    X_aug = np.array(aug_waves, dtype=np.float32)
    y_aug = np.full(len(X_aug), LABEL_PREGNANT, dtype=np.int32)

    # 合併 train（原始 + augmented）
    X_train = np.concatenate([X_p_train, X_aug, X_c_train], axis=0)
    y_train = np.concatenate([y_p_train, y_aug, y_c_train], axis=0)

    # 合併 test（不做 augmentation）
    X_test = np.concatenate([X_p_test, X_c_test], axis=0)
    y_test = np.concatenate([y_p_test, y_c_test], axis=0)

    # Shuffle
    train_idx = np.random.permutation(len(X_train))
    test_idx  = np.random.permutation(len(X_test))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test,  y_test  = X_test[test_idx],   y_test[test_idx]

    # ── 儲存 ────────────────────────────────────────────────────────────────
    np.savez(OUTPUT_PATH, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    npy_dir = os.path.join(DATA_ROOT, "TARNet_data_v1")
    os.makedirs(npy_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, "X_train.npy"), X_train)
    np.save(os.path.join(npy_dir, "y_train.npy"), y_train)
    np.save(os.path.join(npy_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(npy_dir, "y_test.npy"),  y_test)

    # ── 統計資訊 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("✅ 前處理完成！統計資訊：")
    print(f"\n  波形維度（固定長度）：{TARGET_LEN} 個點（{TARGET_LEN/SAMPLING_RATE:.2f} 秒/波）")
    print(f"\n  孕婦組（共 {len(pregnant_subjects)} 人）")
    print(f"    Train：{len(p_train_ids)} 人")
    print(f"    Test ：{len(p_test_ids)} 人")
    print(f"\n  對照組（共 {len(control_subjects)} 人）")
    print(f"    Train：{len(c_train_ids)} 人")
    print(f"    Test ：{len(c_test_ids)} 人")
    print(f"\n  Train set：{len(X_train)} 筆")
    print(f"    孕婦（原始）：{len(X_p_train)}")
    print(f"    孕婦（augmented）：{len(X_aug)}")
    print(f"    對照：{len(X_c_train)}")
    print(f"\n  Test set：{len(X_test)} 筆")
    print(f"    孕婦：{len(X_p_test)}")
    print(f"    對照：{len(X_c_test)}")
    print(f"\n  儲存至：{OUTPUT_PATH}")
    print(f"  .npy 儲存至：{npy_dir}/")


if __name__ == "__main__":
    main()
