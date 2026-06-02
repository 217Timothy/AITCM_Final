"""
PPG 脈搏資料前處理 v2（改良版）
Paper: Pregnancy detection on a smartphone using deep learning (2025)

相比 v1 的改良：
  [Preprocessing]
  1. 使用 IrrHBPosition.txt 過濾不規則心跳波形
  2. RR interval 過濾（<0.3s 或 >1.5s 的異常波形）
  3. Linear detrend：移除每個波形的基線漂移

  [Augmentation] 新增三種方法（共五種）
  4. Time Warping：對時間軸做非線性扭曲
  5. Magnitude Warping：對振幅疊加 smooth 隨機曲線
  6. Window Slicing：隨機截取子段後 resize 回原長度
  原有：Jittering、Scaling

只需要 numpy，不需要 scipy 或 sklearn。
"""

import os
import glob
import numpy as np

# ── 路徑設定 ────────────────────────────────────────────────
DATA_ROOT    = os.path.dirname(os.path.abspath(__file__))
PREGNANT_DIR = os.path.join(DATA_ROOT, "孕婦組")
CONTROL_DIR  = os.path.join(DATA_ROOT, "對照組")
OUTPUT_PATH  = os.path.join(DATA_ROOT, "dataset_v2.npz")

# ── 參數設定 ─────────────────────────────────────────────────
SAMPLING_RATE = 500        # Hz
TARGET_LEN    = 250        # 每個波形插值到的固定點數

# RR interval 過濾閾值（單位：samples @ 500Hz）
RR_MIN = int(0.30 * SAMPLING_RATE)   # 0.30s → 200 bpm，太快
RR_MAX = int(1.50 * SAMPLING_RATE)   # 1.50s →  40 bpm，太慢

TEST_SIZE   = 0.2
RANDOM_SEED = 42

# Augmentation 設定
AUG_TIMES = 3   # 每個波形產生幾個 augmented 版本（v1 是 4，這裡改小因為方法更多樣）

LABEL_PREGNANT = 1
LABEL_CONTROL  = 0


# ════════════════════════════════════════════════════════════
#  工具函式：讀檔
# ════════════════════════════════════════════════════════════
def read_rawdata(filepath):
    values = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            try:
                values.append(float(line))
            except ValueError:
                pass  # 跳過 "Sampling rate = ..." 這類行
    return np.array(values, dtype=np.float32)


def read_positions(filepath):
    """讀取 peakposition 或 IrrHBPosition，回傳 set of int"""
    positions = set()
    if not os.path.exists(filepath):
        return positions
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                positions.add(int(line))
    return positions


# ════════════════════════════════════════════════════════════
#  Preprocessing：波形切割 + 清理
# ════════════════════════════════════════════════════════════
def linear_detrend(wave):
    """
    移除線性基線漂移（baseline wander）。
    用波形首尾的平均值拉一條直線，再減掉。
    """
    n = len(wave)
    trend = np.linspace(wave[0], wave[-1], n)
    return wave - trend


def cut_and_clean_waves(raw, peaks, irr_positions):
    """
    用 peak-to-peak 切出單波，並做：
    1. RR interval 過濾（太快 / 太慢的心跳）
    2. IrrHB 過濾（不規則心跳位置）
    3. Linear detrend（去除基線漂移）
    4. Min-max 正規化
    5. 插值到固定長度 TARGET_LEN
    """
    waves = []
    removed_rr  = 0
    removed_irr = 0

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end   = peaks[i + 1]
        rr    = end - start

        # --- 過濾 1：RR interval 太短或太長 ---
        if rr < RR_MIN or rr > RR_MAX:
            removed_rr += 1
            continue

        # --- 過濾 2：這個波形的起點或終點是不規則心跳 ---
        if start in irr_positions or end in irr_positions:
            removed_irr += 1
            continue

        wave = raw[start:end].copy()

        # --- 去除基線漂移 ---
        wave = linear_detrend(wave)

        # --- Min-max 正規化 ---
        mn, mx = wave.min(), wave.max()
        if mx - mn < 1e-6:
            continue
        wave = (wave - mn) / (mx - mn)

        # --- 插值到固定長度 ---
        x_old = np.linspace(0, 1, len(wave))
        x_new = np.linspace(0, 1, TARGET_LEN)
        wave  = np.interp(x_new, x_old, wave).astype(np.float32)

        waves.append(wave)

    return waves, removed_rr, removed_irr


def process_subject(subject_dir):
    """
    處理單一受試者，合併左右手，回傳所有乾淨波形。
    """
    rawdata_files = glob.glob(os.path.join(subject_dir, "*.wave.rawdata.txt"))

    all_waves  = []
    total_removed_rr  = 0
    total_removed_irr = 0

    for raw_path in rawdata_files:
        # 找對應 peakposition
        base      = raw_path.replace(".wave.rawdata.txt", "")
        peak_path = base + ".peakposition.txt"
        irr_path  = base + ".IrrHBPosition.txt"

        if not os.path.exists(peak_path):
            candidates = glob.glob(os.path.join(subject_dir, "*.peakposition.txt"))
            if candidates:
                peak_path = candidates[0]
            else:
                continue

        raw            = read_rawdata(raw_path)
        peaks          = sorted(read_positions(peak_path))
        irr_positions  = read_positions(irr_path)   # 沒有檔案會回傳空 set

        if len(raw) == 0 or len(peaks) < 2:
            continue

        waves, rm_rr, rm_irr = cut_and_clean_waves(raw, peaks, irr_positions)
        all_waves.extend(waves)
        total_removed_rr  += rm_rr
        total_removed_irr += rm_irr

    return all_waves, total_removed_rr, total_removed_irr


# ════════════════════════════════════════════════════════════
#  Augmentation：五種方法
# ════════════════════════════════════════════════════════════
def aug_jitter(wave, sigma=0.02):
    """加入高斯白雜訊"""
    noise = np.random.normal(0, sigma, size=wave.shape).astype(np.float32)
    return np.clip(wave + noise, 0, 1).astype(np.float32)


def aug_scaling(wave, scale_range=(0.85, 1.15)):
    """隨機縮放振幅"""
    scale = np.random.uniform(*scale_range)
    return np.clip(wave * scale, 0, 1).astype(np.float32)


def aug_time_warp(wave, n_knots=4, sigma=0.15):
    """
    Time Warping：對時間軸做非線性扭曲。
    原理：隨機生成一條 smooth 的「速度曲線」，讓波形局部加速或減速。
    參考：Um et al. 2017, Data Augmentation of Wearable Sensor Data
    """
    n = len(wave)

    # 生成隨機 knot（控制點）
    knot_x = np.linspace(0, 1, n_knots + 2)
    knot_y = np.ones(n_knots + 2)
    knot_y[1:-1] += np.random.normal(0, sigma, n_knots)  # 隨機擾動速度

    # 插值出每個點的速度
    x_full  = np.linspace(0, 1, n)
    speed   = np.interp(x_full, knot_x, knot_y)
    speed   = np.abs(speed) + 1e-6  # 確保速度為正

    # 積分速度得到扭曲後的時間軸
    warped_t = np.cumsum(speed)
    warped_t = (warped_t - warped_t[0]) / (warped_t[-1] - warped_t[0])

    # 用扭曲後的時間軸重新採樣
    orig_t  = np.linspace(0, 1, n)
    warped  = np.interp(orig_t, warped_t, wave).astype(np.float32)

    return np.clip(warped, 0, 1).astype(np.float32)


def aug_magnitude_warp(wave, n_knots=4, sigma=0.15):
    """
    Magnitude Warping：對振幅疊加一條 smooth 的隨機乘數曲線。
    讓波形的不同位置被放大或縮小不同程度。
    """
    n = len(wave)

    # 生成 smooth 隨機乘數
    knot_x  = np.linspace(0, 1, n_knots + 2)
    knot_y  = np.ones(n_knots + 2) + np.random.normal(0, sigma, n_knots + 2)
    x_full  = np.linspace(0, 1, n)
    multiplier = np.interp(x_full, knot_x, knot_y).astype(np.float32)

    warped = wave * multiplier
    return np.clip(warped, 0, 1).astype(np.float32)


def aug_window_slice(wave, crop_ratio_range=(0.7, 0.9)):
    """
    Window Slicing：隨機截取一段子波形，再 resize 回原長度。
    讓模型學習波形局部特徵的不變性。
    """
    n = len(wave)
    ratio = np.random.uniform(*crop_ratio_range)
    crop_len = max(int(n * ratio), 10)

    start = np.random.randint(0, n - crop_len + 1)
    sliced = wave[start:start + crop_len]

    # resize 回 TARGET_LEN
    x_old  = np.linspace(0, 1, len(sliced))
    x_new  = np.linspace(0, 1, n)
    result = np.interp(x_new, x_old, sliced).astype(np.float32)
    return np.clip(result, 0, 1).astype(np.float32)


AUG_METHODS = [aug_jitter, aug_scaling, aug_time_warp, aug_magnitude_warp, aug_window_slice]


def augment_one(wave):
    """隨機挑一種 augmentation 方法套用"""
    method = np.random.choice(AUG_METHODS)
    return method(wave)


def augment_waves(waves, times=AUG_TIMES):
    """對一組波形做 augmentation，回傳 augmented 版本（不含原始）"""
    augmented = []
    for wave in waves:
        for _ in range(times):
            augmented.append(augment_one(wave))
    return augmented


# ════════════════════════════════════════════════════════════
#  Subject-level split（不用 sklearn）
# ════════════════════════════════════════════════════════════
def subject_split(ids, test_size=TEST_SIZE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_test = max(1, round(len(ids) * test_size))
    return ids[n_test:], ids[:n_test]   # train, test


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════
def load_group(group_dir, label):
    subjects = []
    total_rr = total_irr = 0

    for name in sorted(os.listdir(group_dir)):
        subj_dir = os.path.join(group_dir, name)
        if not os.path.isdir(subj_dir):
            continue

        waves, rm_rr, rm_irr = process_subject(subj_dir)
        total_rr  += rm_rr
        total_irr += rm_irr

        if len(waves) == 0:
            print(f"  [警告] {name} 無有效波形，跳過")
            continue

        print(f"  {name}: {len(waves)} 波形  "
              f"(過濾 RR異常:{rm_rr}  IrrHB:{rm_irr})")
        subjects.append((name, waves))

    print(f"  → 共過濾 RR異常:{total_rr}  IrrHB:{total_irr} 個波形")
    return subjects


def collect(subjects, ids, label):
    waves = []
    for i in ids:
        waves.extend(subjects[i][1])
    X = np.array(waves, dtype=np.float32)
    y = np.full(len(X), label, dtype=np.int32)
    return X, y


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("讀取孕婦組...")
    preg = load_group(PREGNANT_DIR, LABEL_PREGNANT)

    print("\n讀取對照組...")
    ctrl = load_group(CONTROL_DIR,  LABEL_CONTROL)

    # Subject-level split
    print("\n" + "=" * 55)
    print("Subject-level split (8:2)...")
    p_train_ids, p_test_ids = subject_split(range(len(preg)))
    c_train_ids, c_test_ids = subject_split(range(len(ctrl)))

    X_p_train, y_p_train = collect(preg, p_train_ids, LABEL_PREGNANT)
    X_p_test,  y_p_test  = collect(preg, p_test_ids,  LABEL_PREGNANT)
    X_c_train, y_c_train = collect(ctrl, c_train_ids, LABEL_CONTROL)
    X_c_test,  y_c_test  = collect(ctrl, c_test_ids,  LABEL_CONTROL)

    # Augmentation（只對 train 的孕婦）
    print("\n" + "=" * 55)
    print(f"Augmentation (x{AUG_TIMES})，使用 5 種方法隨機組合...")
    aug_waves = augment_waves(list(X_p_train), times=AUG_TIMES)
    X_aug = np.array(aug_waves, dtype=np.float32)
    y_aug = np.full(len(X_aug), LABEL_PREGNANT, dtype=np.int32)

    # 合併
    X_train = np.concatenate([X_p_train, X_aug, X_c_train])
    y_train = np.concatenate([y_p_train, y_aug, y_c_train])
    X_test  = np.concatenate([X_p_test,  X_c_test])
    y_test  = np.concatenate([y_p_test,  y_c_test])

    # Shuffle
    rng = np.random.default_rng(RANDOM_SEED)
    X_train = X_train[rng.permutation(len(X_train))]
    y_train = y_train[rng.permutation(len(y_train))]
    X_test  = X_test[rng.permutation(len(X_test))]
    y_test  = y_test[rng.permutation(len(y_test))]

    # 存成 .npz（一個檔案）
    np.savez(OUTPUT_PATH,
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test)

    # 同時存成 .npy（四個分開的檔案，給同學的 TARNet code 用）
    npy_dir = os.path.join(DATA_ROOT, "TARNet_data_v2")
    os.makedirs(npy_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, "X_train.npy"), X_train)
    np.save(os.path.join(npy_dir, "y_train.npy"), y_train)
    np.save(os.path.join(npy_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(npy_dir, "y_test.npy"),  y_test)

    # 統計
    print("\n" + "=" * 55)
    print("✅ 前處理 v2 完成！")
    print(f"\n  波形維度：{TARGET_LEN} 點")
    print(f"\n  孕婦組：{len(preg)} 人  →  Train {len(p_train_ids)} / Test {len(p_test_ids)}")
    print(f"  對照組：{len(ctrl)} 人  →  Train {len(c_train_ids)} / Test {len(c_test_ids)}")

    n_p_aug = (y_train == LABEL_PREGNANT).sum()
    n_c_tr  = (y_train == LABEL_CONTROL).sum()
    print(f"\n  Train set：{len(X_train)} 筆")
    print(f"    孕婦（原始+aug）：{n_p_aug}  |  對照：{n_c_tr}")
    print(f"    孕婦/對照比例：{n_p_aug/n_c_tr:.2f}")
    print(f"\n  Test set：{len(X_test)} 筆")
    print(f"    孕婦：{(y_test==1).sum()}  |  對照：{(y_test==0).sum()}")
    print(f"\n  儲存至：{OUTPUT_PATH}")
    print(f"  .npy 儲存至：{npy_dir}/")


if __name__ == "__main__":
    main()
