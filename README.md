# AITCM_Final

## Data Preprocessing

本專案提供三個前處理腳本，皆使用 Subject-level train/test split（依人切分，避免 data leakage）。

---

### preprocess.py（v1 — 依照 Paper 方法）

依照論文 *Pregnancy detection on a smartphone using deep learning (2025)* 的描述實作。

**前處理流程：**
1. 讀取原始 PPG 訊號（500 Hz）
2. 依 `peakposition.txt` 的峰值位置，切出 peak-to-peak 單波
3. 過濾過短（< 50 points）或過長（> 1000 points）的異常波形
4. Min-max 正規化每個波形至 [0, 1]
5. 插值至固定長度 250 點

**Data Augmentation（只對 train set 的孕婦波形）：**
- Jittering：加入高斯白雜訊
- Scaling：隨機縮放振幅
- 複製 4 倍以平衡 class imbalance

**輸出：** `dataset.npz`、`TARNet_data_v1/`（X_train, y_train, X_test, y_test）

---

### preprocess_v2.py（v2 — 新增與改良方法）

在 v1 基礎上新增資料清理步驟與更多樣的 augmentation 方法。

**新增前處理步驟：**
1. **IrrHB 過濾**：讀取 `IrrHBPosition.txt`，移除含不規則心跳的波形
2. **RR interval 過濾**：移除心率超過 200 bpm（< 0.3s）或低於 40 bpm（> 1.5s）的異常波形
3. **Linear Detrend**：對每個波形做線性去趨勢，移除基線漂移（baseline wander）後再正規化

**Augmentation：**
- Jittering：加入高斯白雜訊
- Scaling：隨機縮放振幅
- Time Warping / Magnitude Warping / Window Slicing 保留在 v2 腳本中，但不建議作為最終版本，因為可能破壞 PPG 形態與 RI 等波形特徵。
- 目前 v2 已改為只在需要時補到 train set 類別接近平衡。

**輸出：** `dataset_v2.npz`、`TARNet_data_v2/`（X_train, y_train, X_test, y_test）

---

### preprocess_v3_tarnet.py（v3 — 貼近 Paper / TARNet 的 5 秒 sliding window）

此版本依論文描述建立較適合 TARNet 類 time-series model 的資料格式。

**前處理流程：**
1. 讀取 500 Hz 原始 PPG 訊號
2. 使用 5:1 downsampling，轉為 100 Hz
3. 以 5 秒 sliding window 切段，每個 window 為 500 點
4. 使用 `peakposition.txt` 計算 RR interval，並用 `IrrHBPosition.txt` 做品質過濾
5. 對每個 window 做 linear detrend 與 min-max 正規化
6. 保留 optional auxiliary features：`rr_mean`, `rr_std`, `heart_rate`, `ri_mean`, `ri_std`, `auc_mean`, `clean_beat_count`

**Data Augmentation：**
- 僅使用論文提到且較保守的 jittering / scaling
- 只用在 train set，且只在類別不平衡時補到平衡
- 不使用 time warping、magnitude warping、window slicing

**輸出：** `dataset_v3_tarnet.npz`、`TARNet_data_v3/`

目前產生的資料：
- `X_train`: `(26674, 500)`
- `X_test`: `(6421, 500)`
- `F_train`: `(26674, 7)`
- Train labels：孕婦 13337 / 對照 13337
- Test labels：孕婦 3829 / 對照 2592
- Train/Test subject overlap：0

---

## TARNet Training Pipeline

### tarnet_v3_pipeline.py

`tarnet_v3_pipeline.py` 是把 v3 的 5 秒 sliding-window 資料轉成 TARNet-style 資料夾格式的工具。

它不是重新做 raw data preprocessing，而是接在 `preprocess_v3_tarnet.py` 後面使用。

**它做的事情：**
1. 讀取 `dataset_v3_tarnet.npz`
2. 檢查 train/test 是否有受試者重疊，避免 data leakage
3. 將 `X_train`, `X_test` 從 `(N, 500)` 轉成 TARNet 較常用的 `(N, 500, 1)`
4. 輸出到 `TARNet/data/PREG_5SEC_SUBJECT_CLEAN/`
5. 保留 `train_subject_ids.npy` 和 `test_subject_ids.npy`，方便做 subject-level voting

**為什麼需要這支腳本：**
- 同學 notebook 原本寫死 Colab 路徑 `/content/...`
- notebook 的 wave-level split 版本會讓同一位受試者同時出現在 train/test
- notebook 使用 peak-to-peak 單波 `(N, 128, 1)`，不是 paper 提到的 5 秒 window
- notebook 沒有使用我們 v3 裡的 IrrHB / RR quality filtering 結果

**目前轉出的 TARNet 資料格式：**
- `X_train`: `(26674, 500, 1)`
- `X_test`: `(6421, 500, 1)`
- Train labels：孕婦 13337 / 對照 13337
- Train/Test subject overlap：0

**使用方式：**

只準備 TARNet 資料，不訓練：

```bash
python3 preprocess_v3_tarnet.py
python3 tarnet_v3_pipeline.py --prepare-only
```

### train_tarnet_v3.py

`train_tarnet_v3.py` 是本專案自己實作的 TARNet-inspired 訓練程式，不需要 pull 官方 TARNet repo。

**模型設計：**
1. 輸入 5 秒 PPG window：`(N, 500)`
2. 使用 linear projection 將單通道 PPG 轉成 embedding
3. 加入 sinusoidal positional encoding
4. 使用 Transformer Encoder 建立 time-series representation
5. 使用 classification head 做孕婦 / 對照組二分類
6. 同時加入 masked reconstruction loss，讓模型在分類之外也學會重建被遮住的 PPG 時間點
7. 可選擇加入 v3 產生的輔助特徵：`rr_mean`, `rr_std`, `heart_rate`, `ri_mean`, `ri_std`, `auc_mean`, `clean_beat_count`

**訓練流程：**
- 從 `dataset_v3_tarnet.npz` 讀資料
- 保持原本 subject-level test split
- 再從 train subjects 中切出 validation subjects
- 用 validation accuracy 選 best model
- 最後輸出 window-level 與 subject-level test results

**輸出：**
- `trained_models/self_tarnet_v3/best_model.pt`
- `trained_models/self_tarnet_v3/last_checkpoint.pt`
- `trained_models/self_tarnet_v3/results.json`

**使用方式：**

```bash
python3 preprocess_v3_tarnet.py
python3 train_tarnet_v3.py --epochs 80 --use-features
```

如果訓練中斷，可以從上一個 epoch 的 checkpoint 繼續：

```bash
python3 train_tarnet_v3.py --epochs 80 --use-features --resume
```

---

## Requirements

```bash
pip install -r requirements.txt
```

主要套件：
- `numpy`：資料前處理與 `.npy/.npz` 儲存
- `matplotlib`：`check_dataset.py` 視覺化
- `torch`：`train_tarnet_v3.py` 模型訓練

## Usage

```bash
python preprocess.py    # 產生 v1 資料集
python preprocess_v2.py # 產生 v2 資料集
python preprocess_v3_tarnet.py # 產生 v3 TARNet 5 秒 window 資料集
python tarnet_v3_pipeline.py --prepare-only # 轉成 TARNet 可讀格式
python train_tarnet_v3.py --epochs 80 --use-features # 訓練自寫 TARNet-inspired 模型
```
