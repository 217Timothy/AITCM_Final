# AITCM_Final

## Data Preprocessing

本專案提供兩個前處理腳本，皆使用 Subject-level train/test split（依人切分，避免 data leakage）。

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

**新增 Augmentation 方法（共五種，每次隨機挑一種）：**
- Jittering：加入高斯白雜訊
- Scaling：隨機縮放振幅
- **Time Warping**：對時間軸做非線性扭曲，模擬心率細微變動
- **Magnitude Warping**：對振幅疊加 smooth 隨機曲線，模擬血壓輕微波動
- **Window Slicing**：隨機截取子段後 resize 回原長度，增加局部特徵的不變性
- 複製 3 倍

**輸出：** `dataset_v2.npz`、`TARNet_data_v2/`（X_train, y_train, X_test, y_test）

---

## Requirements

```
numpy
```

## Usage

```bash
python preprocess.py    # 產生 v1 資料集
python preprocess_v2.py # 產生 v2 資料集
```
