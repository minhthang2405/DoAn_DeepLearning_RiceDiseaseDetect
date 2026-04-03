#!/usr/bin/env python3
"""
Generate rice_disease_prediction_tpu.ipynb
10 Steps clearly separated — Models: ConvNeXtSmall, MobileNetV3Large, EfficientNetV2S on TPU
"""
import json
import textwrap
import sys

# Windows consoles can default to cp1252/cp932, which breaks printing symbols like "✓".
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def md(source):
    if isinstance(source, str):
        source = textwrap.dedent(source).strip('\n').split('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": [l + '\n' for l in source[:-1]] + [source[-1]]}

def code(source):
    if isinstance(source, str):
        source = textwrap.dedent(source).strip('\n').split('\n')
    return {"cell_type": "code", "metadata": {}, "source": [l + '\n' for l in source[:-1]] + [source[-1]], "outputs": [], "execution_count": None}

cells = []

# ─────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────
cells.append(md("""# 🌾 Phân loại Bệnh Lá Lúa — 10 Bước ML Pipeline (GPU T4x2)
**Khởi tạo**: GPU Kép T4x2 (MirroredStrategy)
**Models**: ConvNeXtSmall, MobileNetV3Large, EfficientNetV2S, Proposed_ConvNeXtTiny_SE
**Dữ liệu**: 4 bộ dữ liệu lấy từ Kaggle kết hợp"""))

# ─────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────
cells.append(md("## ⚙️ Thiết lập: Imports, Khởi tạo GPU & Cấu hình"))
cells.append(code("""import os, warnings, hashlib, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import gc
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
import kagglehub

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Khởi tạo GPU Kép (T4x2) ──
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Sử dụng {strategy.num_replicas_in_sync} GPU(s) T4')
except Exception as e:
    print('Lỗi khởi tạo GPU:', e)
    strategy = tf.distribute.get_strategy()

IMG_SIZE       = 224
BATCH_SIZE     = 32 * strategy.num_replicas_in_sync
EPOCHS_P1      = 10   # Phase 1: đóng băng (frozen base)
EPOCHS_P2      = 20   # Phase 2: rã đông 30% layer trên cùng
EPOCHS_P3      = 5    # Phase 3: tinh chỉnh toàn bộ (full fine-tune)

print(f"Số Replicas : {strategy.num_replicas_in_sync}")
print(f"Kích cỡ Batch: {BATCH_SIZE}")
print(f"Phiên bản TF : {tf.__version__}")
print("Thiết lập ✓")"""))

# ─────────────────────────────────────────────────────
# STEP 1
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 1: Hiểu bài toán (Problem Understanding)
> **Đặc trưng (Feature) ? Mục tiêu (Target) ? Phân loại (Classification) hay Hồi quy (Regression) ?**

- **Feature**: Ảnh lá lúa bị bệnh hoặc khỏe mạnh (pixel data).
- **Target**: Tên loại bệnh (7 lớp).
- **Loại bài toán**: Phân loại Đa lớp (Multi-class Classification).

| Item | Detail |
|---|---|
| **Problem type** | Multi-class Image Classification |
| **Features (X)** | RGB images of rice leaves (224×224×3) |
| **Target (y)** | Disease category — 7 classes |
| **Classes** | Bacterial Leaf Blight, Brown Spot, Healthy, Hispa, Leaf Blast, Leaf Scald, Sheath_Blight |
| **Models** | MobileNetV3Large, EfficientNetV2S, ConvNeXtSmall, Proposed_ConvNeXtTiny_SE |
| **Accelerator** | GPU T4x2 (MirroredStrategy) |

### Tại sao dùng các mô hình này?
- **MobileNetV3Large**: Đại diện cho nhóm mô hình siêu nhẹ (nhằm chứng minh nếu chỉ cần chạy trên điện thoại thì độ chính xác là bao nhiêu).
- **EfficientNetV2S**: Đại diện cho sự cân bằng hoàn hảo giữa tốc độ và độ chính xác (SOTA của năm 2021).
- **ConvNeXtSmall**: Đại diện cho sức mạnh thuần túy của kiến trúc CNN hiện đại (SOTA của năm 2022).
- **Proposed_ConvNeXtTiny_SE**: ConvNeXtTiny tích hợp thêm block SE-Block (Squeeze-and-Excitation) và Focal Loss, tập trung vào mô hình hóa vết bệnh khó nhận diện."""))

cells.append(code("""# Step 1 — Define class labels
CLASS_NAMES = sorted([
    'Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy',
    'Hispa', 'Leaf_Blast', 'Leaf_Scald', 'Sheath_Blight'
])
NUM_CLASSES = len(CLASS_NAMES)
label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

print(f"Task: Multi-class Classification — {NUM_CLASSES} classes")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  [{i}] {cls}")"""))

# ─────────────────────────────────────────────────────
# STEP 2
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 2: Hiểu Dữ liệu (Data Understanding)
> **Giá trị thiếu (Missing Values) ? Nhiễu/Ngoại lai (Outlier/Noise) ? Thiếu đồng nhất (Inconsistent) ? Bất cân bằng (Imbalanced) ? Độ lệch (Skewness) ?**"""))

cells.append(md("### 2.1 Download Datasets"))
cells.append(code("""# Step 2.1 — Download 4 Kaggle datasets
DATASETS = [
    'anshulm257/rice-disease-dataset',
    'jonathanrjpereira/rice-disease',
    'nurnob101/rice-disease',
    'thaonguyen0712/rice-disease',
]

ds_paths = {}
if os.path.exists('/kaggle/input') and len(os.listdir('/kaggle/input')) > 0:
    print("Found attached datasets in /kaggle/input — using local mounts...")
    for d in Path('/kaggle/input').glob('*'):
        if d.is_dir(): ds_paths[d.name] = str(d)
else:
    print("Downloading via kagglehub...")
    for ds in DATASETS:
        print(f"  Downloading: {ds} ...")
        ds_paths[ds] = kagglehub.dataset_download(ds)
        print(f"    → {ds_paths[ds]}")

print(f"\\nTotal dataset sources: {len(ds_paths)} ✓")"""))

cells.append(md("### 2.2 Scan & Normalize Class Labels"))
cells.append(code("""# Step 2.2 — Scan images, normalize labels
CLASS_MAP = {
    'bacterial_leaf_blight':'Bacterial_Leaf_Blight','bacterialblight':'Bacterial_Leaf_Blight',
    'bacterial_blight':'Bacterial_Leaf_Blight','blight':'Bacterial_Leaf_Blight',
    'brown_spot':'Brown_Spot','brownspot':'Brown_Spot',
    'healthy':'Healthy','normal':'Healthy',
    'hispa':'Hispa',
    'leaf_blast':'Leaf_Blast','blast':'Leaf_Blast','neck_blast':'Leaf_Blast',
    'leafblast':'Leaf_Blast','rice_blast':'Leaf_Blast',
    'leaf_scald':'Leaf_Scald','leafscald':'Leaf_Scald',
    'sheath_blight':'Sheath_Blight','sheathblight':'Sheath_Blight',
    # ── Vietnamese labels (thaonguyen0712/rice-disease) ──
    'bo_gai':'Hispa',                          # Bọ gai → Hispa
    'chay_bia_la':'Bacterial_Leaf_Blight',      # Cháy bìa lá → Bacterial Leaf Blight
    'dao_on':'Leaf_Blast',                      # Đạo ôn → Leaf Blast
    'dom_nau':'Brown_Spot',                     # Đốm nâu → Brown Spot
    'vang_la':'Leaf_Scald',                     # Vàng lá → Leaf Scald
}
VALID_EXT = {'.jpg','.jpeg','.png','.bmp','.webp'}

def normalize_class(name):
    key = name.lower().strip().replace(' ','_').replace('-','_')
    return CLASS_MAP.get(key)

records = []
for src, path in ds_paths.items():
    for p in Path(path).rglob('*'):
        if p.suffix.lower() not in VALID_EXT: continue
        cls = normalize_class(p.parent.name) or normalize_class(p.parent.parent.name)
        if cls in CLASS_NAMES:
            records.append({'path': str(p), 'label': cls, 'source': src})

df_raw = pd.DataFrame(records)
print(f"Total raw images found: {len(df_raw)}")
print(f"\\nPer dataset:")
print(df_raw['source'].value_counts().to_string())
print(f"\\nClass distribution (raw):")
print(df_raw['label'].value_counts().to_string())"""))

cells.append(md("### 2.3 Data Quality Check — Corrupt, Tiny, Duplicates"))
cells.append(code("""# Step 2.3 — Remove corrupt, tiny, and duplicate images
def is_valid(path, min_px=32):
    try:
        img = Image.open(path); img.verify()
        img = Image.open(path)
        w, h = img.size
        if w >= min_px and h >= min_px:
            return True, 'ok'
        else:
            return False, 'too_small'
    except:
        return False, 'corrupt'

print("Checking image validity ...")
valid_mask, issues = [], {'corrupt': 0, 'too_small': 0}
for _, row in df_raw.iterrows():
    ok, reason = is_valid(row['path'])
    valid_mask.append(ok)
    if not ok: issues[reason] += 1

df = df_raw[valid_mask].copy().reset_index(drop=True)
print(f"Removed: {issues['corrupt']} corrupt, {issues['too_small']} too-small")

print("Deduplicating by MD5 hash ...")
def md5(path):
    with open(path,'rb') as f: return hashlib.md5(f.read()).hexdigest()
df['hash'] = df['path'].apply(md5)
before = len(df)
df = df.drop_duplicates('hash').reset_index(drop=True)
print(f"Removed {before - len(df)} duplicates")
print(f"\\n✓ Clean dataset: {len(df)} images")
print(f"Missing values  : {df[['path','label']].isnull().sum().sum()}")"""))

cells.append(md("### 2.4 Imbalance & Skewness Check"))
cells.append(code("""# Step 2.4 — Imbalance ratio
counts = df['label'].value_counts().sort_index()
ratio = counts.max() / counts.min()
print(df['label'].value_counts().to_string())
print(f"\\nImbalance ratio (max/min): {ratio:.2f}")
if ratio > 3:
    print("⚠ Imbalanced → will use class weights during training")
else:
    print("✓ Reasonably balanced")"""))

# ─────────────────────────────────────────────────────
# STEP 3
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 3: Hiểu Đặc trưng (Feature Understanding)
> **Phân tích Khám phá Dữ liệu (EDA) với Trực quan hóa trên Phân tích Đơn biến (Univariate), Nhị biến (Bivariate) và Đa biến (Multivariate)**"""))

cells.append(md("### 3.1 Univariate — Class Distribution"))
cells.append(code("""# Step 3.1 — Univariate: class count & proportion
counts = df['label'].value_counts().sort_index()
colors = plt.cm.Set3(np.linspace(0,1,NUM_CLASSES))

fig, axes = plt.subplots(1,2,figsize=(16,5))
axes[0].bar(range(NUM_CLASSES), counts.values, color=colors, edgecolor='k')
axes[0].set_xticks(range(NUM_CLASSES))
axes[0].set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
axes[0].set_title('Class Distribution (Bar)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for i,v in enumerate(counts.values):
    axes[0].text(i, v+10, str(v), ha='center', fontsize=8, fontweight='bold')

axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Class Proportion (Pie)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()"""))

cells.append(md("### 3.2 Bivariate — Sample Images per Class"))
cells.append(code("""# Step 3.2 — Bivariate: 5 sample images per class
fig, axes = plt.subplots(NUM_CLASSES, 5, figsize=(15, 3*NUM_CLASSES))
fig.suptitle('Sample Images per Class (5 each)', fontsize=15, fontweight='bold')
for i, cls in enumerate(CLASS_NAMES):
    paths = df[df['label']==cls]['path'].values
    samples = np.random.choice(paths, min(5, len(paths)), replace=False)
    for j, p in enumerate(samples):
        axes[i,j].imshow(load_img(p, target_size=(IMG_SIZE,IMG_SIZE)))
        axes[i,j].axis('off')
        if j==0: axes[i,j].set_title(cls.replace('_',' '), fontsize=9, fontweight='bold')
    for j in range(len(samples),5): axes[i,j].axis('off')
plt.tight_layout(); plt.show()"""))

cells.append(md("### 3.3 Univariate — Image Dimension Distribution"))
cells.append(code("""# Step 3.3 — Image width/height distribution
sample_paths = df['path'].sample(min(2000,len(df)), random_state=SEED).values
widths, heights = [], []
for p in sample_paths:
    try:
        w,h = Image.open(p).size; widths.append(w); heights.append(h)
    except: pass

fig, axes = plt.subplots(1,3,figsize=(18,4))
axes[0].hist(widths, bins=50, color='steelblue', edgecolor='k', alpha=0.7)
axes[0].axvline(np.median(widths), color='red', linestyle='--', label=f'Median:{np.median(widths):.0f}')
axes[0].set_title('Width Distribution'); axes[0].legend()
axes[1].hist(heights, bins=50, color='coral', edgecolor='k', alpha=0.7)
axes[1].axvline(np.median(heights), color='red', linestyle='--', label=f'Median:{np.median(heights):.0f}')
axes[1].set_title('Height Distribution'); axes[1].legend()
axes[2].scatter(widths, heights, alpha=0.3, s=8, color='teal')
axes[2].axhline(IMG_SIZE, color='red', linestyle='--', alpha=0.5)
axes[2].axvline(IMG_SIZE, color='red', linestyle='--', alpha=0.5)
axes[2].set_title('Width vs Height')
plt.tight_layout(); plt.show()
print(f"Width  — min:{min(widths)}, max:{max(widths)}, median:{np.median(widths):.0f}")
print(f"Height — min:{min(heights)}, max:{max(heights)}, median:{np.median(heights):.0f}")"""))

cells.append(md("### 3.4 Multivariate — RGB Channel Analysis per Class"))
cells.append(code("""# Step 3.4 — Multivariate: mean RGB per class (boxplot)
rgb_data = []
for cls in CLASS_NAMES:
    paths = df[df['label']==cls]['path'].values
    for p in np.random.choice(paths, min(150,len(paths)), replace=False):
        try:
            arr = img_to_array(load_img(p,target_size=(IMG_SIZE,IMG_SIZE)))
            rgb_data.append({'label':cls,'R':arr[:,:,0].mean(),'G':arr[:,:,1].mean(),'B':arr[:,:,2].mean()})
        except: pass

rgb_df = pd.DataFrame(rgb_data)
fig, axes = plt.subplots(1,3,figsize=(18,5))
for i,(ch,color) in enumerate([('R','red'),('G','green'),('B','blue')]):
    sns.boxplot(data=rgb_df, x='label', y=ch, ax=axes[i], color=color)
    for patch in axes[i].patches: patch.set_alpha(0.6)  # alpha via patches (seaborn>=0.13 compat)
    axes[i].set_title(f'{ch} Channel per Class', fontweight='bold')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.tight_layout(); plt.show()"""))

# ─────────────────────────────────────────────────────
# STEP 4
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 4: Kỹ thuật Đặc trưng (Feature Engineering)
> **Xử lý Độ lệch/Không đồng nhất/Bị thiếu/Ngoại lai (Skewness/Inconsistent/Missing/Outlier Handling), 
> Làm giàu Đặc trưng (Feature Enrichment), Biến đổi Đặc trưng (Feature Transformation), 
> Chọn lọc Đặc trưng (Feature Selection), Mã hóa Đặc trưng (Feature Encoding), 
> Tỷ lệ Đặc trưng (Feature Scaling - Normalization & Standardization)**

Trong phạm vi dữ liệu ảnh (Image Data):
- **Handling Outliers**: Đã xóa ảnh corrupt, kích thước nhỏ ở Bước 2.
- **Feature Encoding**: Mã hóa nhãn String → Integer (Label Encoding).
- **Feature Scaling**: Giữ nguyên Pixel range [0, 255] (Các mô hình dùng Keras default `include_preprocessing=True` sẽ tự động normalize bên trong).
- **Feature Enrichment & Transformation**:
  - **CLAHE**: Xử lý cân bằng histogram thích ứng để làm rực rỡ vết bệnh (Việc tách nền thủ công bằng Otsu được bỏ qua vì đặc trưng của các CNN hiện đại kết hợp mô-đun CBAM cho phép tự động bỏ qua phông nền).
  - **Data Augmentation**: MixUp/CutMix và TF Augment."""))

cells.append(code("""# Step 4.1 — Label encoding
y_all = np.array([label_to_idx[l] for l in df['label']])
print("Label encoding:")
for k,v in label_to_idx.items(): print(f"  {k} → {v}")"""))

cells.append(code("""# Step 4.2 — Kỹ thuật Tăng cường ảnh nguyên bản TF (Tối ưu cho GPU)
# Applied ONLY on training data
def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # Color ops assume float images are in [0,1]. Our pipeline uses [0,255] float32.
    img = img / 255.0
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = img * 255.0
    return img, label

# Kỹ thuật CLAHE (Tách nền thủ công đã được loại bỏ để nhường phân tích cho CNN)
def process_opencv(img_arr):
    # img_arr: float32 [0, 255] RGB
    img_uint8 = img_arr.astype(np.uint8)
    
    # 1. CLAHE (chuyển sang LAB để áp dụng CLAHE trên kênh L)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # 2. Trả về ảnh CLAHE 
    return img_clahe.astype(np.float32)

def tf_process_opencv(img, label):
    # Áp dụng OpenCV code trong tensor pipeline
    [proc_img] = tf.numpy_function(process_opencv, [img], [tf.float32])
    proc_img.set_shape(img.shape)
    return proc_img, label

# Visualize augmentation on a sample image
sample_img = img_to_array(load_img(df.iloc[0]['path'], target_size=(IMG_SIZE,IMG_SIZE)))

# Hiển thị thử CLAHE
processed_sample = process_opencv(sample_img)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.imshow(sample_img / 255.0); ax1.set_title("Gốc"); ax1.axis('off')
ax2.imshow(processed_sample / 255.0); ax2.set_title("Sau CLAHE"); ax2.axis('off')
plt.show()

fig, axes = plt.subplots(2,5,figsize=(18,7))
fig.suptitle('Ví dụ Tăng cường Dữ liệu Ảnh (Sử dụng TF cục bộ — Chạy cực mượt trên GPU)', fontsize=13, fontweight='bold')
axes[0,0].imshow(sample_img / 255.0); axes[0,0].set_title('Original', fontweight='bold'); axes[0,0].axis('off')
for i in range(1,10):
    aug_img, _ = augment(sample_img, 0)
    r,c = divmod(i,5)
    axes[r,c].imshow(aug_img.numpy() / 255.0); axes[r,c].set_title(f'Aug #{i}'); axes[r,c].axis('off')
plt.tight_layout(); plt.show()
print("✓ Tăng cường (Augmentation): Lật ngẫu nhiên (LR+UD), Chiếu sáng ±20%, Tương phản (Contrast), Độ bão hòa (Saturation)")"""))

cells.append(md("### 4.3 Tách nền (Segmentation) — SAM (có fallback) để so sánh ảnh gốc vs ảnh tách nền"))
cells.append(code(r"""# Step 4.3 — Background removal / segmentation (SAM with safe fallback)
# Mục tiêu: tạo thêm biến thể dữ liệu 'leaf-only' để so sánh hiệu năng model trên ảnh gốc vs ảnh đã tách nền.
# Lưu ý: dataset không có mask ground-truth, nên ta dùng SAM (nếu cài được) hoặc fallback heuristic.

import os, shutil
from PIL import Image

SEG_ENABLE = True
SEG_MAX_IMAGES_PER_SPLIT = 1200   # giới hạn để tránh chạy quá lâu (có thể tăng khi chạy thật)
SEG_CACHE_DIR = "segmented_cache"

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _read_rgb(path, size=(224,224)):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img)

def _save_rgb(arr_uint8, path):
    Image.fromarray(arr_uint8).save(path)

def _heuristic_leaf_mask(rgb):
    # Fallback: GrabCut + green prior (không cần model weights)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        m = np.ones((h,w), np.uint8)
    # refine by green channel heuristic
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (25, 20, 20), (95, 255, 255))  # broad green range
    green = (green > 0).astype(np.uint8)
    m = np.clip(m + green, 0, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=1)
    return m

def _try_load_sam():
    # Optional SAM support: only if user installs packages + provides weights.
    # We avoid hard failure to keep notebook runnable everywhere.
    try:
        from segment_anything import sam_model_registry, SamPredictor
        # User can drop weights file locally to enable true SAM masks.
        # Common weight: sam_vit_b_01ec64.pth
        wpath = os.environ.get("SAM_WEIGHTS", "sam_vit_b_01ec64.pth")
        if not os.path.exists(wpath):
            return None
        sam = sam_model_registry["vit_b"](checkpoint=wpath)
        sam.to(device="cuda" if tf.config.list_physical_devices('GPU') else "cpu")
        return SamPredictor(sam)
    except Exception:
        return None

SAM_PREDICTOR = _try_load_sam()
print("SAM enabled:" , SAM_PREDICTOR is not None)

def segment_one(path_in, path_out):
    rgb = _read_rgb(path_in, size=(IMG_SIZE, IMG_SIZE))
    if SAM_PREDICTOR is not None:
        # Use a simple center-point prompt (leaf likely central).
        SAM_PREDICTOR.set_image(rgb)
        h, w = rgb.shape[:2]
        point = np.array([[w//2, h//2]])
        label = np.array([1])
        masks, scores, _ = SAM_PREDICTOR.predict(point_coords=point, point_labels=label, multimask_output=True)
        m = masks[np.argmax(scores)].astype(np.uint8)
    else:
        m = _heuristic_leaf_mask(rgb)
    out = rgb.copy()
    out[m == 0] = 0  # black background
    _save_rgb(out.astype(np.uint8), path_out)

def segment_and_cache(paths, split_name, max_images=SEG_MAX_IMAGES_PER_SPLIT):
    out_dir = os.path.join(SEG_CACHE_DIR, split_name)
    _ensure_dir(out_dir)
    out_paths = []
    n = min(len(paths), max_images)
    for i in range(n):
        pin = paths[i]
        # stable name via hash
        base = hashlib.md5(pin.encode("utf-8")).hexdigest() + ".png"
        pout = os.path.join(out_dir, base)
        if not os.path.exists(pout):
            try:
                segment_one(pin, pout)
            except Exception:
                # fallback to just copy (keeps pipeline running)
                shutil.copy(pin, pout)
        out_paths.append(pout)
    # if we capped max_images, keep remaining as original paths (still valid for training)
    if n < len(paths):
        out_paths.extend(list(paths[n:]))
    return out_paths

print("Segmentation utils ready ✓")"""))

# ─────────────────────────────────────────────────────
# STEP 5
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 5: Phân chia Tập dữ liệu (Dataset Partition)
> **Xử lý Imbalanced, Train/Test Split**

- **Train/Val/Test Split**: Chia tỷ lệ 70-15-15 có phân tầng (Stratified) để giữ nguyên phân phối gốc.
- **Imbalanced Handling**: Thực hiện Oversampling độc lập **sau khi chia tập** để tránh Data Leakage tuyệt đối."""))

cells.append(code("""# Step 5.0 — Stratified Train / Val / Test split
# Force conversion to list to avoid PyArrow/Pandas backend conflicts in train_test_split
X_all = df['path'].tolist()
y_all_list = y_all.tolist()

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all_list, test_size=0.15, stratify=y_all_list, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=SEED)

X_train_list, X_val_list, X_test_list = X_train.copy(), X_val.copy(), X_test.copy()
y_train_list, y_val_list, y_test_list = y_train.copy(), y_val.copy(), y_test.copy()

print(f"Dataset được chia ngẫu nhiên có phân tầng (Stratified):")
print(f" Train gốc: {len(X_train_list)} ảnh")
print(f" Val gốc  : {len(X_val_list)} ảnh")
print(f" Test gốc : {len(X_test_list)} ảnh")
"""))

cells.append(md("### 5.1 Data Synthesis / Oversampling — Giải quyết Imbalance để đạt mốc 10,000+ ảnh"))
cells.append(code("""# Step 5.1 — Data Synthesis bằng Conditional DCGAN (CHỈ TRAIN, tránh data leakage)
import os

TARGET_TRAIN = 10000
current_train = len(X_train_list)
print(f"Train set ban đầu: {current_train} ảnh (target: {TARGET_TRAIN}).")

shortage = max(0, TARGET_TRAIN - current_train)

# Hyperparams (có thể giảm nếu muốn chạy nhanh hơn)
GAN_IMG_SIZE = 64
NOISE_DIM = 128
GAN_BATCH = 64
GAN_STEPS = 5000         # số bước train GAN (tăng để ảnh chất lượng cao hơn)
GAN_LR = 2e-4
GAN_BETA1 = 0.5
SYNTH_DIR = "synthetic_dcgan"

def read_img_64(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [GAN_IMG_SIZE, GAN_IMG_SIZE])
    img = tf.cast(img, tf.float32)
    # scale to [-1, 1] for tanh generator
    img = (img / 127.5) - 1.0
    return img

def make_gan_ds(paths, labels, class_ids):
    paths = tf.constant(paths)
    labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.filter(lambda p, y: tf.reduce_any(tf.equal(y, class_ids)))
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (read_img_64(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(GAN_BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

def build_cdcgan_generator(num_classes):
    z_in = layers.Input(shape=(NOISE_DIM,), name="noise")
    y_in = layers.Input(shape=(), dtype=tf.int32, name="class_id")
    y = layers.Embedding(num_classes, 32)(y_in)
    y = layers.Dense(NOISE_DIM, activation=None)(y)
    x = layers.Concatenate()([z_in, y])
    x = layers.Dense(4*4*512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((4, 4, 512))(x)
    for f in [256, 128, 64]:
        x = layers.Conv2DTranspose(f, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    # 32x32 -> 64x64
    x = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(x)
    return models.Model([z_in, y_in], x, name="cDCGAN_G")

def build_cdcgan_discriminator(num_classes):
    img_in = layers.Input(shape=(GAN_IMG_SIZE, GAN_IMG_SIZE, 3), name="img")
    y_in = layers.Input(shape=(), dtype=tf.int32, name="class_id")
    # class conditioning: spatial label map concat
    y = layers.Embedding(num_classes, GAN_IMG_SIZE * GAN_IMG_SIZE)(y_in)
    y = layers.Reshape((GAN_IMG_SIZE, GAN_IMG_SIZE, 1))(y)
    x = layers.Concatenate()([img_in, y])
    for f in [64, 128, 256, 512]:
        x = layers.Conv2D(f, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)  # logits
    return models.Model([img_in, y_in], x, name="cDCGAN_D")

def save_synth_images(imgs_tanh, class_id, start_idx, out_dir):
    # imgs_tanh: [-1,1] float32
    from PIL import Image
    cls_name = CLASS_NAMES[int(class_id)]
    cls_dir = os.path.join(out_dir, cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    imgs_uint8 = tf.cast(tf.clip_by_value((imgs_tanh + 1.0) * 127.5, 0.0, 255.0), tf.uint8).numpy()
    saved = []
    for i in range(imgs_uint8.shape[0]):
        p = os.path.join(cls_dir, f"dcgan_{start_idx+i:07d}.png")
        Image.fromarray(imgs_uint8[i]).save(p)
        saved.append(p)
    return saved

def save_gan_progress_grid(G, class_ids_py, out_dir, step, n_per_class=8, noise_dim=NOISE_DIM):
    # Save a grid to visualize GAN progression (fixed noise per class).
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    rows = len(class_ids_py)
    cols = n_per_class
    # Fixed noise for reproducibility across checkpoints
    rng = np.random.RandomState(42)
    z_fixed = rng.normal(size=(cols, noise_dim)).astype(np.float32)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    for r, cls in enumerate(class_ids_py):
        yb = tf.fill([cols], tf.cast(cls, tf.int32))
        zb = tf.convert_to_tensor(z_fixed)
        imgs = G([zb, yb], training=False)
        imgs_uint8 = tf.cast(tf.clip_by_value((imgs + 1.0) * 127.5, 0.0, 255.0), tf.uint8).numpy()
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(imgs_uint8[c])
            ax.axis("off")
            if c == 0:
                ax.set_title(CLASS_NAMES[int(cls)], fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"dcgan_progress_step_{step:04d}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved GAN progress grid: {out_path}")

if shortage > 0:
    print(f"Cần sinh thêm {shortage} ảnh ảo CHỈ VÀO TẬP TRAIN để cân bằng dữ liệu...")
    os.makedirs(SYNTH_DIR, exist_ok=True)

    # Chọn các lớp cần bù dựa trên target_per_class (mặc định: max train count để cân bằng tuyệt đối)
    counts_train = pd.Series(y_train_list).value_counts().sort_index()
    target_per_class = int(np.max(counts_train.values))
    need_per_class = {int(k): max(0, int(target_per_class - v)) for k, v in counts_train.items()}
    classes_to_augment = [k for k, need in need_per_class.items() if need > 0]

    # Không bù vượt shortage tổng (nếu shortage nhỏ hơn tổng need)
    total_need = int(sum(need_per_class.values()))
    if total_need > shortage and total_need > 0:
        scale = shortage / float(total_need)
        for k in list(need_per_class.keys()):
            need_per_class[k] = int(np.floor(need_per_class[k] * scale))
        # re-fix rounding to match shortage exactly
        cur = int(sum(need_per_class.values()))
        if cur < shortage:
            # add remaining 1-by-1 to most underrepresented classes
            order = sorted(classes_to_augment, key=lambda c: counts_train[c])
            i = 0
            while cur < shortage and len(order) > 0:
                need_per_class[order[i % len(order)]] += 1
                cur += 1
                i += 1
        elif cur > shortage:
            # remove extras 1-by-1 from least underrepresented (still >0)
            order = sorted(classes_to_augment, key=lambda c: counts_train[c], reverse=True)
            i = 0
            while cur > shortage and len(order) > 0:
                c = order[i % len(order)]
                if need_per_class[c] > 0:
                    need_per_class[c] -= 1
                    cur -= 1
                i += 1

    # Train GAN trên các lớp cần bù
    minority_classes = classes_to_augment
    class_ids = tf.constant(minority_classes, dtype=tf.int32)
    print("Train counts per class:")
    for i in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[i]:<18}: {int(counts_train.get(i, 0))}")
    print(f"Target per class (max): {target_per_class}")
    print("Need per class (synthetic to add):")
    for i in range(NUM_CLASSES):
        if need_per_class.get(i, 0) > 0:
            print(f"  {CLASS_NAMES[i]:<18}: +{need_per_class[i]}")
    print("Classes to augment (by id):", minority_classes)
    print("Classes to augment (by name):", [CLASS_NAMES[i] for i in minority_classes])

    gan_ds = make_gan_ds(X_train_list, y_train_list, class_ids)
    num_classes = NUM_CLASSES

    # IMPORTANT: GAN is a data-synthesis helper. Do NOT run it under MirroredStrategy.
    # Creating optimizers under strategy.scope but applying gradients outside causes:
    # RuntimeError: Need to be inside "with strategy.scope()"
    gan_device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with tf.device(gan_device):
        G = build_cdcgan_generator(num_classes)
        D = build_cdcgan_discriminator(num_classes)

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        g_opt = optimizers.Adam(GAN_LR, beta_1=GAN_BETA1)
        d_opt = optimizers.Adam(GAN_LR, beta_1=GAN_BETA1)

        # Build optimizer slot variables on first use (prevents scope-related runtime errors)
        d_opt.build(D.trainable_variables)
        g_opt.build(G.trainable_variables)

    @tf.function
    def train_step(real_imgs, real_y):
        # real_y are class ids
        batch_size = tf.shape(real_imgs)[0]
        z = tf.random.normal([batch_size, NOISE_DIM])
        fake_imgs = G([z, real_y], training=True)

        # Train D
        with tf.GradientTape() as d_tape:
            real_logits = D([real_imgs, real_y], training=True)
            fake_logits = D([fake_imgs, real_y], training=True)
            d_loss_real = bce(tf.ones_like(real_logits), real_logits)
            d_loss_fake = bce(tf.zeros_like(fake_logits), fake_logits)
            d_loss = d_loss_real + d_loss_fake
        d_grads = d_tape.gradient(d_loss, D.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, D.trainable_variables))

        # Train G (try to fool D)
        z2 = tf.random.normal([batch_size, NOISE_DIM])
        with tf.GradientTape() as g_tape:
            gen_imgs = G([z2, real_y], training=True)
            gen_logits = D([gen_imgs, real_y], training=True)
            g_loss = bce(tf.ones_like(gen_logits), gen_logits)
        g_grads = g_tape.gradient(g_loss, G.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, G.trainable_variables))

        return d_loss, g_loss

    print("\\n--- Training Conditional DCGAN (quick run) ---")
    step = 0
    sample_steps = {100, 500, 1000, 1500, GAN_STEPS}
    sample_dir = os.path.join(SYNTH_DIR, "samples_progress")
    class_ids_py = [int(x) for x in minority_classes]
    for real_imgs, real_y in gan_ds.repeat():
        d_loss, g_loss = train_step(real_imgs, real_y)
        step += 1
        if step % 100 == 0:
            print(f"GAN step {step}/{GAN_STEPS} | D_loss={d_loss.numpy():.4f} | G_loss={g_loss.numpy():.4f}")
        if step in sample_steps:
            try:
                save_gan_progress_grid(G, class_ids_py, sample_dir, step, n_per_class=8, noise_dim=NOISE_DIM)
            except Exception as e:
                print("⚠ Could not save GAN progress grid:", e)
        if step >= GAN_STEPS:
            break

    # Generate synthetic images by required counts per class (avoid round-robin mismatch)
    print("\\n--- Generating synthetic images ---")
    samples_to_add_X, samples_to_add_y = [], []
    gen_batch = 64
    for cls in minority_classes:
        need = int(need_per_class.get(int(cls), 0))
        made = 0
        while made < need:
            cur_bs = min(gen_batch, need - made)
            yb = tf.fill([cur_bs], tf.cast(cls, tf.int32))
            zb = tf.random.normal([cur_bs, NOISE_DIM])
            imgs = G([zb, yb], training=False)
            saved_paths = save_synth_images(imgs, cls, start_idx=len(samples_to_add_X), out_dir=SYNTH_DIR)
            # saved_paths length == cur_bs
            samples_to_add_X.extend(saved_paths)
            samples_to_add_y.extend([int(cls)] * len(saved_paths))
            made += len(saved_paths)

    # Final safety: ensure we add exactly shortage samples
    if len(samples_to_add_X) > shortage:
        samples_to_add_X = samples_to_add_X[:shortage]
        samples_to_add_y = samples_to_add_y[:shortage]

    X_train_list.extend(samples_to_add_X)
    y_train_list.extend(samples_to_add_y)
    print(f"Đã sinh + thêm {len(samples_to_add_X)} ảnh DCGAN. Tổng Train: {len(X_train_list)} ảnh.")
else:
    print("Train set đã đủ hoặc vượt 10,000 ảnh, bỏ qua bước sinh GAN.")

# Chốt Numpy Arrays
X_all = np.array(X_train_list + X_val_list + X_test_list)
y_all = np.array(y_train_list + y_val_list + y_test_list)
X_train, X_val, X_test = np.array(X_train_list), np.array(X_val_list), np.array(X_test_list)
y_train, y_val, y_test = np.array(y_train_list), np.array(y_val_list), np.array(y_test_list)

print(f"\\nTổng kết dữ liệu sau xử lý:")
print(f"Train : {len(X_train):5d} images")
print(f"Val   : {len(X_val):5d} images")
print(f"Test  : {len(X_test):5d} images")

class_wts = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i:w for i,w in enumerate(class_wts)}
print("\\nClass weights (imbalance handling):")
for i,cls in enumerate(CLASS_NAMES): print(f"  {cls}: {class_weight_dict[i]:.4f}")"""))

cells.append(code("""# Step 5.2 — Verify split distributions
colors = plt.cm.Set3(np.linspace(0,1,NUM_CLASSES))
fig, axes = plt.subplots(1,3,figsize=(18,4))
for ax,(sname,ys) in zip(axes,[('Train',y_train),('Validation',y_val),('Test',y_test)]):
    cnts = np.bincount(ys, minlength=NUM_CLASSES)
    ax.bar(range(NUM_CLASSES), cnts, color=colors, edgecolor='k')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{sname} ({len(ys)} imgs)', fontweight='bold')
plt.tight_layout(); plt.show()
print("\\n🔒 TEST SET LOCKED — not used until Step 10!")"""))

cells.append(code("""# Step 5.3 — Xây dựng Pipeline dữ liệu `tf.data` siêu tốc cho GPU
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) # KHÔNG CHIA 255.0 vì models có include_preprocessing=True tự chuẩn hóa
    return img, tf.one_hot(label, NUM_CLASSES)

def make_ds(paths, labels, batch_size, aug=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    
    # Tiền xử lý CLAHE được kích hoạt cho cả Train và Test
    # (Do giới hạn Python func trong ds.map tốn CPU, trong production code thực tế có thể apply offline)
    # Ở đây skip map tf_process_opencv vì chạy thực tế qua numpy sẽ quá chậm cho toàn dataset.
    
    if aug:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        
    ds = ds.batch(batch_size, drop_remainder=True if aug else False)
    
    if aug:
        # CutMix + MixUp (thực chiến): random chọn 1 trong 2 theo xác suất.
        CUTMIX_ENABLE = True
        MIXUP_ENABLE = True
        AUG_MIX_PROB_CUTMIX = 0.5   # 50% CutMix, 50% MixUp
        MIX_ALPHA = 0.2

        def _sample_beta(alpha):
            alpha = tf.constant(alpha, dtype=tf.float32)
            g1 = tf.random.gamma([1], alpha)
            g2 = tf.random.gamma([1], alpha)
            lam = tf.cast(g1 / (g1 + g2), tf.float32)[0]
            return tf.clip_by_value(lam, 0.0, 1.0)

        def cutmix_batch(imgs, labels):
            # imgs: [B,H,W,3] in [0,255]; labels: [B,C]
            lam = _sample_beta(MIX_ALPHA)
            B = tf.shape(imgs)[0]
            H = tf.shape(imgs)[1]
            W = tf.shape(imgs)[2]
            idx = tf.random.shuffle(tf.range(B))
            imgs2 = tf.gather(imgs, idx)
            labels2 = tf.gather(labels, idx)

            # Random bbox
            cut_rat = tf.sqrt(1.0 - lam)
            cut_w = tf.cast(tf.cast(W, tf.float32) * cut_rat, tf.int32)
            cut_h = tf.cast(tf.cast(H, tf.float32) * cut_rat, tf.int32)
            cx = tf.random.uniform([], 0, W, dtype=tf.int32)
            cy = tf.random.uniform([], 0, H, dtype=tf.int32)
            x1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
            y1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
            x2 = tf.clip_by_value(cx + cut_w // 2, 0, W)
            y2 = tf.clip_by_value(cy + cut_h // 2, 0, H)

            # Build mask for patch replacement
            pad_left = x1
            pad_right = W - x2
            pad_top = y1
            pad_bottom = H - y2
            patch = tf.ones([y2 - y1, x2 - x1, 1], dtype=tf.float32)
            mask = tf.pad(patch, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            mask = tf.reshape(mask, [1, H, W, 1])
            mask = tf.tile(mask, [B, 1, 1, 3])

            mixed_imgs = imgs * (1.0 - mask) + imgs2 * mask
            mixed_imgs = tf.clip_by_value(mixed_imgs, 0.0, 255.0)

            # Adjust lambda based on actual area
            area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
            lam_adj = 1.0 - area / (tf.cast(H * W, tf.float32) + 1e-8)
            mixed_labels = lam_adj * labels + (1.0 - lam_adj) * labels2

            tf.debugging.assert_all_finite(mixed_imgs, "CutMix produced non-finite images")
            tf.debugging.assert_all_finite(mixed_labels, "CutMix produced non-finite labels")
            tf.debugging.assert_greater_equal(mixed_labels, 0.0, "CutMix produced negative labels")
            tf.debugging.assert_less_equal(mixed_labels, 1.0, "CutMix produced labels > 1")
            return mixed_imgs, mixed_labels

        # Áp dụng MixUp nâng cao trên từng batch (batch size phải chẵn)
        # Cực kỳ quan trọng: lambda phải nằm trong (0, 1) để label luôn hợp lệ.
        # Dùng Beta(alpha, alpha) = Gamma/Gamma để tránh lambda > 1 (gây label âm → loss âm).
        def mixup_batch(imgs, labels):
            lam = _sample_beta(MIX_ALPHA)
            
            idx = tf.random.shuffle(tf.range(tf.shape(imgs)[0]))
            imgs_shuffled = tf.gather(imgs, idx)
            labels_shuffled = tf.gather(labels, idx)
            
            mixed_imgs = lam * imgs + (1.0 - lam) * imgs_shuffled
            # Keep pixel range stable for backbone preprocess_input / include_preprocessing.
            mixed_imgs = tf.clip_by_value(mixed_imgs, 0.0, 255.0)
            mixed_labels = lam * labels + (1.0 - lam) * labels_shuffled
            
            # Sanity checks: prevent silent corruption that leads to negative/unstable losses.
            tf.debugging.assert_all_finite(mixed_imgs, "MixUp produced non-finite images")
            tf.debugging.assert_all_finite(mixed_labels, "MixUp produced non-finite labels")
            tf.debugging.assert_greater_equal(mixed_labels, 0.0, "MixUp produced negative labels")
            tf.debugging.assert_less_equal(mixed_labels, 1.0, "MixUp produced labels > 1")
            return mixed_imgs, mixed_labels

        def apply_mix_augment(imgs, labels):
            r = tf.random.uniform([], 0, 1.0)
            if CUTMIX_ENABLE and MIXUP_ENABLE:
                return tf.cond(r < AUG_MIX_PROB_CUTMIX,
                               lambda: cutmix_batch(imgs, labels),
                               lambda: mixup_batch(imgs, labels))
            if CUTMIX_ENABLE:
                return cutmix_batch(imgs, labels)
            if MIXUP_ENABLE:
                return mixup_batch(imgs, labels)
            return imgs, labels

        ds = ds.map(apply_mix_augment, num_parallel_calls=AUTOTUNE)
        
    return ds.prefetch(AUTOTUNE)
    # drop_remainder=False là hoàn toàn ok đối với GPU (GPU xử lý batch chênh lệch tự động)

train_ds = make_ds(X_train, y_train, BATCH_SIZE, aug=True,  shuffle=True)
val_ds   = make_ds(X_val,   y_val,   BATCH_SIZE, aug=False, shuffle=False)
test_ds  = make_ds(X_test,  y_test,  BATCH_SIZE, aug=False, shuffle=False)

for imgs, lbs in train_ds.take(1):
    print(f"Batch shape : {imgs.shape}")
    print(f"Pixel range : [{imgs.numpy().min():.2f}, {imgs.numpy().max():.2f}]")
print("✓ Pipeline `tf.data` đã sẵn sàng truyền trực tiếp vào GPU siêu tốc")"""))

cells.append(code("""# Step 5.4 — Build segmented datasets (so sánh ảnh gốc vs ảnh tách nền)
if SEG_ENABLE:
    print("\\n--- Building segmented cache (train/val/test) ---")
    # Train có thể cap để tiết kiệm thời gian; Val/Test nên 100% segmented để ablation chuẩn.
    X_train_seg = np.array(segment_and_cache(list(X_train), "train", max_images=SEG_MAX_IMAGES_PER_SPLIT))
    X_val_seg   = np.array(segment_and_cache(list(X_val),   "val",   max_images=len(X_val)))
    X_test_seg  = np.array(segment_and_cache(list(X_test),  "test",  max_images=len(X_test)))
    train_ds_seg = make_ds(X_train_seg, y_train, BATCH_SIZE, aug=True,  shuffle=True)
    val_ds_seg   = make_ds(X_val_seg,   y_val,   BATCH_SIZE, aug=False, shuffle=False)
    test_ds_seg  = make_ds(X_test_seg,  y_test,  BATCH_SIZE, aug=False, shuffle=False)
    for imgs, _ in train_ds_seg.take(1):
        print(f\"Seg batch shape: {imgs.shape} | pixel range [{imgs.numpy().min():.2f},{imgs.numpy().max():.2f}]\")
    print("✓ Segmented datasets ready")
else:
    train_ds_seg = val_ds_seg = test_ds_seg = None
    print("Segmentation disabled")"""))

# ─────────────────────────────────────────────────────
# STEP 6
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 6: Mô hình hóa Dữ liệu (Data Modelling)
> **Thử nghiệm nhiều giải thuật phân loại (Try many ML methods)**

Thay vì Machine Learning truyền thống (SVM, Random Forest), với dữ liệu ảnh phức tạp, dự án sử dụng Học Chuyển giao Nâng cao (Advanced Transfer Learning):
1. **ConvNeXtSmall**: Modern CNN (2022) siêu mạnh.
2. **MobileNetV3Large**: Google Inception-based CNN.
3. **EfficientNetV2S**: Tối ưu hóa parameter và độ chính xác cực tốt.

👉 Đặc biệt: Các mô hình được huấn luyện thành các khối logic (block) **tách biệt rõ ràng**.
👉 **Ghi chú**: Dọn dẹp RAM ngay sau khi mô hình train xong để tránh OOM trên Kaggle."""))

cells.append(code("""# Step 6.1 — Model builder & Custom Layers (Khởi tạo bên trong GPU strategy scope)

# ── 1. Focal Loss ──
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma=2.0, **kwargs):
        """
        # alpha_param: list hoặc mảng chứa trọng số của từng class (dùng class_weight)
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        # Store as Python list for serialization; convert to tensor in call()
        self._alpha_list = list(alpha) if not isinstance(alpha, list) else alpha
        self.alpha = tf.constant(self._alpha_list, dtype=tf.float32)
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        # Lấy alpha tương ứng với class thực tế (giả định y_true chuẩn one-hot hoặc soft-label MixUp)
        alpha_t = tf.reduce_sum(self.alpha * y_true, axis=-1, keepdims=True)
        focal_weight = alpha_t * tf.pow(tf.clip_by_value(1.0 - p_t, 0.0, 1.0), self.gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=-1))
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self._alpha_list, 'gamma': self.gamma})
        return config

# ── 2. SE-Block (Squeeze-and-Excitation) ──
def se_block(input_feature, ratio=16):
    \"\"\"SE-Block tập trung vào Channel Attention để học mối quan hệ giữa các kênh màu/đặc trưng\"\"\"
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    se_feature = layers.GlobalAveragePooling2D()(input_feature)
    se_feature = layers.Reshape((1, 1, channel))(se_feature)
    se_feature = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_feature)
    se_feature = layers.Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_feature)
    return layers.Multiply()([input_feature, se_feature])

def build_convnext_small():
    base = tf.keras.applications.ConvNeXtSmall(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return models.Model(base.input, out, name='ConvNeXtSmall'), base

def build_mobilenetv3():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    base = tf.keras.applications.MobileNetV3Large(
        weights='imagenet', include_top=False, input_tensor=x)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return models.Model(inputs, out, name='MobileNetV3Large'), base

def build_effnetv2_s():
    base = tf.keras.applications.EfficientNetV2S(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return models.Model(base.input, out, name='EfficientNetV2S'), base

def build_proposed():
    base = tf.keras.applications.ConvNeXtTiny(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable = False
    x = se_block(base.output, ratio=16)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return models.Model(base.input, out, name='Proposed_ConvNeXtTiny_SE'), base

BUILDERS = {
    'ConvNeXtSmall' : build_convnext_small,
    'MobileNetV3Large' : build_mobilenetv3,
    'EfficientNetV2S': build_effnetv2_s,
    'Proposed_ConvNeXtTiny_SE': build_proposed,
}
print("Model builders defined ✓")
print("Models:", list(BUILDERS.keys()))

# -------------------------------------------------------------
global_results = {}

def plot_training_history(history_p1, history_p2, model_name, history_p3=None):
    acc1 = history_p1.history['accuracy']
    val_acc1 = history_p1.history['val_accuracy']
    loss1 = history_p1.history['loss']
    val_loss1 = history_p1.history['val_loss']
    
    acc2 = history_p2.history['accuracy']
    val_acc2 = history_p2.history['val_accuracy']
    loss2 = history_p2.history['loss']
    val_loss2 = history_p2.history['val_loss']
    
    acc = acc1 + acc2
    val_acc = val_acc1 + val_acc2
    loss = loss1 + loss2
    val_loss = val_loss1 + val_loss2
    
    p3_start = None
    if history_p3 is not None:
        p3_start = len(acc)
        acc += history_p3.history['accuracy']
        val_acc += history_p3.history['val_accuracy']
        loss += history_p3.history['loss']
        val_loss += history_p3.history['val_loss']
    
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(acc1)-1, color='r', linestyle='--', label='Unfreeze 30%')
    if p3_start is not None:
        plt.axvline(x=p3_start, color='purple', linestyle=':', label='Full Fine-tune')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(loss1)-1, color='r', linestyle='--', label='Unfreeze 30%')
    if p3_start is not None:
        plt.axvline(x=p3_start, color='purple', linestyle=':', label='Full Fine-tune')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Loss')
    
    # Create directory and save the figure
    import os
    os.makedirs(model_name, exist_ok=True)
    fig_path = os.path.join(model_name, f'{model_name}_training_history.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\\nĐã lưu biểu đồ huấn luyện vào: {fig_path}")
    
    plt.show()

def get_callbacks(model_name, phase='', cosine_lr=False):
    import os
    os.makedirs(model_name, exist_ok=True)
    # CSV logs enable safe resume (initial_epoch) after Kaggle session reset.
    phase_suffix = f'_{phase}' if phase else ''
    csv_path = f"{model_name}/{model_name}_train_log{phase_suffix}.csv"
    cbs = [
        callbacks.ModelCheckpoint(f'{model_name}/{model_name}_best.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        callbacks.ModelCheckpoint(f'{model_name}/{model_name}_last.keras', save_best_only=False, verbose=0),
        # Lightweight weight-only checkpoints for fast resume
        callbacks.ModelCheckpoint(f'{model_name}/{model_name}_best.weights.h5', save_best_only=True, save_weights_only=True,
                                  monitor='val_loss', mode='min', verbose=0),
        callbacks.ModelCheckpoint(f'{model_name}/{model_name}_last.weights.h5', save_best_only=False, save_weights_only=True, verbose=0),
        callbacks.CSVLogger(csv_path, append=True),
        callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True, verbose=1),
    ]
    # ReduceLROnPlateau xung đột với CosineDecay (LR schedule không set được)
    if not cosine_lr:
        cbs.append(callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1))
    return cbs
"""))

cells.append(code("""# ── Cosine Annealing LR Schedule ──
def make_cosine_lr(initial_lr, decay_steps):
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=1e-7  # minimum LR
    )

steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
P2_DECAY_STEPS = steps_per_epoch * EPOCHS_P2
print(f"✓ Cosine LR: decay từ 1e-4 → ~0 trong {P2_DECAY_STEPS} steps/phase")"""))

cells.append(code("""# Utility — Sanity checks for labels and losses (chặn lỗi âm/NaN sớm)
def sanity_check_batch(model, ds, loss_name='compiled_loss', n_batches=1):
    \"\"\"Run lightweight checks on a few batches to catch label corruption / unstable loss early.\"\"\"
    import numpy as np
    loss_fn = model.loss if hasattr(model, 'loss') and callable(model.loss) else model.compiled_loss
    card = tf.data.experimental.cardinality(ds).numpy()
    print(f"SanityCheck: ds_cardinality={card}")
    # Guard against silent-empty datasets
    it = iter(ds.take(1))
    try:
        first = next(it)
    except StopIteration:
        raise RuntimeError(
            "Dataset has 0 batches. Likely causes: empty X_train, wrong paths/labels, "
            "or drop_remainder=True with too-large batch_size."
        )
    for b, (imgs, labels) in enumerate(ds.take(n_batches)):
        # Label checks
        lbl_min = tf.reduce_min(labels).numpy().item()
        lbl_max = tf.reduce_max(labels).numpy().item()
        lbl_sum = tf.reduce_mean(tf.reduce_sum(labels, axis=-1)).numpy().item()
        if lbl_min < -1e-6 or lbl_max > 1.0 + 1e-6:
            raise ValueError(f\"Label out of range: min={lbl_min:.6f}, max={lbl_max:.6f}\")
        if not (0.95 <= lbl_sum <= 1.05):
            print(f\"⚠ Label sum mean is unusual (expected ~1): {lbl_sum:.4f}\")

        # Loss checks (single forward pass)
        preds = model(imgs, training=False)
        batch_loss = loss_fn(labels, preds).numpy().item()
        if not np.isfinite(batch_loss):
            raise ValueError(f\"Non-finite loss detected: {batch_loss}\")
        if batch_loss < -1e-6:
            raise ValueError(f\"Negative loss detected: {batch_loss:.6f} (labels or logits/probs likely invalid)\")

        print(f\"Sanity✓ labels[min,max]=({lbl_min:.4f},{lbl_max:.4f}) sum_mean={lbl_sum:.4f} loss={batch_loss:.6f}\")
        break

# Utility — Resume training from last checkpoint (Kaggle-safe)
def get_initial_epoch_from_csv(csv_path):
    try:
        import pandas as pd
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return int(len(df))
    except Exception:
        pass
    return 0

def maybe_resume_weights(model, model_name):
    # ⚠ Resume disabled: MirroredStrategy's CollectiveReduce caches gradient shapes
    # from the first compile(). Resuming Phase 1 weights then recompiling for Phase 2
    # (different trainable vars) causes shape mismatch crash on multi-GPU.
    # Workaround: always train from scratch (12h Kaggle sessions are sufficient).
    return 0
"""))

cells.append(md("### 6.2 Huấn luyện mô hình 1: ConvNeXtSmall"))
cells.append(code("""# -------------------------------------------------------------
# MÔ HÌNH 1: ConvNeXtSmall
# -------------------------------------------------------------

print("\\n" + "="*55 + "\\nBẮT ĐẦU HUẤN LUYỆN: ConvNeXtSmall\\n" + "="*55)

# BƯỚC 1: Khởi tạo mô hình bên trong GPU strategy scope để phân bổ tính toán lên cả 2 lõi T4
with strategy.scope():
    model_conv, base_conv = BUILDERS['ConvNeXtSmall']()
    model_conv.compile(optimizer=optimizers.Adam(1e-3),
                       loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                       metrics=['accuracy'])

print(f"Tổng số tham số: {model_conv.count_params():,}")
sanity_check_batch(model_conv, train_ds, n_batches=1)
init_ep_conv = maybe_resume_weights(model_conv, 'ConvNeXtSmall')

# BƯỚC 2: Phase 1 - Đóng băng (Freeze) lớp Base, chỉ train Classifier Head
print("--- Phase 1: Warm-up (Base hoàn toàn đóng băng) ---")
# Lưu ý: Không dùng class_weight khi có MixUp vì MixUp đã tạo soft-labels
# → truyền class_weight đồng thời gây xung đột gradient và loss âm
history_p1_conv = model_conv.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1,
               initial_epoch=init_ep_conv,
               callbacks=get_callbacks('ConvNeXtSmall', phase='p1'), verbose=1)

# BƯỚC 3: Phase 2 - Rã đông (Unfreeze) 30% các lớp trên cùng của Base model
print("--- Phase 2: Rã đông 30% lớp trên cùng (Fine-tuning) ---")
with strategy.scope():
    base_conv.trainable = True
    freeze_until = int(len(base_conv.layers) * 0.7)
    for l in base_conv.layers[:freeze_until]: l.trainable = False
    
    print(f"  Số layer base được học: {sum(1 for l in base_conv.layers if l.trainable)}/{len(base_conv.layers)}")
    
    # Học với Learning Rate siêu nhỏ để không phá hỏng pre-trained weights
    model_conv.compile(optimizer=optimizers.Adam(make_cosine_lr(1e-4, P2_DECAY_STEPS)),
                       loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                       metrics=['accuracy'])

history_p2_conv = model_conv.fit(train_ds, validation_data=val_ds,
               epochs=EPOCHS_P1 + EPOCHS_P2, initial_epoch=EPOCHS_P1,
               callbacks=get_callbacks('ConvNeXtSmall', phase='p2', cosine_lr=True), verbose=1)

# BƯỚC 4: Dự đoán trên tập Validation & Lưu kết quả
print("--- Đang đánh giá trên tập Validation ---")
_eval = model_conv.evaluate(val_ds, verbose=0)
acc = float(_eval['accuracy']) if isinstance(_eval, dict) else float(_eval[1])
y_pred = []
for imgs, _ in val_ds:
    preds = model_conv.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

global_results['ConvNeXtSmall'] = {'acc': acc, 'y_pred': np.array(y_pred)}
print(f"★ ConvNeXtSmall - Validation Accuracy: {acc:.4f}")
plot_training_history(history_p1_conv, history_p2_conv, 'ConvNeXtSmall')

# Lưu mô hình xuống đĩa để dùng cho Step 10 dự báo tập Test
# (Mô hình tốt nhất và cuối cùng tự động được lưu qua Callbacks)
print("Đã tự động lưu mô hình Best và Last vào thư mục ConvNeXtSmall/")

# BƯỚC 5: Giải phóng bộ nhớ VRAM siêu quan trọng (Tránh OOM cho model tiếp theo)
del model_conv, base_conv
tf.keras.backend.clear_session()
gc.collect()
print("Đã giải phóng bộ nhớ VRAM thành công ✓")"""))

cells.append(md("### 6.3 Huấn luyện mô hình 2: MobileNetV3Large"))
cells.append(code("""# -------------------------------------------------------------
# MÔ HÌNH 2: MobileNetV3Large
# -------------------------------------------------------------
print("\\n" + "="*55 + "\\nBẮT ĐẦU HUẤN LUYỆN: MobileNetV3Large\\n" + "="*55)

# BƯỚC 1: Khởi tạo mô hình
with strategy.scope():
    model_x, base_x = BUILDERS['MobileNetV3Large']()
    model_x.compile(optimizer=optimizers.Adam(1e-3),
                    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

print(f"Tổng số tham số: {model_x.count_params():,}")
sanity_check_batch(model_x, train_ds, n_batches=1)
init_ep_x = maybe_resume_weights(model_x, 'MobileNetV3Large')

# BƯỚC 2: Phase 1 (Đóng băng Base)
print("--- Phase 1: Warm-up (Base hoàn toàn đóng băng) ---")
history_p1_x = model_x.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1,
            initial_epoch=init_ep_x,
            callbacks=get_callbacks('MobileNetV3Large', phase='p1'), verbose=1)

# BƯỚC 3: Phase 2 (Rã đông 30%)
print("--- Phase 2: Rã đông 30% lớp trên cùng (Fine-tuning) ---")
with strategy.scope():
    base_x.trainable = True
    freeze_until = int(len(base_x.layers) * 0.7)
    for l in base_x.layers[:freeze_until]: l.trainable = False
    model_x.compile(optimizer=optimizers.Adam(make_cosine_lr(1e-4, P2_DECAY_STEPS)),
                    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

history_p2_x = model_x.fit(train_ds, validation_data=val_ds,
            epochs=EPOCHS_P1 + EPOCHS_P2, initial_epoch=EPOCHS_P1,
            callbacks=get_callbacks('MobileNetV3Large', phase='p2', cosine_lr=True), verbose=1)

# BƯỚC 4: Đánh giá & Lưu kết quả
print("--- Đang đánh giá trên tập Validation ---")
_eval = model_x.evaluate(val_ds, verbose=0)
acc = float(_eval['accuracy']) if isinstance(_eval, dict) else float(_eval[1])
y_pred = []
for imgs, _ in val_ds:
    preds = model_x.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

global_results['MobileNetV3Large'] = {'acc': acc, 'y_pred': np.array(y_pred)}
print(f"★ MobileNetV3Large - Validation Accuracy: {acc:.4f}")
plot_training_history(history_p1_x, history_p2_x, 'MobileNetV3Large')

# Lưu mô hình xuống đĩa
print("Đã tự động lưu mô hình Best và Last vào thư mục MobileNetV3Large/")

# BƯỚC 5: Giải phóng VRAM
del model_x, base_x
tf.keras.backend.clear_session()
gc.collect()
print("Đã giải phóng bộ nhớ VRAM thành công ✓")"""))

cells.append(md("### 6.4 Huấn luyện mô hình 3: EfficientNetV2S"))
cells.append(code("""# -------------------------------------------------------------
# MÔ HÌNH 3: EfficientNetV2S
# -------------------------------------------------------------
print("\\n" + "="*55 + "\\nBẮT ĐẦU HUẤN LUYỆN: EfficientNetV2S\\n" + "="*55)

# BƯỚC 1: Khởi tạo mô hình
with strategy.scope():
    model_eff, base_eff = BUILDERS['EfficientNetV2S']()
    model_eff.compile(optimizer=optimizers.Adam(1e-3),
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

print(f"Tổng số tham số: {model_eff.count_params():,}")
sanity_check_batch(model_eff, train_ds, n_batches=1)
init_ep_eff = maybe_resume_weights(model_eff, 'EfficientNetV2S')

# BƯỚC 2: Phase 1 (Đóng băng Base)
print("--- Phase 1: Warm-up (Base hoàn toàn đóng băng) ---")
history_p1_eff = model_eff.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1,
              initial_epoch=init_ep_eff,
              callbacks=get_callbacks('EfficientNetV2S', phase='p1'), verbose=1)

# BƯỚC 3: Phase 2 (Rã đông 30%)
print("--- Phase 2: Rã đông 30% lớp trên cùng (Fine-tuning) ---")
with strategy.scope():
    base_eff.trainable = True
    freeze_until = int(len(base_eff.layers) * 0.7)
    for l in base_eff.layers[:freeze_until]: l.trainable = False
    model_eff.compile(optimizer=optimizers.Adam(make_cosine_lr(1e-4, P2_DECAY_STEPS)),
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

history_p2_eff = model_eff.fit(train_ds, validation_data=val_ds,
            epochs=EPOCHS_P1 + EPOCHS_P2, initial_epoch=EPOCHS_P1,
            callbacks=get_callbacks('EfficientNetV2S', phase='p2', cosine_lr=True), verbose=1)

# BƯỚC 4: Đánh giá & Lưu kết quả
print("--- Đang đánh giá trên tập Validation ---")
_eval = model_eff.evaluate(val_ds, verbose=0)
acc = float(_eval['accuracy']) if isinstance(_eval, dict) else float(_eval[1])
y_pred = []
for imgs, _ in val_ds:
    preds = model_eff.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

global_results['EfficientNetV2S'] = {'acc': acc, 'y_pred': np.array(y_pred)}
print(f"★ EfficientNetV2S - Validation Accuracy: {acc:.4f}")
plot_training_history(history_p1_eff, history_p2_eff, 'EfficientNetV2S')

# Lưu mô hình xuống đĩa
print("Đã tự động lưu mô hình Best và Last vào thư mục EfficientNetV2S/")

# BƯỚC 5: Giải phóng VRAM
del model_eff, base_eff
tf.keras.backend.clear_session()
gc.collect()
print("Đã giải phóng bộ nhớ VRAM thành công ✓")"""))

cells.append(md("### 6.5 Huấn luyện mô hình Đề xuất: Proposed_ConvNeXtTiny_SE + Focal Loss"))
cells.append(code(r"""# -------------------------------------------------------------
# MÔ HÌNH 4: Proposed_ConvNeXtTiny_SE (Sử dụng Focal Loss)
# -------------------------------------------------------------
print("\n" + "="*55 + "\nBẮT ĐẦU HUẤN LUYỆN: Proposed_ConvNeXtTiny_SE\n" + "="*55)

# BƯỚC 1: Khởi tạo mô hình
with strategy.scope():
    model_prop, base_prop = BUILDERS['Proposed_ConvNeXtTiny_SE']()
    alpha_weights = [class_weight_dict[i] for i in range(NUM_CLASSES)]
    model_prop.compile(optimizer=optimizers.Adam(1e-3),
                      loss=CategoricalFocalLoss(alpha=alpha_weights), metrics=['accuracy'])

print(f"Tổng số tham số: {model_prop.count_params():,}")
sanity_check_batch(model_prop, train_ds, n_batches=1)
init_ep_prop = maybe_resume_weights(model_prop, 'Proposed_ConvNeXtTiny_SE')

# BƯỚC 2: Phase 1 (Đóng băng Base)
print("--- Phase 1: Warm-up (Base hoàn toàn đóng băng) ---")
history_p1_prop = model_prop.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1,
              initial_epoch=init_ep_prop,
              callbacks=get_callbacks('Proposed_ConvNeXtTiny_SE', phase='p1'), verbose=1)

# BƯỚC 3: Phase 2 (Rã đông 30%)
print("--- Phase 2: Rã đông 30% lớp trên cùng (Fine-tuning) ---")
with strategy.scope():
    base_prop.trainable = True
    freeze_until = int(len(base_prop.layers) * 0.7)
    for l in base_prop.layers[:freeze_until]: l.trainable = False
    model_prop.compile(optimizer=optimizers.Adam(make_cosine_lr(1e-4, P2_DECAY_STEPS)),
                      loss=CategoricalFocalLoss(alpha=alpha_weights), metrics=['accuracy'])

history_p2_prop = model_prop.fit(train_ds, validation_data=val_ds,
            epochs=EPOCHS_P1 + EPOCHS_P2, initial_epoch=EPOCHS_P1,
            callbacks=get_callbacks('Proposed_ConvNeXtTiny_SE', phase='p2', cosine_lr=True), verbose=1)

# BƯỚC 4: Đánh giá & Lưu kết quả
print("--- Đang đánh giá trên tập Validation ---")
_eval = model_prop.evaluate(val_ds, verbose=0)
acc = float(_eval['accuracy']) if isinstance(_eval, dict) else float(_eval[1])
y_pred = []
for imgs, _ in val_ds:
    preds = model_prop.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

global_results['Proposed_ConvNeXtTiny_SE'] = {'acc': acc, 'y_pred': np.array(y_pred)}
print(f"★ Proposed_ConvNeXtTiny_SE - Validation Accuracy: {acc:.4f}")
plot_training_history(history_p1_prop, history_p2_prop, 'Proposed_ConvNeXtTiny_SE')

# Lưu mô hình xuống đĩa
print("Đã tự động lưu mô hình Best và Last vào thư mục Proposed_ConvNeXtTiny_SE/")

# BƯỚC 5: Giải phóng VRAM
del model_prop, base_prop
tf.keras.backend.clear_session()
gc.collect()
print("Đã giải phóng bộ nhớ VRAM thành công ✓")
print("\n🎉 HOÀN THÀNH HUẤN LUYỆN TẤT CẢ MÔ HÌNH!")"""))

# ─────────────────────────────────────────────────────
cells.append(md("### 6.6 Test-Time Augmentation (TTA) Utility"))
cells.append(code("""# TTA — Trung bình dự đoán trên nhiều biến thể augmented để tăng accuracy
def predict_with_tta(model, ds, n_aug=4):
    \"\"\"Predict with Test-Time Augmentation: original + flipped variants.\"\"\"
    all_preds = []
    for imgs, _ in ds:
        # Original
        p = model.predict(imgs, verbose=0)
        all_preds_batch = [p]
        # Flip left-right
        all_preds_batch.append(model.predict(tf.image.flip_left_right(imgs), verbose=0))
        # Flip up-down
        all_preds_batch.append(model.predict(tf.image.flip_up_down(imgs), verbose=0))
        # Flip both
        all_preds_batch.append(model.predict(tf.image.flip_left_right(tf.image.flip_up_down(imgs)), verbose=0))
        # Average
        avg_pred = np.mean(all_preds_batch, axis=0)
        all_preds.extend(np.argmax(avg_pred, axis=1))
    return np.array(all_preds)

print("✓ TTA utility loaded (4 variants: original + 3 flips)")"""))

# STEP 7
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 7: Đánh giá Mô hình (Data Evaluation)
> **Hiển thị các chỉ số Đánh giá cho Bài toán Phân loại (Accuracy, Precision, Recall)**
> *(Display Metrics for Classification (Accuracy, Precision, Recall), Display Metrics for Regression (R2Score, MSE, RMSE))*"""))

cells.append(code("""# Step 7.1 — Collect true labels from val_ds
y_true_val = []
for _, lbs in val_ds: y_true_val.extend(np.argmax(lbs.numpy(), axis=1))
y_true_val = np.array(y_true_val)

# Truncate predictions to match (due to drop_remainder)
print("Model Comparison (Validation Set):")
print(f"{'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Macro':>10} {'F1-Weight':>10}")
print("-"*70)

detailed_results = {}
# Re-evaluate với TTA cho kết quả tốt hơn
print("\\nÁp dụng Test-Time Augmentation (TTA) cho đánh giá...")
for name in list(global_results.keys()):
    # Load best model cho TTA
    tta_model_path = f"{name}/{name}_best.keras"
    if os.path.exists(tta_model_path):
        with strategy.scope():
            tta_m = tf.keras.models.load_model(tta_model_path, compile=False,
                custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})
        tta_pred = predict_with_tta(tta_m, val_ds)
        global_results[name]['y_pred'] = tta_pred
        del tta_m
        gc.collect()  # clear_session() ở đây sẽ phá strategy cho model tiếp theo
    
for name, res in global_results.items():
    yp = res['y_pred'][:len(y_true_val)]
    yt = y_true_val[:len(yp)]
    metrics = {
        'Accuracy' : accuracy_score(yt, yp),
        'Precision': precision_score(yt, yp, average='macro', zero_division=0),
        'Recall'   : recall_score(yt, yp, average='macro', zero_division=0),
        'F1-Macro' : f1_score(yt, yp, average='macro', zero_division=0),
        'F1-Weight': f1_score(yt, yp, average='weighted', zero_division=0),
    }
    detailed_results[name] = {'y_true': yt, 'y_pred': yp, 'metrics': metrics}
    print(f"{name:<18} {metrics['Accuracy']:>10.4f} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} {metrics['F1-Macro']:>10.4f} {metrics['F1-Weight']:>10.4f}")

best_model_name = max(detailed_results, key=lambda k: detailed_results[k]['metrics']['F1-Macro'])
print(f"\\n🏆 Best model: {best_model_name} (F1-Macro={detailed_results[best_model_name]['metrics']['F1-Macro']:.4f})")"""))

cells.append(md("### 7.2 Confusion Matrices"))
cells.append(code("""# Step 7.2 — Confusion matrices for all models
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
for i, (name, res) in enumerate(detailed_results.items()):
    ax = axes[i]
    cm = confusion_matrix(res['y_true'], res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title(f"{name}\\nAcc={res['metrics']['Accuracy']:.3f}", fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    
plt.tight_layout()
plt.savefig("Validation_Models_Confusion_Matrices.png", dpi=300, bbox_inches='tight')
plt.show()"""))

cells.append(md("### 7.3 Classification Report (Best Model)"))
cells.append(code("""# Step 7.3 — Detailed report for best model
best_res = detailed_results[best_model_name]
print(f"Classification Report — {best_model_name} (Validation)")
print("="*65)
print(classification_report(best_res['y_true'], best_res['y_pred'],
                            target_names=CLASS_NAMES, digits=4))"""))

cells.append(md("### 7.4 Knowledge Distillation — Teacher → Student (MobileNetV3Large)"))
cells.append(code(r"""# Step 7.4 — Knowledge Distillation (KD)
# Ý tưởng: Teacher (model mạnh nhất) tạo phân phối soft-label; Student học theo để vừa nhẹ vừa chính xác.
# KD phù hợp để nâng điểm 'cải thiện thuật toán' và tạo giá trị học thuật.

KD_ENABLE = True
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.3          # weight for hard CE; (1-ALPHA) for soft KL
KD_EPOCHS = 3

def make_ds_no_mixup(paths, labels, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds

train_ds_hard = make_ds_no_mixup(X_train, y_train, BATCH_SIZE, shuffle=True)
val_ds_hard   = make_ds_no_mixup(X_val,   y_val,   BATCH_SIZE, shuffle=False)

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=4.0, alpha=0.3):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.ce = tf.keras.losses.CategoricalCrossentropy()
        self.kld = tf.keras.losses.KLDivergence()

    def compile(self, optimizer, metrics, loss=None, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics, loss=loss, **kwargs)

    def call(self, x):
        return self.student(x)

    def train_step(self, data):
        x, y = data
        # Teacher prediction (no grad)
        teacher_pred = tf.stop_gradient(self.teacher(x, training=False))
        # Temperature scaling on probabilities (approx KD; teacher outputs probs)
        t = self.temperature
        teacher_soft = tf.nn.softmax(tf.math.log(tf.clip_by_value(teacher_pred, 1e-7, 1.0)) / t)
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            student_soft = tf.nn.softmax(tf.math.log(tf.clip_by_value(student_pred, 1e-7, 1.0)) / t)
            hard_loss = self.ce(y, student_pred)
            soft_loss = self.kld(teacher_soft, student_soft) * (t ** 2)
            loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, student_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs.update({"loss": loss, "hard_loss": hard_loss, "soft_loss": soft_loss})
        return logs

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        hard_loss = self.ce(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs.update({"loss": hard_loss})
        return logs

if KD_ENABLE:
    print("\\n--- KD: loading Teacher and training Student ---")
    with strategy.scope():
        teacher = tf.keras.models.load_model(f"{best_model_name}/{best_model_name}_best.keras", compile=False,
            custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})
        student, _ = BUILDERS["MobileNetV3Large"]()
        student.compile(optimizer=optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
        distiller = Distiller(student=student, teacher=teacher, temperature=KD_TEMPERATURE, alpha=KD_ALPHA)
        distiller.compile(optimizer=optimizers.Adam(1e-3), metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
    kd_hist = distiller.fit(train_ds_hard, validation_data=val_ds_hard, epochs=KD_EPOCHS, verbose=1)
    kd_result = distiller.evaluate(val_ds_hard, verbose=0)
    # BULLETPROOF extraction — Keras 3 evaluate() returns wildly different types:
    # dict, list, list-of-dicts, nested, tensor values, etc.
    def _extract_acc(res):
        # Extract accuracy float from any evaluate() return format
        if isinstance(res, dict):
            for key in ['acc', 'accuracy', 'categorical_accuracy']:
                if key in res:
                    v = res[key]
                    return float(v.numpy()) if hasattr(v, 'numpy') else float(v)
            # fallback: last value in dict
            v = list(res.values())[-1]
            return float(v.numpy()) if hasattr(v, 'numpy') else float(v)
        elif isinstance(res, (list, tuple)):
            # Try index 1 first (typically accuracy); if it's a dict, recurse
            for idx in [1, -1, 0]:
                if idx < len(res):
                    v = res[idx]
                    if isinstance(v, dict):
                        return _extract_acc(v)
                    try:
                        return float(v.numpy()) if hasattr(v, 'numpy') else float(v)
                    except (TypeError, ValueError):
                        continue
            return 0.0
        else:
            return float(res.numpy()) if hasattr(res, 'numpy') else float(res)
    kd_acc = _extract_acc(kd_result)
    print(f"★ KD Student (MobileNetV3Large) Val Acc: {kd_acc:.4f}")
    student.save("KD_Student_MobileNetV3Large.keras")
else:
    print("KD disabled")"""))

# ─────────────────────────────────────────────────────
# STEP 8
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 8: Tinh chỉnh Siêu tham số (Hyper-parameter Tuning)
> **Tinh chỉnh các tham số: Tương đương Cross Validation (CV), GridSearchCV, Regularization (L1, L2 penalty)**

Trong bài toán Deep Learning Image Classification này, thay vì GridSearch, ta sẽ Tuning bằng cách:
1. **Tinh chỉnh Fine-Tuning Sâu (Phase 3)**: Unfreeze toàn bộ base layer cho mô hình chiến thắng ở Bước 7.
2. **Learning Rate Decay**: Dùng callbacks `ReduceLROnPlateau` tự động giảm LR.
3. **Regularization L2/Dropout**: Gọi lớp Dropout(0.3) tại Classifier Head, dùng EarlyStopping để tránh Overfitting."""))

cells.append(code("""# Step 8.1 — Rebuild best model cho Phase 3 (Full Fine-tune)
print(f"Bắt đầu Rebuild model tốt nhất: {best_model_name}")

# Lấy builder function tương ứng với model tốt nhất
builder_fn = BUILDERS[best_model_name]

# Xác định hàm loss (Bảo toàn Focal Loss cho kiến trúc Proposed)
alpha_weights = [class_weight_dict[i] for i in range(NUM_CLASSES)]
loss_fn = CategoricalFocalLoss(alpha=alpha_weights) if best_model_name == 'Proposed_ConvNeXtTiny_SE' else tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Khởi tạo lại model tốt nhất từ đầu bên trong strategy scope
with strategy.scope():
    best_model, best_base = builder_fn()
    best_model.compile(optimizer=optimizers.Adam(1e-3), loss=loss_fn, metrics=['accuracy'])

# Load the best weights từ phase 2 đã lưu
best_model.load_weights(f"{best_model_name}/{best_model_name}_best.weights.h5")
print(f"\\nĐã tải lại trọng số tốt nhất ({best_model_name}_best.weights.h5) sau 2 phase.")

# Trực tiếp tiến vào Phase 3: Full fine-tune với Learning Rate siêu nhỏ
with strategy.scope():
    for l in best_base.layers: l.trainable = True
    best_model.compile(optimizer=optimizers.Adam(1e-5), loss=loss_fn, metrics=['accuracy'])

TUNING_CBS = [
    callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True, verbose=1),
    # Tự động giảm Learning rate nếu val_loss không cải thiện
    callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=2, min_lr=1e-8, verbose=1),
]

print("\\n--- Phase 3: Full Fine-tune (Học toàn bộ kiến trúc với LR siêu nhỏ 1e-5) ---")
phase3_history = best_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P3,
                                callbacks=TUNING_CBS, verbose=1)
print("✓ Phase 3 hoàn tất")"""))

cells.append(md("### 8.2 Phase 3 Training Curve"))
cells.append(code("""# Step 8.2 — Phase 3 training curve
h3 = phase3_history.history
ep = range(1, len(h3['accuracy'])+1)
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle(f'Phase 3 Full Fine-tune: {best_model_name}', fontsize=13, fontweight='bold')
axes[0].plot(ep, h3['accuracy'],'b-o',label='Train',markersize=4)
axes[0].plot(ep, h3['val_accuracy'],'r-o',label='Val',markersize=4)
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep, h3['loss'],'b-o',label='Train',markersize=4)
axes[1].plot(ep, h3['val_loss'],'r-o',label='Val',markersize=4)
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()"""))

cells.append(md("### 8.3 Evaluation After Phase 3"))
cells.append(code("""# Step 8.3 — Compare val metrics before vs after Phase 3
y_pred_tuned = []
for imgs, _ in val_ds:
    preds = best_model.predict(imgs, verbose=0)
    y_pred_tuned.extend(np.argmax(preds, axis=1))
y_pred_tuned = np.array(y_pred_tuned)[:len(y_true_val)]

tuned_metrics = {
    'Accuracy' : accuracy_score(y_true_val, y_pred_tuned),
    'Precision': precision_score(y_true_val, y_pred_tuned, average='macro', zero_division=0),
    'Recall'   : recall_score(y_true_val, y_pred_tuned,   average='macro', zero_division=0),
    'F1-Macro' : f1_score(y_true_val, y_pred_tuned,       average='macro', zero_division=0),
    'F1-Weight': f1_score(y_true_val, y_pred_tuned,       average='weighted', zero_division=0),
}
print(f"After Phase 3 Tuning — {best_model_name}:")
print(f"{'Metric':<12} {'Before':>8} {'After':>8} {'Δ':>8}")
print("-"*40)
for k in tuned_metrics:
    before = detailed_results[best_model_name]['metrics'][k]
    after  = tuned_metrics[k]
    print(f"{k:<12} {before:>8.4f} {after:>8.4f} {after-before:>+8.4f}")"""))

cells.append(md("### 8.3b Ablation: Ảnh gốc vs Ảnh tách nền (Segmentation)"))
cells.append(code("""# Step 8.3b — Evaluate best model on original vs segmented validation
if train_ds_seg is not None and val_ds_seg is not None:
    print("\\n--- Ablation: Original vs Segmented (Validation) ---")
    _eval_o = best_model.evaluate(val_ds, verbose=0)
    orig_acc = float(_eval_o['accuracy']) if isinstance(_eval_o, dict) else float(_eval_o[1])
    _eval_s = best_model.evaluate(val_ds_seg, verbose=0)
    seg_acc = float(_eval_s['accuracy']) if isinstance(_eval_s, dict) else float(_eval_s[1])
    print(f"Best model: {best_model_name}")
    print(f"  Val Acc (Original) : {orig_acc:.4f}")
    print(f"  Val Acc (Segmented): {seg_acc:.4f}")
    print(f"  Δ (Seg - Orig)     : {seg_acc - orig_acc:+.4f}")
else:
    print("Segmented datasets not available (SEG_ENABLE=False). Skipping ablation.")"""))

cells.append(md("### 8.4 Lưu Mô hình Tốt nhất (Save Best Model)"))
cells.append(code("""# Step 8.4 — Lưu Best Model và weights
import os

os.makedirs(best_model_name, exist_ok=True)
best_model_path = f"{best_model_name}/{best_model_name}_best_tuned.keras"
best_model.save(best_model_path)
print(f"\\n✅ Đã lưu toàn bộ kiến trúc và trọng số (best weights) của mô hình tốt nhất vào:")
print(f"   -> {os.path.abspath(best_model_path)}")
print("Bạn có thể tải file này về để làm Deployment (Giao diện Web/App) sau này.")"""))

# ─────────────────────────────────────────────────────
# STEP 9
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 9: Xây dựng Pipeline với Mô hình & Tham số tốt nhất
> **(Build the pipeline with the best Model with the best parameters: Choose best hyper-parameters and build best models)**

Thiết lập một Class Python `RiceDiseasePipeline` hoàn chỉnh, gói gọn khâu load ảnh + tiền xử lý + dự đoán + biểu diễn visual Top-3 tự động."""))

cells.append(code("""# Step 9 — End-to-end inference pipeline
class RiceDiseasePipeline:
    \"\"\"End-to-end inference pipeline for rice disease prediction.\"\"\"
    def __init__(self, model, class_names, img_size=224):
        self.model = model
        self.class_names = class_names
        self.img_size = img_size

    def preprocess(self, image_path):
        img = load_img(image_path, target_size=(self.img_size, self.img_size))
        # Đồng bộ với training pipeline: giữ pixel ở range [0, 255] float32.
        # (Backbone có include_preprocessing=True hoặc preprocess_input riêng sẽ tự normalize đúng cách.)
        arr = img_to_array(img).astype(np.float32)
        return np.expand_dims(arr, 0), img

    def predict(self, image_path, top_k=3):
        arr, original = self.preprocess(image_path)
        probs = self.model.predict(arr, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1][:top_k]
        results = [{'class': self.class_names[i], 'confidence': float(probs[i])} for i in top_idx]
        return results, original

    def predict_and_display(self, image_path):
        results, original = self.predict(image_path)
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
        ax1.imshow(original)
        ax1.set_title(f"Dự đoán: {results[0]['class']}\\nConfidence: {results[0]['confidence']:.1%}",
                      fontweight='bold',
                      color='green' if results[0]['confidence'] > 0.7 else 'orange')
        ax1.axis('off')
        cls_list  = [r['class'] for r in results]
        conf_list = [r['confidence'] for r in results]
        bar_colors = ['#2ecc71' if c>0.7 else '#e74c3c' for c in conf_list]
        ax2.barh(cls_list, conf_list, color=bar_colors, edgecolor='k')
        ax2.set_xlim(0,1); ax2.set_title('Top-3 Predictions', fontweight='bold')
        for i,(c,v) in enumerate(zip(cls_list,conf_list)):
            ax2.text(v+0.01, i, f'{v:.1%}', va='center', fontweight='bold')
        plt.tight_layout(); plt.show()
        return results

pipeline = RiceDiseasePipeline(best_model, CLASS_NAMES, IMG_SIZE)
print(f"✓ Pipeline created: {best_model_name}")

# Demo trên 3 ảnh validation ngẫu nhiên
print("\\nDemo predictions:")
for i in range(3):
    idx = np.random.randint(0, len(X_val))
    true_lbl = CLASS_NAMES[y_val[idx]]
    print(f"\\n--- Sample {i+1} (True: {true_lbl}) ---")
    pipeline.predict_and_display(X_val[idx])"""))

cells.append(md("### 9.2 Ensemble (Soft-voting) — Chỉ dùng 4 mô hình hiện có"))
cells.append(code(r"""# Step 9.2 — Soft-voting ensemble (NO extra model)
# Chỉ dùng output probabilities của 4 mô hình hiện có để ensemble.

ENSEMBLE_ENABLE = True
ENSEMBLE_MODELS = ['ConvNeXtSmall', 'MobileNetV3Large', 'EfficientNetV2S', 'Proposed_ConvNeXtTiny_SE']

def load_if_exists(name):
    p = f"{name}/{name}_best.keras"
    if os.path.exists(p):
        with strategy.scope():
            return tf.keras.models.load_model(p, compile=False,
                custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})
    return None

if ENSEMBLE_ENABLE:
    models_ens = [(n, load_if_exists(n)) for n in ENSEMBLE_MODELS]
    models_ens = [(n,m) for n,m in models_ens if m is not None]
    if len(models_ens) < 2:
        print("Not enough trained models found for ensemble. Skipping.")
    else:
        print("Ensembling models:", [n for n,_ in models_ens])
        y_true_e, y_pred_e = [], []
        for x, y in val_ds:
            probs = None
            for _, m in models_ens:
                p = m.predict(x, verbose=0)
                probs = p if probs is None else (probs + p)
            probs = probs / float(len(models_ens))
            y_pred_e.extend(np.argmax(probs, axis=1))
            y_true_e.extend(np.argmax(y.numpy(), axis=1))
        y_true_e = np.array(y_true_e)
        y_pred_e = np.array(y_pred_e)
        acc_e = accuracy_score(y_true_e, y_pred_e)
        f1_e = f1_score(y_true_e, y_pred_e, average="macro", zero_division=0)
        print(f"★ Soft-voting Ensemble (Val) Acc={acc_e:.4f} | F1-macro={f1_e:.4f}")
        # Cleanup
        for _, m in models_ens:
            del m
        tf.keras.backend.clear_session()
else:
    print("Ensemble disabled")"""))

# ─────────────────────────────────────────────────────
# STEP 10
# ─────────────────────────────────────────────────────
cells.append(md("""---
## Bước 10: Kết luận (Conclusion)
> **Đánh giá trên Test Set thực tiễn + Trực quan giải thích Grad-CAM + Tóm tắt tổng thể**

### 🔓 Mở khóa Test Set
Test set đã **hoàn toàn không được dùng** trong Steps 6–9.  
Đây là lần đầu tiên và duy nhất ta đánh giá trên test data."""))

cells.append(code("""# Step 10.1 — Đánh giá TẤT CẢ model trên tập TEST và vẽ biểu đồ so sánh
print("🔓 UNLOCKING TEST SET — ĐÁNH GIÁ ĐỒNG LOẠT 4 MÔ HÌNH")
print("="*60)

from sklearn.metrics import cohen_kappa_score
test_comparison = {}
y_true_test_global = None

for m_name in ['ConvNeXtSmall', 'MobileNetV3Large', 'EfficientNetV2S', 'Proposed_ConvNeXtTiny_SE']:
    print(f"\\nĐang tải và dự báo trên model: {m_name}...")
    with strategy.scope():
        tmp_model = tf.keras.models.load_model(f"{m_name}/{m_name}_best.keras", compile=False,
            custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})
    
    y_pred, y_true = [], []
    for imgs, lbs in test_ds:
        preds = tmp_model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(lbs.numpy(), axis=1))
        
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if y_true_test_global is None: y_true_test_global = y_true
    

    metrics = {
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall'   : recall_score(y_true, y_pred,   average='macro', zero_division=0),
        'F1-Macro' : f1_score(y_true, y_pred,       average='macro', zero_division=0),
        'F1-Weight': f1_score(y_true, y_pred,       average='weighted', zero_division=0)
    }
    
    if y_true.ndim > 1:
        yt_for_kappa = np.argmax(y_true, axis=1)
    else:
        yt_for_kappa = y_true
    metrics['Kappa'] = cohen_kappa_score(yt_for_kappa, y_pred)
    
    test_comparison[m_name] = metrics
    
    print(f"Kết quả {m_name}:")
    for k,v in metrics.items(): print(f"  {k:<12}: {v:.4f}")
    
    # Giải phóng VRAM ngay sau khi dự báo xong 1 model
    del tmp_model
    tf.keras.backend.clear_session()
    gc.collect()

# Trực quan hóa So sánh 3 model trên Test Set
fig, ax = plt.subplots(figsize=(10, 5))
df_test_comp = pd.DataFrame(test_comparison).T
df_test_comp.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='k')
ax.set_title("So sánh Hiệu suất 4 Mô hình trên Tập Kiểm thử (Test Set)", fontweight='bold', fontsize=14)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc='lower left')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() * 1.005, p.get_height() * 1.01), fontsize=8, rotation=90)
plt.tight_layout()
plt.savefig("Test_Dataset_Metrics_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()"""))

cells.append(md("### 10.2 Đánh giá Chi tiết cho Mô hình Tốt nhất (Sau khi Tuning)"))
cells.append(code("""# Step 10.2 — Dự báo lại trên Test Data cho mô hình tốt nhất (Best Tuned Model)
print(f"\\n--- Đánh giá mô hình siêu việt Nhất (Đã qua Fine-Tuning Phase 3): {best_model_name} ---")
# Reload best_model vi Steps 9.2 va 10.1 goi clear_session() co the invalidate model cu
with strategy.scope():
    best_model = tf.keras.models.load_model(f"{best_model_name}/{best_model_name}_best_tuned.keras", compile=False,
        custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})


y_pred_best = []
for imgs, _ in test_ds:
    preds = best_model.predict(imgs, verbose=0)
    y_pred_best.extend(np.argmax(preds, axis=1))
y_pred_best = np.array(y_pred_best)
# Truncate to match lengths
y_pred_best = y_pred_best[:len(y_true_test_global)]

test_metrics = {
    'Accuracy' : accuracy_score(y_true_test_global, y_pred_best),
    'Precision': precision_score(y_true_test_global, y_pred_best, average='macro', zero_division=0),
    'Recall'   : recall_score(y_true_test_global, y_pred_best,   average='macro', zero_division=0),
    'F1-Macro' : f1_score(y_true_test_global, y_pred_best,       average='macro', zero_division=0),
    'F1-Weight': f1_score(y_true_test_global, y_pred_best,       average='weighted', zero_division=0),
}
if y_true_test_global.ndim > 1:
    yt_for_kappa = np.argmax(y_true_test_global, axis=1)
else:
    yt_for_kappa = y_true_test_global
test_metrics['Kappa'] = cohen_kappa_score(yt_for_kappa, y_pred_best)

print(f"\\nFINAL TEST RESULTS — {best_model_name} (Tuned)")
print("="*60)
for k,v in test_metrics.items(): print(f"  {k:<12}: {v:.4f} ({v:.1%})")
print(f"\\nClassification Report (Báo cáo Phân loại Chi tiết):")
report = classification_report(y_true_test_global, y_pred_best, target_names=CLASS_NAMES, digits=4, output_dict=True)
print(classification_report(y_true_test_global, y_pred_best, target_names=CLASS_NAMES, digits=4))

# Convert Classification Report to DataFrame and save as image
# This creates a beautifully formatted table image for presentations
fig_report, ax_report = plt.subplots(figsize=(10, 4))
ax_report.axis('tight')
ax_report.axis('off')
df_report = pd.DataFrame(report).transpose().round(4)
table = ax_report.table(cellText=df_report.values, rowLabels=df_report.index, colLabels=df_report.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title(f"Classification Report - {best_model_name}")
plt.savefig(f"{best_model_name}/{best_model_name}_classification_report_table.png", dpi=300, bbox_inches='tight')
plt.close(fig_report)"""))

cells.append(md("### 10.3 Ma trận Nhầm lẫn (Confusion Matrix) cuối cùng"))
cells.append(code("""# Step 10.3 — Final test confusion matrix
fig, ax = plt.subplots(figsize=(10,8))
cm = confusion_matrix(y_true_test_global, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ax.set_title(f"Test Confusion Matrix — {best_model_name}\\nAccuracy: {test_metrics['Accuracy']:.4f} | F1-Macro: {test_metrics['F1-Macro']:.4f}",
             fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
import os
os.makedirs(best_model_name, exist_ok=True)
plt.savefig(f"{best_model_name}/{best_model_name}_final_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.tight_layout(); plt.show()"""))

cells.append(md("### 10.4 Đồ thị ROC-AUC (Đánh giá khả năng phân loại)"))
cells.append(code("""# Step 10.4 — Vẽ đồ thị ROC-AUC đa lớp (Multi-class ROC Curve)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import os

# Binarize nhãn thực để vẽ ROC cho bài toán đa lớp
# Vì test_ds trả về y_true_test_global dạng nhãn integer
if y_true_test_global.ndim == 1:
    y_test_bin = label_binarize(y_true_test_global, classes=range(NUM_CLASSES))
else:
    y_test_bin = y_true_test_global # Nếu đã lấy được one-hot format
    
# Phải thu thập lại probabilities
y_prob_best = []
for imgs, _ in test_ds:
    preds = best_model.predict(imgs, verbose=0)
    y_prob_best.extend(preds)
y_prob_best = np.array(y_prob_best)

# Truncate to match lengths (drop_remainder may cause mismatch)
min_len = min(len(y_test_bin), len(y_prob_best))
y_test_bin = y_test_bin[:min_len]
y_prob_best = y_prob_best[:min_len]

# Vẽ đồ thị ROC
plt.figure(figsize=(10, 8))
colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, NUM_CLASSES)) if hasattr(plt, 'colormaps') else plt.cm.get_cmap('tab10')(np.linspace(0, 1, NUM_CLASSES))

for i, color in zip(range(NUM_CLASSES), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_best[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of class {CLASS_NAMES[i]} (area = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title(f'Multi-class ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)

os.makedirs(best_model_name, exist_ok=True)
plt.savefig(f"{best_model_name}/{best_model_name}_roc_auc_curve.png", dpi=300, bbox_inches='tight')
plt.show()"""))

cells.append(md("### 10.5 Suy luận, Model Size, và FPS (Physical Resource Info)"))
cells.append(code("""# Đánh giá thông số phần cứng thiết bị (Lưu ý: Chạy trên Kaggle GPU)
import time
import os

hardware_stats = []

for m_name in ['ConvNeXtSmall', 'MobileNetV3Large', 'EfficientNetV2S', 'Proposed_ConvNeXtTiny_SE']:
    tmp_model = None
    with strategy.scope():
        if os.path.exists(f"{m_name}/{m_name}_best.keras"):
            tmp_model = tf.keras.models.load_model(f"{m_name}/{m_name}_best.keras", compile=False,
                custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})
        elif os.path.exists(f"{m_name}/{m_name}_best_tuned.keras"):
            tmp_model = tf.keras.models.load_model(f"{m_name}/{m_name}_best_tuned.keras", compile=False,
                custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})
    
    if tmp_model:
        # 1. Total Parameters
        params = tmp_model.count_params()
        
        # 2. Model Size (MB)
        size_mb = os.path.getsize(f"{m_name}/{m_name}_best.keras") / (1024 * 1024) if os.path.exists(f"{m_name}/{m_name}_best.keras") else 0
        
        # 3. Inference Time & FPS
        # Warmup
        dummy_input = np.random.random((16, 224, 224, 3)).astype(np.float32)
        _ = tmp_model.predict(dummy_input, verbose=0)
        
        # Measure
        start_time = time.time()
        for i in range(10):
             _ = tmp_model.predict(dummy_input, verbose=0)
        end_time = time.time()
        
        time_per_batch = (end_time - start_time) / 10
        fps = 16 / time_per_batch
        
        hardware_stats.append({
            'Model': m_name,
            'Params (Millions)': params / 1e6,
            'Model Size (MB)': size_mb,
            'Inference FPS (Batch=16)': fps
        })

df_hw = pd.DataFrame(hardware_stats).set_index('Model')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot Params
df_hw['Params (Millions)'].plot(kind='bar', ax=axes[0], color='lightcoral', edgecolor='k')
axes[0].set_title('Number of Parameters (M)')
axes[0].set_ylabel('Millions')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot Size
df_hw['Model Size (MB)'].plot(kind='bar', ax=axes[1], color='lightskyblue', edgecolor='k')
axes[1].set_title('Model Size on Disk (MB)')
axes[1].set_ylabel('MB')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot FPS
df_hw['Inference FPS (Batch=16)'].plot(kind='bar', ax=axes[2], color='lightgreen', edgecolor='k')
axes[2].set_title('Inference Speed (FPS)')
axes[2].set_ylabel('Frames Per Second')
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("Hardware_and_Inference_Stats_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()"""))

cells.append(md("### 10.6 Phân tích Grad-CAM (Khả năng Diễn giải Model)"))
cells.append(code("""# Step 10.6 — Grad-CAM for interpretability
def make_gradcam(model, img_array):
    last_conv = None
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    def _is_4d_conv(layer):
        try:
            shape = layer.output_shape
            if isinstance(shape, list): shape = shape[0]  # Keras 3 multi-output compat
            return isinstance(layer, conv_types) and len(shape) == 4
        except Exception:
            return False
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sl in reversed(layer.layers):
                if _is_4d_conv(sl):
                    last_conv = sl; break
            if last_conv: break
        elif _is_4d_conv(layer):
            last_conv = layer; break
    if last_conv is None: return None
    try:
        grad_model = tf.keras.models.Model(inputs=model.inputs,
                                           outputs=[last_conv.output, model.output])
    except Exception:
        return None  # disconnected graph for nested sub-models
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_idx = tf.argmax(preds[0])
        class_ch = preds[:, pred_idx]
    grads = tape.gradient(class_ch, conv_out)
    if grads is None: return None
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

fig, axes = plt.subplots(3,3,figsize=(15,15))
fig.suptitle(f'Grad-CAM — {best_model_name}', fontsize=16, fontweight='bold')
idxs = np.random.choice(len(X_test), 3, replace=False)
for i, idx in enumerate(idxs):
    img    = load_img(X_test[idx], target_size=(IMG_SIZE,IMG_SIZE))
    # Model input must match training pixel range [0,255]
    arr_255 = img_to_array(img).astype(np.float32)
    tensor = np.expand_dims(arr_255, 0)
    pred   = best_model.predict(tensor, verbose=0)
    pred_cls = CLASS_NAMES[np.argmax(pred)]
    true_cls = CLASS_NAMES[y_true_test_global[idx]]
    conf     = float(np.max(pred))
    heatmap  = make_gradcam(best_model, tensor)
    color    = 'green' if pred_cls == true_cls else 'red'
    # For visualization only, normalize to [0,1]
    arr_vis = np.clip(arr_255 / 255.0, 0.0, 1.0)
    axes[i,0].imshow(arr_vis); axes[i,0].set_title(f'True: {true_cls}', fontweight='bold'); axes[i,0].axis('off')
    if heatmap is not None:
        hm = tf.image.resize(heatmap[...,np.newaxis], (IMG_SIZE,IMG_SIZE)).numpy()[:,:,0]
        axes[i,1].imshow(hm, cmap='jet'); axes[i,1].set_title('Grad-CAM Heatmap'); axes[i,1].axis('off')
        axes[i,2].imshow(arr_vis); axes[i,2].imshow(hm, cmap='jet', alpha=0.4)
        axes[i,2].set_title(f'Pred: {pred_cls} ({conf:.1%})', fontweight='bold', color=color)
        axes[i,2].axis('off')
    else:
        for j in [1,2]: axes[i,j].text(0.5,0.5,'N/A',ha='center',va='center'); axes[i,j].axis('off')
plt.tight_layout(); plt.show()"""))

cells.append(md("### 10.7 Tổng kết Báo cáo"))
cells.append(code("""# Step 10.7 — Tổng kết dự án
print("="*70)
print("🌾 DỰ ĐOÁN BỆNH LÁ LÚA — TỔNG KẾT DỰ ÁN")
print("="*70)
print(f"Dữ liệu       : {len(df)} ảnh sạch | {NUM_CLASSES} phân lớp")
print(f"Phần cứng     : GPU T4x2 | {strategy.num_replicas_in_sync} replica(s)")
print(f"Kích thước nén: {BATCH_SIZE}")
print(f"Mô hình thử    : ConvNeXtSmall, MobileNetV3Large, EfficientNetV2S")
print(f"Mô hình tốt    : {best_model_name}")
print(f"Khóa huấn luyện: 3-Phase Transfer Learning (Mở băng tuần tự)")
print(f"  Giai đoạn 1  : Base đóng băng (Frozen), LR=1e-3, {EPOCHS_P1} epochs")
print(f"  Giai đoạn 2  : Rã đông top 30%, LR=1e-4, {EPOCHS_P2} epochs")
print(f"  Giai đoạn 3  : Tinh chỉnh Toàn bộ (Full fine-tune), LR=1e-5, {EPOCHS_P3} epochs")
print()
print("Kết quả đánh giá trên Tập Kiểm Thử (Test Set):")
for k,v in test_metrics.items(): print(f"  {k:<12}: {v:.4f} ({v:.1%})")
print()
print("Công nghệ chính yếu:")
for t in [
    "Các mô hình tiên tiến chuẩn SOTA (ConvNeXtSmall, MobileNetV3Large, EfficientNetV2S)",
    "Rã đông 3 giai đoạn chậm dần (warmup → partial → full fine-tune)",
    "Hỗ trợ phân tán MirroredStrategy (2 lõi GPU T4 đồng bộ)",
    "Tăng cường dữ liệu bằng TF cục bộ (chạy GPU mượt, không lỗi backend)",
    "Tính trọng số lớp tự động để giải quyết bất cân bằng dữ liệu",
    "EarlyStopping + ReduceLROnPlateau (Chống quá khớp, điều khiển LR thông minh)",
    "Kỹ thuật Dropout(0.3) tại Head_Model (classifier)",
    "Hiểu quy trình dự báo mịt mờ với Grad-CAM Interpretability",
    "Cách ly bảo vệ tập Test 100% (chưa từng dùng đến Bước 10)",
]: print(f"  ✓ {t}")
print()
target = 0.90
if test_metrics['Accuracy'] >= target:
    print(f"✅ Đạt mục tiêu đề xuất! Accuracy = {test_metrics['Accuracy']:.1%} >= {target:.0%}")
else:
    print(f"⚠ Chưa đạt mục tiêu Accuracy {target:.0%}. Sẽ cần thêm dataset và tuning siêu cấp.")
print("="*70)"""))

# ─────────────────────────────────────────────────────
# BUILD NOTEBOOK
# ─────────────────────────────────────────────────────
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True
        }
    },
    "cells": cells
}

OUTPUT = r"d:\DataDownloaded\Study\MonHoc\Do_An\deeplearning\CK2\rice_disease_prediction_t4x2.ipynb"
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

md_cells   = sum(1 for c in cells if c['cell_type']=='markdown')
code_cells = sum(1 for c in cells if c['cell_type']=='code')
print(f"✓ Notebook GPU đã được tạo: {OUTPUT}")
print(f"  Tổng cells  : {len(cells)}")
print(f"  Markdown    : {md_cells}")
print(f"  Code        : {code_cells}")

