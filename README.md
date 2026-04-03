# 🌾 Rice Leaf Disease Classification — Deep Learning Pipeline

> **Phân loại Bệnh Lá Lúa sử dụng Deep Learning (10-Step ML Pipeline)**  
> IEEE-format paper | 4 CNN models | Ensemble Soft-Voting | Grad-CAM Explainability

---

## 📌 Tổng quan

Hệ thống phân loại **7 loại bệnh lá lúa** sử dụng pipeline học sâu 10 bước:
- Thu thập & lọc dữ liệu từ **4 bộ dataset Kaggle**
- Tiền xử lý nâng cao: **CLAHE, CutMix, MixUp, GrabCut**
- Cân bằng lớp bằng **Conditional DCGAN** tự code
- Huấn luyện **4 mô hình CNN** với chiến lược 3-Phase fine-tuning
- Đánh giá bằng **6 chỉ số**: Accuracy, Precision, Recall, F1-Macro, F1-Weighted, Kappa
- Kết hợp dự đoán bằng **Ensemble Soft-Voting**
- Giải thích bằng **Grad-CAM** (Explainable AI)

---

## 🤖 4 Mô hình CNN

| Model | Năm | Params | FPS (T4 GPU) | Test Accuracy |
|-------|-----|--------|--------------|---------------|
| MobileNetV3Large | 2019 | ~3.0M | **47 FPS** | 85.7% |
| EfficientNetV2S | 2021 | ~21.5M | 42 FPS | **88.8%** |
| ConvNeXtSmall | 2022 | ~49.0M | 35 FPS | 88.6% |
| **Proposed_ConvNeXtTiny_SE** | **2026** | ~29.0M | 39 FPS | 87.6% |

> Proposed model = ConvNeXt-Tiny backbone + **SE-Block (channel attention)** + **Focal Loss** (γ=2.0)

---

## 📁 Cấu trúc dự án

```
CK2/
├── 📓 rice_disease_prediction_t4x2.ipynb   ← Notebook chính (GPU T4×2)
├── 📓 rice_disease_10steps.ipynb           ← Notebook demo 10 bước
├── 🐍 gradcam_visualize.py                 ← Script tạo Grad-CAM 4 model
├── 🐍 generate_paper_15pages.py            ← Script tạo bài báo IEEE
├── 🐍 generate_notebook_gpu.py             ← Script generate notebook GPU
├── 🐍 patch_algorithm.py                   ← Patch cho các bước thuật toán
├── 🐍 gen_kich_ban_word.py                 ← Tạo kịch bản thuyết trình Word
├── 📄 BaoCao_BenhLaLua_IEEE_Final.docx     ← Bài báo IEEE hoàn chỉnh
├── 📄 Kich_Ban_Bao_Cao_HoanChinh_V4.docx   ← Kịch bản thuyết trình
├── 📊 picture_results/                     ← Kết quả biểu đồ
├── 📝 bao_cao_danh_gia.md                  ← Đánh giá chi tiết
└── 🚫 model/                               ← Chứa file .keras (gitignored)
```

---

## 🚀 Cách chạy

### 1. Huấn luyện trên Kaggle (GPU T4×2)
```bash
# Upload rice_disease_prediction_t4x2.ipynb lên Kaggle
# Enable GPU T4×2 accelerator
# Run All Cells
```

### 2. Tạo Grad-CAM locally
```bash
pip install tensorflow keras matplotlib numpy Pillow
python gradcam_visualize.py
# Output: GradCAM_4Models_Compare.png
```

### 3. Tạo bài báo IEEE
```bash
pip install python-docx
python generate_paper_15pages.py
# Output: BaoCao_BenhLaLua_IEEE_Final.docx
```

### 4. Tạo kịch bản thuyết trình Word
```bash
python gen_kich_ban_word.py
# Output: Kich_Ban_Bao_Cao_HoanChinh_V4.docx
```

---

## 📊 Kết quả Test Set

| Chỉ số | EfficientNetV2S | ConvNeXtSmall | Proposed_SE | MobileNetV3L |
|--------|----------------|---------------|-------------|--------------|
| Accuracy | **0.888** | 0.886 | 0.876 | 0.857 |
| F1-Macro | **0.878** | 0.873 | 0.861 | 0.860 |
| Kappa | **0.866** | 0.862 | 0.852 | 0.829 |

> ✅ **Tất cả 4 model đều đạt Kappa > 0.81** (Almost Perfect Agreement)

---

## 🛠️ Yêu cầu
- Python ≥ 3.9
- TensorFlow ≥ 2.12 / Keras ≥ 3.0
- python-docx, matplotlib, numpy, Pillow, opencv-python

---

## 📜 Pipeline 10 bước

| Step | Tên | Kỹ thuật chính |
|------|-----|----------------|
| 1 | Problem Understanding | Multi-class (7 lớp), CNN-only |
| 2 | Data Collection | 4 Kaggle datasets, MD5 dedup |
| 3 | EDA | Boxplot RGB, phân bố lớp |
| 4 | Feature Engineering | CLAHE, CutMix, MixUp, GrabCut |
| 5 | Data Balancing | Conditional DCGAN (5000 steps) |
| 6 | Training | 3-Phase fine-tune, Cosine LR |
| 7 | Knowledge Distillation | Teacher→Student (T=4.0, α=0.3) |
| 8 | Phase-3 Full Fine-tune | LR=1e-5, toàn bộ layers |
| 9 | Ensemble | Soft-Voting (4 models) |
| 10 | Evaluation | Grad-CAM, ROC-AUC, Kappa |

---

## ⚠️ Lưu ý

- File model `.keras` (~400MB) **không được commit** lên GitHub (xem `.gitignore`).
- Dataset **không được commit** — tải từ Kaggle hoặc liên hệ nhóm.
- Ảnh GAN sinh ra chỉ dùng cho tập **Train**, tuyệt đối không dùng cho Val/Test.

---

## 📦 Dataset (Kaggle)

Dự án sử dụng **4 bộ dataset** gộp lại từ Kaggle. Tải về và đặt vào thư mục `datasets/`:

| # | Dataset | Link |
|---|---------|------|
| 1 | Rice Disease (thaonguyen0712) | [🔗 Tải về](https://www.kaggle.com/datasets/thaonguyen0712/rice-disease) |
| 2 | Rice Disease (nurnob101) | [🔗 Tải về](https://www.kaggle.com/datasets/nurnob101/rice-disease) |
| 3 | Rice Disease (jonathanrjpereira) | [🔗 Tải về](https://www.kaggle.com/datasets/jonathanrjpereira/rice-disease) |
| 4 | Rice Disease Dataset (anshulm257) | [🔗 Tải về](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset) |

> 💡 **Tải nhanh bằng Kaggle CLI:**
> ```bash
> pip install kaggle
> kaggle datasets download thaonguyen0712/rice-disease -p datasets/
> kaggle datasets download nurnob101/rice-disease -p datasets/
> kaggle datasets download jonathanrjpereira/rice-disease -p datasets/
> kaggle datasets download anshulm257/rice-disease-dataset -p datasets/
> ```
