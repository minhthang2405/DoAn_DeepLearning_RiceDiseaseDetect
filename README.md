# 🌾 Rice Leaf Disease Classification — Deep Learning

> **Phân loại Bệnh Lá Lúa sử dụng Deep Learning (10-Step ML Pipeline)**  
> 4 CNN Models · Ensemble Soft-Voting · Grad-CAM Explainability · GPU T4×2

---

## 📌 Tổng quan

Hệ thống phân loại **7 loại bệnh lá lúa** từ ảnh chụp, sử dụng pipeline học sâu 10 bước hoàn chỉnh:

1. Thu thập & lọc dữ liệu từ **4 bộ dataset Kaggle** (MD5 dedup)
2. Phân tích khám phá dữ liệu (EDA) — BoxPlot RGB, phân bố lớp
3. Tiền xử lý nâng cao: **CLAHE, CutMix, MixUp**
4. Cân bằng lớp thiểu số bằng **Conditional DCGAN**
5. Huấn luyện **4 mô hình CNN** với chiến lược **3-Phase fine-tuning** trên GPU T4×2
6. Knowledge Distillation (Teacher → Student)
7. Đánh giá bằng **6 chỉ số**: Accuracy, Precision, Recall, F1-Macro, F1-Weighted, Kappa
8. Kết hợp dự đoán bằng **Ensemble Soft-Voting** (4 models)
9. Giải thích quyết định bằng **Grad-CAM** (Explainable AI)

---

## 🤖 4 Mô hình CNN

| Model | Năm | Params | FPS (T4) | Test Accuracy |
|-------|-----|--------|----------|---------------|
| MobileNetV3Large | 2019 (Google) | ~3.0M | **47** | 85.7% |
| EfficientNetV2S | 2021 (Google) | ~21.5M | 42 | **88.8%** |
| ConvNeXtSmall | 2022 (Meta) | ~49.0M | 35 | 88.6% |
| **Proposed_ConvNeXtTiny_SE** | **2026 (Nhóm)** | ~29.0M | 39 | 87.6% |

> **Proposed model** = ConvNeXt-Tiny backbone + SE-Block (channel attention) + Focal Loss (γ=2.0)

---

## 📊 Kết quả Test Set

| Chỉ số | EfficientNetV2S | ConvNeXtSmall | Proposed_SE | MobileNetV3L |
|--------|----------------|---------------|-------------|--------------|
| Accuracy | **0.888** | 0.886 | 0.876 | 0.857 |
| F1-Macro | **0.878** | 0.873 | 0.861 | 0.860 |
| Kappa | **0.866** | 0.862 | 0.852 | 0.829 |

> ✅ Tất cả 4 model đều đạt **Kappa > 0.81** (Almost Perfect Agreement)

---

## 🔬 Kết quả trực quan (Grad-CAM)

Xem thư mục [`results/`](results/) để xem toàn bộ hình kết quả:
- Grad-CAM heatmap so sánh 4 mô hình
- Training curves, Confusion Matrix
- ROC-AUC, biểu đồ đánh giá

---

## 📁 Cấu trúc Repository

```
├── 📓 rice_disease_prediction_t4x2.ipynb   ← Notebook huấn luyện chính (GPU T4×2)
├── 📊 results/                             ← Hình kết quả (Grad-CAM, biểu đồ, ...)
├── 📄 README.md                            ← File này
└── 🚫 .gitignore                           ← Ẩn model, dataset, script phụ
```

> ⚠️ File model `.keras` (~1GB tổng), dataset, script phụ trợ **không được push** lên GitHub.

---

## 📦 Dataset (Kaggle)

Dự án sử dụng **4 bộ dataset** gộp lại:

| # | Dataset | Link |
|---|---------|------|
| 1 | Rice Disease (thaonguyen0712) | [🔗 Kaggle](https://www.kaggle.com/datasets/thaonguyen0712/rice-disease) |
| 2 | Rice Disease (nurnob101) | [🔗 Kaggle](https://www.kaggle.com/datasets/nurnob101/rice-disease) |
| 3 | Rice Disease (jonathanrjpereira) | [🔗 Kaggle](https://www.kaggle.com/datasets/jonathanrjpereira/rice-disease) |
| 4 | Rice Disease Dataset (anshulm257) | [🔗 Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset) |

```bash
# Tải nhanh bằng Kaggle CLI:
pip install kaggle
kaggle datasets download thaonguyen0712/rice-disease
kaggle datasets download nurnob101/rice-disease
kaggle datasets download jonathanrjpereira/rice-disease
kaggle datasets download anshulm257/rice-disease-dataset
```

---

## 📜 Pipeline 10 bước

| Step | Tên | Kỹ thuật chính |
|------|-----|----------------|
| 1 | Problem Understanding | Multi-class (7 lớp), CNN-only |
| 2 | Data Collection | 4 Kaggle datasets, MD5 dedup |
| 3 | EDA | BoxPlot RGB, phân bố lớp |
| 4 | Feature Engineering | CLAHE, CutMix, MixUp |
| 5 | Data Balancing | Conditional DCGAN |
| 6 | Training | 3-Phase fine-tune, Cosine LR, Focal Loss |
| 7 | Knowledge Distillation | Teacher → Student (T=4.0, α=0.3) |
| 8 | Full Fine-tune | LR=1e-5, unfreeze 100% layers |
| 9 | Ensemble | Soft-Voting (trung bình xác suất 4 models) |
| 10 | Evaluation | Grad-CAM, ROC-AUC, Confusion Matrix, Kappa |

---

## 🛠️ Yêu cầu

- Python ≥ 3.9
- TensorFlow ≥ 2.12 / Keras ≥ 3.0
- Kaggle GPU T4×2 (huấn luyện)
