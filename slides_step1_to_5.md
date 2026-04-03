# NỘI DUNG THUYẾT TRÌNH — STEP 1 ĐẾN STEP 10
## Đề tài: Phân loại Bệnh Lá Lúa bằng Deep Learning

---

# SLIDE 1: TRANG BÌA

**Đề tài:** Phân loại Bệnh Lá Lúa sử dụng các mô hình Deep Learning tiên tiến

*(Rice Leaf Disease Classification using Advanced Deep Learning Models)*

- **Môn học:** Đồ Án Deep Learning
- **Phần cứng:** GPU T4×2 (MirroredStrategy) — Kaggle
- **Models:** ConvNeXtSmall, MobileNetV3Large, EfficientNetV2S, Proposed_ConvNeXtTiny_SE

---

# SLIDE 2: TỔNG QUAN PIPELINE (10 Bước)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: Hiểu bài toán (Problem Understanding)                       │
│ Step 2: Hiểu Dữ liệu (Data Understanding)                          │
│ Step 3: Hiểu Đặc trưng (Feature Understanding) — EDA                │
│ Step 4: Kỹ thuật Đặc trưng (Feature Engineering)                    │
│ Step 5: Phân chia Tập dữ liệu & Cân bằng dữ liệu (Partition)       │
│ Step 6: Mô hình hóa (Data Modelling)                                │
│ Step 7: Training 3-Phase (Gradual Unfreezing)                       │
│ Step 8: Đánh giá mô hình (Evaluation)                               │
│ Step 9: Inference Pipeline                                           │
│ Step 10: Kết luận & Grad-CAM                                        │
└──────────────────────────────────────────────────────────────────────┘
```

> Slide thuyết trình hôm nay tập trung vào **Step 1 → Step 5** (Chuẩn bị dữ liệu)

---

# SLIDE 3: STEP 1 — HIỂU BÀI TOÁN (Problem Understanding)

## 1.1 Xác định bài toán

| Tiêu chí | Chi tiết |
|---|---|
| **Loại bài toán** | Phân loại Đa lớp (Multi-class Image Classification) |
| **Đặc trưng (Feature X)** | Ảnh RGB lá lúa, kích thước 224×224×3 pixel |
| **Mục tiêu (Target y)** | Tên loại bệnh — 7 lớp |
| **Đầu vào** | Ảnh chụp lá lúa ngoài đồng |
| **Đầu ra** | Nhãn dự đoán: 1 trong 7 loại bệnh/trạng thái |

## 1.2 Danh sách 7 lớp phân loại

| STT | Tên lớp (Tiếng Anh) | Tên lớp (Tiếng Việt) |
|:---:|---|---|
| 0 | Bacterial_Leaf_Blight | Bạc lá vi khuẩn |
| 1 | Brown_Spot | Đốm nâu |
| 2 | Healthy | Khỏe mạnh |
| 3 | Hispa | Sâu cuốn lá nhỏ |
| 4 | Leaf_Blast | Đạo ôn lá |
| 5 | Leaf_Scald | Bỏng lá |
| 6 | Sheath_Blight | Khô vằn |

## 1.3 Tại sao chọn 4 mô hình này?

| Mô hình | Vai trò | Đặc điểm nổi bật | Tham số |
|---|---|---|---|
| **MobileNetV3Large** | Đại diện nhóm siêu nhẹ | Kiểm chứng khi chạy trên thiết bị di động | ~4.2M |
| **EfficientNetV2S** | Cân bằng tốc độ & chính xác | SOTA 2021, tối ưu hóa parameter | ~21M |
| **ConvNeXtSmall** | Sức mạnh CNN hiện đại thuần túy | SOTA 2022, cạnh tranh Transformer | ~50M |
| **Proposed_ConvNeXtTiny_SE** | Mô hình đề xuất riêng | ConvNeXtTiny + SE-Block + Focal Loss | ~29M |

> **Điểm nhấn:** Mô hình đề xuất (Proposed) tích hợp **SE-Block** (Squeeze-and-Excitation) để tăng khả năng tập trung vào các kênh đặc trưng quan trọng, kết hợp **Focal Loss** để xử lý lớp bị mất cân bằng.

---

# SLIDE 4: STEP 2 — HIỂU DỮ LIỆU (Data Understanding)

## 2.1 Nguồn dữ liệu — Kết hợp 4 bộ từ Kaggle

| # | Dataset Kaggle | Tác giả | Ghi chú |
|:---:|---|---|---|
| 1 | `anshulm257/rice-disease-dataset` | Anshul M. | ~6 lớp |
| 2 | `jonathanrjpereira/rice-disease` | Jonathan R. | 4 lớp |
| 3 | `nurnob101/rice-disease` | Nurnob | ~4 lớp |
| 4 | `thaonguyen0712/rice-disease` | Thao Nguyen | ~4 lớp |

> **Tổng cộng:** Kết hợp 4 nguồn → chuẩn hóa về **7 lớp thống nhất**, tạo bộ dữ liệu phong phú và đa dạng hơn.

## 2.2 Chuẩn hóa nhãn (Label Normalization)

- Nhiều bộ dữ liệu đặt tên lớp khác nhau → Cần **ánh xạ (mapping)** về tên chuẩn
- Ví dụ mapping (tiếng Anh):

```
"bacterial_blight", "bacterialblight", "blight" → Bacterial_Leaf_Blight
"blast", "rice_blast", "neck_blast"             → Leaf_Blast
"normal"                                        → Healthy
```

- Mapping **tiếng Việt** (dataset `thaonguyen0712/rice-disease`):

| Tên folder (Tiếng Việt) | Ý nghĩa | → Lớp chuẩn |
|---|---|---|
| `bo_gai` | Bọ gai | Hispa |
| `chay_bia_la` | Cháy bìa lá | Bacterial_Leaf_Blight |
| `dao_on` | Đạo ôn | Leaf_Blast |
| `dom_nau` | Đốm nâu | Brown_Spot |
| `vang_la` | Vàng lá | Leaf_Scald |

## 2.3 Kiểm tra chất lượng dữ liệu (Data Quality Check)

Kiểm tra **3 vấn đề** trước khi đưa vào huấn luyện:

| Vấn đề | Cách xử lý | Phương pháp |
|---|---|---|
| **Ảnh hỏng (Corrupt)** | Xóa bỏ | `PIL.Image.verify()` |
| **Ảnh quá nhỏ (< 32px)** | Xóa bỏ | Kiểm tra `width × height` |
| **Ảnh trùng lặp (Duplicate)** | Xóa bỏ | So sánh hash **MD5** |

## 2.4 Kiểm tra mất cân bằng (Imbalance Check)

- Tính **Imbalance Ratio** = `max(count) / min(count)`
- Nếu ratio > 3 → ⚠ **Mất cân bằng** → Sẽ dùng **class weights** + **DCGAN synthesis** ở bước sau

---

# SLIDE 5: STEP 3 — HIỂU ĐẶC TRƯNG (Feature Understanding / EDA)

## 3.1 Phân tích Đơn biến (Univariate Analysis)

### Phân phối số lượng ảnh mỗi lớp
- **Biểu đồ cột (Bar Chart):** Hiển thị số lượng ảnh mỗi lớp
- **Biểu đồ tròn (Pie Chart):** Hiển thị tỷ lệ phần trăm mỗi lớp
- → Nhận diện lớp nào thiểu số, lớp nào đa số

### Phân phối kích thước ảnh
- **Biểu đồ Histogram** cho Width và Height
- **Biểu đồ Scatter** Width vs Height
- Xác định median kích thước để chọn `IMG_SIZE = 224` hợp lý

## 3.2 Phân tích Nhị biến (Bivariate Analysis)

### Ảnh mẫu theo từng lớp
- Hiển thị **5 ảnh mẫu** cho mỗi lớp bệnh
- Giúp **trực quan hóa** sự khác biệt giữa các loại bệnh
- Nhận xét: Một số bệnh có triệu chứng tương tự (ví dụ: Brown Spot vs Leaf Blast) → thách thức cho mô hình

## 3.3 Phân tích Đa biến (Multivariate Analysis)

### Phân tích kênh màu RGB theo lớp
- **Boxplot** cho từng kênh R, G, B theo mỗi lớp bệnh
- **Mục đích:** Kiểm tra xem kênh màu nào có sự khác biệt rõ ràng giữa các lớp?
- **Nhận xét kỳ vọng:**
  - Lá **Healthy** thường có kênh **G (Green)** cao hơn
  - Lá bị **Brown Spot** có kênh **R (Red)** nhỉnh hơn
  - → CNN có thể tận dụng các đặc trưng màu này

---

# SLIDE 6: STEP 4 — KỸ THUẬT ĐẶC TRƯNG (Feature Engineering)

## 4.1 Mã hóa nhãn (Label Encoding)

- Chuyển nhãn chuỗi → số nguyên:
```
Bacterial_Leaf_Blight → 0
Brown_Spot            → 1
Healthy               → 2
Hispa                 → 3
Leaf_Blast            → 4
Leaf_Scald            → 5
Sheath_Blight         → 6
```

## 4.2 Tăng cường dữ liệu (Data Augmentation)

### Augmentation cơ bản bằng TensorFlow (chạy trực tiếp trên GPU):

| Kỹ thuật | Tham số | Mục đích |
|---|---|---|
| `random_flip_left_right` | — | Lật ngang ngẫu nhiên |
| `random_flip_up_down` | — | Lật dọc ngẫu nhiên |
| `random_brightness` | ±20% | Thay đổi độ sáng |
| `random_contrast` | [0.8, 1.2] | Thay đổi độ tương phản |
| `random_saturation` | [0.8, 1.2] | Thay đổi độ bão hòa |
| `random_hue` | ±0.05 | Thay đổi tông màu |

### Augmentation nâng cao (áp dụng trên batch):

| Kỹ thuật | Mô tả | Xác suất |
|---|---|---|
| **CutMix** | Cắt 1 vùng từ ảnh khác dán vào → trộn nhãn theo tỷ lệ diện tích | 50% |
| **MixUp** | Trộn 2 ảnh theo hệ số λ ~ Beta(0.2, 0.2) → trộn nhãn tương ứng | 50% |

> **Tại sao CutMix + MixUp?** Giúp mô hình không quá phụ thuộc vào một vùng cục bộ, tăng khả năng tổng quát hóa (generalization), giảm overfitting.

## 4.3 CLAHE — Cân bằng Histogram Thích ứng

```
Ảnh gốc → Chuyển sang LAB → Áp CLAHE lên kênh L → Chuyển lại RGB
```

| Bước | Chi tiết |
|---|---|
| 1. Chuyển không gian màu | RGB → LAB |
| 2. Tách kênh L (Lightness) | Kênh chứa thông tin độ sáng |
| 3. CLAHE | `clipLimit=2.0`, `tileGridSize=(8,8)` |
| 4. Kết quả | Vết bệnh trở nên **rõ ràng** và **tương phản cao hơn** |

> **Tại sao CLAHE?** Ảnh ngoài đồng thường bị lệch sáng, CLAHE giúp cân bằng cục bộ → vết bệnh nổi bật hơn cho CNN trích xuất đặc trưng.

## 4.4 Tách nền (Background Segmentation)

- **Mục tiêu:** Tạo thêm biến thể "leaf-only" để so sánh hiệu năng
- **Phương pháp 1:** SAM (Segment Anything Model) — nếu có weights
- **Phương pháp 2 (Fallback):** GrabCut + Green prior heuristic
- **Quy trình Fallback:**

```
Ảnh RGB → GrabCut (rect prompt) → Refine bằng HSV green range → Morphology Close → Mask
```

---

# SLIDE 7: STEP 5 — PHÂN CHIA TẬP DỮ LIỆU (Dataset Partition)

## 5.0 Chia tập Train / Val / Test

### Chiến lược chia:

| Tập | Tỷ lệ | Mục đích |
|---|---|---|
| **Train** | 70% | Huấn luyện mô hình |
| **Validation** | 15% | Theo dõi overfitting, early stopping |
| **Test** | 15% | Đánh giá cuối cùng (KHÔNG chạm cho đến Step 10) |

- Sử dụng **Stratified Split** (phân tầng) → Giữ nguyên tỷ lệ phân phối gốc ở mỗi tập
- Test set được **"khóa"** hoàn toàn cho đến bước đánh giá cuối

## 5.1 Data Synthesis — Conditional DCGAN (Kiến trúc CNN)

### Tại sao cần sinh ảnh?
- Tập **Train** ban đầu chưa đạt mốc 10,000 ảnh (tính riêng Train, không tính Val/Test)
- Một số lớp có số lượng ít hơn → cần **cân bằng**
- **Chỉ sinh vào tập Train** để tránh Data Leakage tuyệt đối

### Kiến trúc Conditional DCGAN (CNN):

> **Lưu ý quan trọng**: DCGAN thuộc họ kiến trúc **CNN** (Convolutional Neural Network), KHÔNG phải RNN. Generator dùng Conv2DTranspose (Transposed Convolution), Discriminator dùng Conv2D thông thường.

#### Generator (Mạng sinh):
```
Noise (128 chiều) + Class Embedding
    ↓ Dense(4×4×512)
    ↓ Conv2DTranspose(256, stride=2) → 8×8
    ↓ Conv2DTranspose(128, stride=2) → 16×16
    ↓ Conv2DTranspose(64,  stride=2) → 32×32
    ↓ Conv2DTranspose(3,   stride=2) → 64×64 (tanh)
```
- Sử dụng **Transposed Convolution** (phép tích chập chuyển vị) để **tăng kích thước** từ noise → ảnh
- **BatchNorm + ReLU** giữa các lớp (theo đúng chuẩn paper DCGAN gốc)
- Output: ảnh 64×64×3 trong khoảng [-1, 1]

#### Discriminator (Mạng phân biệt):
```
Ảnh (64×64×3) + Class Label Map
    ↓ Conv2D(64,  stride=2) → 32×32
    ↓ Conv2D(128, stride=2) → 16×16
    ↓ Conv2D(256, stride=2) → 8×8
    ↓ Conv2D(512, stride=2) → 4×4
    ↓ Flatten → Dense(1) → logit thật/giả
```
- Sử dụng **CNN (Conv2D stride=2)** để **giảm kích thước** ảnh → phân loại
- **LeakyReLU + Dropout(0.3)** giữa các lớp

#### Hyperparameters:

| Tham số | Giá trị |
|---|---|
| Noise dim | 128 |
| Image size | 64×64 |
| Batch size | 64 |
| Steps | **5000** (tăng từ 2000 để ảnh chất lượng cao hơn) |
| Learning rate | 2e-4 |
| Optimizer | Adam (β₁=0.5) |
| Loss | BinaryCrossentropy (from_logits) |
| Target | Train set đạt 10,000 ảnh |

### Quy trình sinh ảnh:
1. Tính `shortage = max(0, TARGET_TRAIN - len(X_train_list))` — dựa trên **kích thước tập Train** thực tế
2. Tính số ảnh cần bù cho từng lớp thiểu số (target = lớp nhiều nhất)
3. Train GAN 5000 steps trên tập Train
4. Sinh ảnh ảo theo đúng số lượng cần bù
5. Thêm ảnh sinh vào tập Train
6. Tính **Class Weights** (balanced) để xử lý phần imbalance còn lại

## 5.2 Kiểm tra phân phối sau chia

- Biểu đồ cột so sánh phân phối 3 tập Train / Val / Test
- Đảm bảo **Stratified Split** giữ đúng tỷ lệ

## 5.3 Xây dựng Pipeline `tf.data` siêu tốc

```python
tf.data.Dataset
    → shuffle()
    → map(load_and_preprocess)  # đọc ảnh, resize 224×224
    → map(augment)              # chỉ Train: flip, brightness, contrast...
    → batch()
    → map(CutMix/MixUp)        # chỉ Train: trộn ảnh nâng cao
    → prefetch(AUTOTUNE)        # nạp trước batch tiếp theo trong lúc GPU xử lý
```

| Đặc điểm | Giải thích |
|---|---|
| **Không chia 255** | Các model dùng `include_preprocessing=True` → tự chuẩn hóa bên trong |
| **AUTOTUNE** | TensorFlow tự tối ưu số luồng song song |
| **drop_remainder** | Đảm bảo batch size đồng nhất cho GPU |

## 5.4 Tạo Dataset phân đoạn (Segmented)

- Chạy segmentation trên 3 tập (Train/Val/Test)
- Tạo song song 2 bộ dataset: **ảnh gốc** vs **ảnh tách nền**
- So sánh hiệu năng (ablation study) ở Step đánh giá

---

# SLIDE 8: TÓM TẮT STEP 1-5 & TỔNG QUAN STEP 6-10

> **Lưu ý kiến trúc**: Toàn bộ dự án sử dụng **CNN (Convolutional Neural Network)** — KHÔNG sử dụng RNN, LSTM hay GRU. CNN phù hợp hoàn hảo cho bài toán phân loại ảnh.

| Step | Tên | Công việc chính | Kiến trúc | Kết quả |
|:---:|---|---|:---:|---|
| 1 | Problem Understanding | Xác định bài toán, 7 lớp, 4 mô hình | — | Multi-class Classification |
| 2 | Data Understanding | Gộp 4 dataset, loại ảnh hỏng/trùng | — | Bộ dữ liệu sạch, chuẩn hóa |
| 3 | Feature Understanding | EDA: univariate, bivariate, multivariate | — | Hiểu rõ phân phối & đặc trưng màu |
| 4 | Feature Engineering | Label Encoding, Augmentation, CLAHE, Tách nền | — | Tăng cường & cải thiện đặc trưng |
| 5 | Dataset Partition | Train/Val/Test (70-15-15) + DCGAN synthesis | **CNN** (DCGAN) | >10,000 ảnh, pipeline `tf.data` |
| 6 | Data Modelling | 4 mô hình Transfer Learning + SE-Block + Focal Loss | **CNN** ×4 | ConvNeXtSmall, MobileNetV3, EfficientNetV2S, Proposed |
| 7 | Data Evaluation | So sánh 4 mô hình, Confusion Matrix, Knowledge Distillation | **CNN** | Best Model chọn theo F1-Macro |
| 8 | Hyper-parameter Tuning | Phase 3 Full Fine-tune + Ablation Study | **CNN** | Cải thiện thêm Accuracy |
| 9 | Inference Pipeline | Pipeline dự đoán + Ensemble Soft-voting | **CNN** ×4 | RiceDiseasePipeline class |
| 10 | Conclusion | Test evaluation, ROC-AUC, Grad-CAM | **CNN** | Đánh giá cuối cùng trung thực |

> **Kết luận:** Toàn bộ 10 bước sử dụng kiến trúc CNN. Dữ liệu đã được chuẩn bị kỹ lưỡng và huấn luyện qua 3-Phase Transfer Learning.

---

# SLIDE 9: STEP 6 — MÔ HÌNH HÓA DỮ LIỆU (Data Modelling)

## 6.1 Tổng quan 4 mô hình CNN

> Tất cả 4 mô hình đều là **CNN thuần túy**, không có thành phần RNN nào.

| Mô hình | Kiến trúc CNN | Đặc điểm | Tham số |
|---|---|---|---|
| **ConvNeXtSmall** | DepthwiseConv + LayerNorm + GELU | SOTA 2022, cạnh tranh Vision Transformer | ~50M |
| **MobileNetV3Large** | Depthwise Separable Conv + SE + H-Swish | Siêu nhẹ, cho mobile | ~4.2M |
| **EfficientNetV2S** | Fused-MBConv + SE-Block | Cân bằng tốc độ & chính xác | ~21M |
| **Proposed_ConvNeXtTiny_SE** | ConvNeXtTiny + SE-Block + Focal Loss | Mô hình đề xuất riêng | ~29M |

## 6.2 Classifier Head thống nhất

```
Base CNN (ImageNet weights, frozen/unfrozen)
    ↓ GlobalAveragePooling2D
    ↓ Dropout(0.3)
    ↓ Dense(7, softmax) → 7 lớp bệnh
```

Riêng Proposed thêm:
```
Base CNN → SE-Block → GAP → Dense(512, gelu) → BN → Dropout(0.4)
    → Dense(256, gelu) → Dropout(0.3) → Dense(7, softmax)
```

## 6.3 Huấn luyện 3 giai đoạn (Transfer Learning)

| Giai đoạn | Base CNN | LR | Epochs | Mục đích |
|---|---|---|---|---|
| Phase 1 (Warm-up) | Đóng băng 100% | 1e-3 | 10 | Head học phân loại |
| Phase 2 (Fine-tune) | Mở 30% trên cùng | CosineDecay(1e-4→~0) | **20** | CNN chuyên biệt hóa cho lá lúa |
| Phase 3 (Full tune) | Mở 100% | 1e-5 | 5 | Tinh chỉnh toàn bộ kiến trúc |

> **Biểu đồ training history** hiển thị cả 3 phase với đường dọc đánh dấu: đỏ = Unfreeze 30%, tím = Full Fine-tune.
>
> **Kỹ thuật mới:**
> - **Label Smoothing (0.1)**: Thay vì one-hot cứng [0,0,1,0,...], dùng nhãn mềm [0.014, 0.014, 0.914, ...] → giảm overconfidence.
> - **Cosine Annealing LR**: LR giảm mượt theo đường cosine từ 1e-4 → ~0, giúp hội tụ tốt hơn flat LR.
> - **Test-Time Augmentation (TTA)**: Dự đoán trên 4 biến thể (original + 3 flips), lấy trung bình → tăng accuracy.

---

# SLIDE 10: STEP 7 — ĐÁNH GIÁ MÔ HÌNH (Data Evaluation)

## 7.1 So sánh 4 mô hình CNN

- So sánh trên tập **Validation** bằng: Accuracy, Precision, Recall, F1-Macro, F1-Weighted
- Áp dụng **Test-Time Augmentation (TTA)**: load best weights, dự đoán trên 4 biến thể flip → lấy trung bình
- Chọn Best Model theo **F1-Macro** cao nhất
- Vẽ **Confusion Matrix** cho cả 4 mô hình
- **Classification Report** chi tiết cho Best Model

## 7.2 Knowledge Distillation (CNN → CNN)

| Vai trò | Mô hình | Mục đích |
|---|---|---|
| **Teacher** | Best CNN (mạnh) | Tạo soft-label (T=4.0) |
| **Student** | MobileNetV3Large (nhẹ) | Học từ Teacher, giữ size nhỏ |

> Cả Teacher và Student đều là **CNN**. KD cho phép triển khai model nhẹ trên thiết bị di động.

---

# SLIDE 11: STEP 8 — TINH CHỈNH (Hyper-parameter Tuning)

## 8.1 Phase 3: Full Fine-tune

| Tham số | Giá trị |
|---|---|
| Base CNN | Mở khóa **100%** layers |
| Learning Rate | **1e-5** (siêu nhỏ) |
| Epochs | 5 |
| EarlyStopping | patience=5 |
| ReduceLROnPlateau | min_lr=1e-8 |

## 8.2 Regularization

- Dropout(0.3) tại Classifier Head
- L2 penalty (1e-4) trên Dense cuối
- EarlyStopping + ReduceLROnPlateau

## 8.3 Ablation Study

- So sánh **ảnh gốc** vs **ảnh tách nền** trên Best Model CNN
- Bảng delta: Accuracy, Precision, Recall, F1 trước/sau Phase 3

---

# SLIDE 12: STEP 9 — PIPELINE & ENSEMBLE

## 9.1 Inference Pipeline

```python
RiceDiseasePipeline(best_model, CLASS_NAMES, IMG_SIZE=224)
    → load_img → resize 224×224 → [0,255] float32 → predict → Top-3
```

## 9.2 Soft-voting Ensemble (4 CNN)

```
ConvNeXtSmall  ──┐
MobileNetV3    ──┤  Average   → argmax → Predicted Class
EfficientNetV2 ──┤  Probs
Proposed_SE    ──┘
```

> Ensemble thường cho kết quả cao hơn bất kỳ model đơn lẻ nào.

---

# SLIDE 13: STEP 10 — KẾT LUẬN

## 10.1 Mở khóa Test Set

- Đánh giá đồng loạt **4 mô hình CNN** trên Test
- 6 chỉ số: Accuracy, Precision, Recall, F1-Macro, F1-Weighted, Cohen's Kappa
- Biểu đồ so sánh hiệu suất

## 10.2 ROC-AUC & Confusion Matrix

- Đồ thị **ROC-AUC đa lớp** cho Best Model
- **Confusion Matrix** cuối cùng trên Test
- **Classification Report** chi tiết

## 10.3 Grad-CAM (Giải thích CNN)

- Trực quan hóa vùng CNN tập trung khi phân loại
- Kỹ thuật này **chỉ hoạt động với CNN** (cần lớp Conv2D)
- Kỳ vọng: vùng nóng trùng với vết bệnh trên lá

## 10.4 Thông số phần cứng

- So sánh 4 CNN: Params (M), Size (MB), FPS

---

# SLIDE 14: TỔNG KẾT KIẾN TRÚC

> **Toàn bộ dự án sử dụng CNN — Không có RNN**

| Thành phần | Kiến trúc | Layers chính |
|---|:---:|---|
| DCGAN Generator | **CNN** | Conv2DTranspose, BatchNorm, ReLU, tanh |
| DCGAN Discriminator | **CNN** | Conv2D, LeakyReLU, Dropout |
| ConvNeXtSmall | **CNN** | DepthwiseConv, LayerNorm, GELU |
| MobileNetV3Large | **CNN** | DepthwiseSepConv, SE, H-Swish |
| EfficientNetV2S | **CNN** | Fused-MBConv, SE |
| Proposed_ConvNeXtTiny_SE | **CNN** | ConvNeXtTiny + SE-Block |
| Grad-CAM | **CNN** | Gradient trên Conv2D cuối |

**Tại sao không dùng RNN?** Bài toán phân loại ảnh không có tính tuần tự. RNN (LSTM, GRU) phù hợp cho dữ liệu chuỗi (text, time series, audio). CNN phù hợp cho dữ liệu có cấu trúc không gian (ảnh, video frames).

---
