# BÁO CÁO ĐỀ TÀI: DỰ ĐOÁN BỆNH Ở LÚA SỬ DỤNG ADVANCED VISION MODELS TỐI ƯU TRÊN TPU V5E-8

---

## 1. Tên đề tài

**Dự đoán bệnh ở lúa dựa trên hình ảnh lá sử dụng các mô hình Deep Learning tiên tiến (SwinV2, ConvNeXtV2, DINOv2) trên kiến trúc TPU v5e-8**

*(Rice Leaf Disease Prediction using Advanced Vision Models on TPU v5e-8 Architecture)*

---

## 2. Mục tiêu của đề tài

### 2.1. Mục tiêu chung
Xây dựng một hệ thống phân loại mạnh mẽ, chính xác cao để nhận diện các trạng thái/loại bệnh trên lá lúa bằng cách áp dụng các kiến trúc State-of-the-Art (SOTA) trong thị giác máy tính, đồng thời khai thác tối đa sức mạnh của phần cứng TPU v5e-8.

### 2.2. Mục tiêu cụ thể
- **Đa phân lớp**: Nhận diện 7 trạng thái lá lúa (Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Sheath Blight, Hispa, Healthy).
- **So sánh 3 kiến trúc SOTA**: 
  1. **Swin Transformer V2**: Mô hình transformer với cơ chế cửa sổ trượt (shifted windows) tối ưu cho ảnh.
  2. **ConvNeXt V2**: Sự trở lại của kiến trúc CNN thuần túy cạnh tranh trực tiếp với Transformers.
  3. **DINOv2 (ViT-Large)**: Mô hình Foundation (từ Meta AI) học self-supervised cho trích xuất đặc trưng cực mạnh.
- **Tối ưu hóa TPU**: Implements `TPUStrategy`, Mixed Precision bfloat16, và `tf.data.AUTOTUNE` để tăng tốc độ huấn luyện trên Kaggle TPU v5e-8.
- Đạt độ chính xác tuyệt đối ≥ 90% trên tập test.

---

## 3. Dataset

### 3.1. Nguồn dữ liệu (Kết hợp 4 kho Kaggle: >10.000 ảnh)
| # | Dataset (Kaggle URL) | Tác giả | Quy mô |
|---|----------------------|---------|--------|
| 1 | `anshulm257/rice-disease-dataset` | Anshul M. | ~6 class |
| 2 | `jonathanrjpereira/rice-disease` | Jonathan R. | 4 class |
| 3 | `nurnob101/rice-disease` | Nurnob | ~4 class |
| 4 | `thaonguyen0712/rice-disease` | Thao Nguyen | ~4 class |

### 3.2. Tiền xử lý & Cân bằng (Tối ưu cho TPU)
- **Kích thước ảnh**: Tùy mô hình, tiêu chuẩn ở mức `224x224` (Swin/ConvNeXt) nhưng DINOv2 có thể scale lớn hơn nếu được thiết lập.
- **Lọc nhiễu & Trùng lặp**: Check mã MD5 để loại bỏ hash trùng và dùng chuỗi PIL verify ảnh hỏng.
- **Tạo Dataset `tf.data` Pipeline**: Bắt buộc drop batches bị rỗng (drop_remainder) để phân mảnh đều trên 8 Cores TPU.

---

## 4. Pipeline dự kiến của đề tài

Pipeline được thiết kế 10 bước chuyên nghiệp trên Kaggle:

```text
┌──────────────────────────────────────────────────────────────────┐
│ Step 0: TPU Initialization & Mixed Precision Policy (bfloat16)   │
│ Step 1: Problem Definition                                       │
│ Step 2: Download & Clean Data (kagglehub + de-duplicate)         │
│ Step 3: Fast EDA & Distribution check                            │
│ Step 4: Distributed Feature Augmentation Pipeline                │
│ Step 5: Tf.data Partitioning (Train/Val/Test + drop_remainder)   │
│ Step 6: Load Advanced Models via HuggingFace inside TPU Strategy │
│ Step 7: Train Phase 1 (Frozen Head) on Validation Set            │
│ Step 8: Train Phase 2 & 3 (Gradual Unfreeze) via TPU             │
│ Step 9: Inference Pipeline setup                                 │
│ Step 10: Conclusion & Grad-CAM Metrics                           │
└──────────────────────────────────────────────────────────────────┘
```

### Chiến lược Training 3-Phase (Gradual Unfreezing):
Các model SOTA rất nhạy cảm nếu fine-tune toàn bộ lớp ngay lập tức với lượng data ít:
1. **Warmup / Frozen (5-10 Epoch)**: Đóng băng Base Model (DINOv2/SwinV2/ConvNeXtV2), chỉ học Linear/Dense Classifier Head ở Learning Rate (LR) cao (e.g., 1e-3).
2. **Partial Unfreeze (10-15 Epoch)**: Mở 30-50% các block trên cùng (Top Blocks), train với LR thấp hơn (e.g., 1e-4).
3. **Full Fine-tune (Tùy chọn)**: Dùng Learning Rate siêu nhỏ (1e-5) với Cosine Decay để tối ưu hóa tận cùng trọng số.

---

## 5. Các chỉ số đánh giá & Phần cứng

### Các Metrics:
1. **Accuracy (Top-1)** và **F1-Score** (Trung bình hòa hợp có trọng số).
2. **Confusion Matrix**: Phát hiện lớp bị nhận diện nhầm thường xuyên nhất.
3. **Training Scalability**: Phân tích thời gian hoàn thành 1 epoch thông qua 8 TPU cores với Mixed Precision.

### Tối ưu TPU v5e-8:
- Mức độ sử dụng Core TPU (128 Batch Size = 16 img/core).
- Bộ nhớ Cache: `tf.data.Dataset.cache()` trên Kaggle VM Ram.

---

## 6. Các tài liệu tham khảo và nghiên cứu tương tự

1. **Liu, Z. et al. (2022).** *"Swin Transformer V2: Scaling Up Capacity and Resolution"*. CVPR 2022.
2. **Woo, S. et al. (2023).** *"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"*. Meta AI Research.
3. **Oquab, M. et al. (2023).** *"DINOv2: Learning Robust Visual Features without Supervision"*. arXiv:2304.07193.
4. Tài liệu TensorFlow: *"Use TPUs"*, *"Mixed Precision"*, Google Cloud.

---

## 7. Lịch trình và Nhiệm vụ

| Thành viên | Nhiệm vụ chính |
|------------|-----------------|
| TV1 | Kỹ sư Data: Xử lý bộ tf.data siêu tốc, Lọc dữ liệu, Cân bằng lớp. |
| TV2 | Kỹ sư ML (Hạ tầng): Setup Strategy TPU, Load Pre-trained Weights. |
| TV3 | Kỹ sư ML (Triển khai): Giám sát Training, Checkpoint, Đánh giá metrics và Grad-CAM. |
| Cả nhóm | Viết báo cáo, review hiệu năng suy luận, Báo cáo tiến độ chuẩn Kaggle. |
