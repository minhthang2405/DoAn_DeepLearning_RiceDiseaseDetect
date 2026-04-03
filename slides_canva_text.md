# NỘI DUNG TỪNG SLIDE — COPY VÀO CANVA

---

## SLIDE 1: TRANG BÌA

**Tiêu đề chính:**
Phân loại Bệnh Lá Lúa sử dụng các mô hình Deep Learning tiên tiến

**Tiêu đề phụ (tiếng Anh):**
Rice Leaf Disease Classification using Advanced Deep Learning Models

**Thông tin bổ sung:**
Môn học: Đồ Án Deep Learning
Phần cứng: GPU T4×2 (MirroredStrategy) — Kaggle
Models: ConvNeXtSmall · MobileNetV3Large · EfficientNetV2S · Proposed_ConvNeXtTiny_SE

---

## SLIDE 2: TỔNG QUAN PIPELINE

**Tiêu đề:** Quy trình 10 bước (ML Pipeline)

**Nội dung:**
Dự án được thiết kế theo quy trình chuyên nghiệp gồm 10 bước, từ việc hiểu bài toán cho đến đánh giá kết quả cuối cùng. Bài thuyết trình hôm nay trình bày 5 bước đầu tiên, tập trung vào giai đoạn chuẩn bị dữ liệu.

Step 1: Hiểu bài toán (Problem Understanding)
Step 2: Hiểu dữ liệu (Data Understanding)
Step 3: Hiểu đặc trưng (Feature Understanding — EDA)
Step 4: Kỹ thuật đặc trưng (Feature Engineering)
Step 5: Phân chia tập dữ liệu và cân bằng dữ liệu (Dataset Partition)
Step 6–10: Mô hình hóa, Huấn luyện, Đánh giá, Suy luận, Kết luận

---

## SLIDE 3: STEP 1 — HIỂU BÀI TOÁN

**Tiêu đề:** Bước 1: Hiểu bài toán (Problem Understanding)

**Nội dung:**
Bài toán thuộc dạng Phân loại Đa lớp (Multi-class Image Classification). Đầu vào là ảnh RGB chụp lá lúa ngoài đồng, kích thước chuẩn hóa 224×224×3 pixel. Đầu ra là nhãn dự đoán thuộc 1 trong 7 lớp bệnh hoặc trạng thái khỏe mạnh.

**7 lớp phân loại:**
Bacterial Leaf Blight (Bạc lá vi khuẩn), Brown Spot (Đốm nâu), Healthy (Khỏe mạnh), Hispa (Sâu cuốn lá nhỏ), Leaf Blast (Đạo ôn lá), Leaf Scald (Bỏng lá), Sheath Blight (Khô vằn)

---

## SLIDE 4: STEP 1 — TẠI SAO CHỌN 4 MÔ HÌNH NÀY?

**Tiêu đề:** Lý do chọn 4 mô hình

**MobileNetV3Large:**
Đại diện cho nhóm mô hình siêu nhẹ (~4.2M tham số). Mục đích là kiểm chứng độ chính xác đạt được khi triển khai trên thiết bị di động hoặc thiết bị có tài nguyên hạn chế.

**EfficientNetV2S:**
Đại diện cho sự cân bằng hoàn hảo giữa tốc độ và độ chính xác (~21M tham số). Đây là kiến trúc SOTA của năm 2021, tối ưu hóa số lượng tham số trong khi vẫn giữ hiệu suất cao.

**ConvNeXtSmall:**
Đại diện cho sức mạnh thuần túy của kiến trúc CNN hiện đại (~50M tham số). Đây là kiến trúc SOTA của năm 2022, được thiết kế để cạnh tranh trực tiếp với các mô hình Vision Transformer.

**Proposed_ConvNeXtTiny_SE (Mô hình đề xuất):**
Dựa trên ConvNeXtTiny (~28M tham số), tích hợp thêm SE-Block (Squeeze-and-Excitation) để tăng khả năng tập trung vào các kênh đặc trưng quan trọng. Kết hợp Focal Loss để xử lý hiệu quả các lớp bị mất cân bằng và tập trung vào mẫu khó nhận diện. Tổng cộng khoảng ~29M tham số.

---

## SLIDE 5: STEP 2 — HIỂU DỮ LIỆU

**Tiêu đề:** Bước 2: Hiểu dữ liệu (Data Understanding)

**Nguồn dữ liệu:**
Dữ liệu được tổng hợp từ 4 bộ dataset trên Kaggle: anshulm257/rice-disease-dataset, jonathanrjpereira/rice-disease, nurnob101/rice-disease, và thaonguyen0712/rice-disease. Bốn nguồn được kết hợp và chuẩn hóa về 7 lớp thống nhất để tạo bộ dữ liệu phong phú hơn.

**Chuẩn hóa nhãn:**
Do mỗi bộ dữ liệu đặt tên lớp khác nhau, cần xây dựng bảng ánh xạ để chuẩn hóa. Ví dụ: "bacterial_blight", "bacterialblight", "blight" đều được ánh xạ về Bacterial_Leaf_Blight. Tương tự, "blast", "rice_blast", "neck_blast" được ánh xạ về Leaf_Blast. "normal" được ánh xạ về Healthy.

**Mapping tiếng Việt (dataset thaonguyen0712/rice-disease):**
Dataset này sử dụng tên folder tiếng Việt: "bo_gai" (Bọ gai) ánh xạ về Hispa, "chay_bia_la" (Cháy bìa lá) ánh xạ về Bacterial_Leaf_Blight, "dao_on" (Đạo ôn) ánh xạ về Leaf_Blast, "dom_nau" (Đốm nâu) ánh xạ về Brown_Spot, "vang_la" (Vàng lá) ánh xạ về Leaf_Scald.

---

## SLIDE 6: STEP 2 — KIỂM TRA CHẤT LƯỢNG DỮ LIỆU

**Tiêu đề:** Kiểm tra chất lượng dữ liệu (Data Quality Check)

**Ảnh hỏng (Corrupt):**
Sử dụng hàm PIL.Image.verify() để kiểm tra cấu trúc file ảnh. Hàm này đọc toàn bộ nội dung file và xác minh cấu trúc bên trong có đúng chuẩn định dạng JPEG hoặc PNG hay không. Nếu file bị hỏng hoặc không đọc được thì bị loại bỏ.

**Ảnh quá nhỏ:**
Kiểm tra chiều rộng và chiều cao của ảnh. Nếu bất kỳ chiều nào nhỏ hơn 32 pixel thì bị loại bỏ, bởi khi resize lên 224×224 pixel sẽ mất hết chi tiết và trở thành nhiễu cho mô hình.

**Ảnh trùng lặp:**
Tính mã hash MD5 cho toàn bộ nội dung binary của từng file ảnh. Nếu hai file có cùng mã hash thì nội dung hoàn toàn giống nhau, chỉ giữ lại một bản. Đây là phương pháp rất nhanh với độ phức tạp O(n), xác suất xung đột hash gần bằng 0 (khoảng 1/2^128).

**Kiểm tra mất cân bằng:**
Tính Imbalance Ratio bằng cách chia số lượng lớp lớn nhất cho số lượng lớp nhỏ nhất. Nếu tỷ lệ lớn hơn 3, dữ liệu được xem là mất cân bằng và sẽ được xử lý bằng class weights và DCGAN synthesis ở các bước sau.

---

## SLIDE 7: STEP 3 — PHÂN TÍCH ĐƠN BIẾN

**Tiêu đề:** Bước 3: Phân tích Khám phá Dữ liệu — Đơn biến (Univariate)

**Phân phối số lượng ảnh mỗi lớp:**
Sử dụng biểu đồ cột (Bar Chart) để hiển thị số lượng ảnh mỗi lớp, kết hợp biểu đồ tròn (Pie Chart) để thể hiện tỷ lệ phần trăm. Qua đó nhận diện lớp nào thuộc nhóm thiểu số và lớp nào thuộc nhóm đa số.

**Phân phối kích thước ảnh:**
Sử dụng biểu đồ Histogram cho chiều rộng (Width) và chiều cao (Height) của ảnh. Biểu đồ Scatter thể hiện mối quan hệ Width và Height. Kết quả cho thấy kích thước ảnh rất đa dạng, từ 224 đến hơn 3000 pixel. Median Width là 1461px, median Height là 1200px. Hầu hết ảnh có tỷ lệ gần vuông. Điều này chứng minh việc resize đồng nhất về 224×224 là cần thiết.

---

## SLIDE 8: STEP 3 — PHÂN TÍCH NHỊ BIẾN VÀ ĐA BIẾN

**Tiêu đề:** Phân tích Nhị biến (Bivariate) và Đa biến (Multivariate)

**Phân tích nhị biến — Ảnh mẫu theo lớp:**
Hiển thị 5 ảnh mẫu cho mỗi lớp bệnh, giúp trực quan hóa sự khác biệt giữa các loại bệnh. Một số bệnh có triệu chứng tương tự nhau, ví dụ Brown Spot và Leaf Blast đều có đốm trên lá, đây là thách thức cho mô hình phân loại.

**Phân tích đa biến — Kênh màu RGB theo lớp:**
Sử dụng Boxplot cho từng kênh R, G, B phân theo lớp bệnh. Kết quả cho thấy kênh G (Green) là đặc trưng phân biệt mạnh nhất: lá Healthy có giá trị G cao nhất khoảng 200–210 do nhiều diệp lục, trong khi các lá bệnh như Bacterial Leaf Blight hay Leaf Scald có giá trị G thấp hơn nhiều khoảng 120–130 do mất diệp lục. Healthy và Hispa có giá trị RGB tương tự nhau nên khó phân biệt chỉ bằng màu sắc, cần CNN trích xuất đặc trưng hình dạng của vết sâu ăn. Điều này chứng minh CNN có cơ sở để học phân biệt các lớp dựa trên cả đặc trưng màu lẫn đặc trưng cấu trúc bề mặt.

---

## SLIDE 9: STEP 4 — MÃ HÓA NHÃN VÀ AUGMENTATION CƠ BẢN

**Tiêu đề:** Bước 4: Kỹ thuật Đặc trưng — Mã hóa và Tăng cường

**Mã hóa nhãn (Label Encoding):**
Chuyển đổi nhãn dạng chuỗi thành số nguyên: Bacterial_Leaf_Blight thành 0, Brown_Spot thành 1, Healthy thành 2, Hispa thành 3, Leaf_Blast thành 4, Leaf_Scald thành 5, Sheath_Blight thành 6.

**Tăng cường dữ liệu cơ bản (Data Augmentation) bằng TensorFlow:**
Tất cả các phép biến đổi chạy trực tiếp trên GPU để tối ưu tốc độ, bao gồm: Lật ngang ngẫu nhiên (Random Flip Left-Right), Lật dọc ngẫu nhiên (Random Flip Up-Down), Thay đổi độ sáng ±20% (Random Brightness), Thay đổi độ tương phản trong khoảng 0.8 đến 1.2 (Random Contrast), Thay đổi độ bão hòa trong khoảng 0.8 đến 1.2 (Random Saturation), Thay đổi tông màu ±0.05 (Random Hue).

---

## SLIDE 10: STEP 4 — CUTMIX VÀ MIXUP

**Tiêu đề:** Tăng cường Dữ liệu Nâng cao — CutMix và MixUp

**MixUp (50% xác suất):**
Lấy 2 ảnh bất kỳ trong batch, trộn chúng theo tỷ lệ lambda được lấy từ phân phối Beta(0.2, 0.2). Công thức: Ảnh mới bằng lambda nhân Ảnh A cộng (1 trừ lambda) nhân Ảnh B. Nhãn cũng được trộn theo cùng tỷ lệ, tạo ra nhãn mềm (soft label). Phân phối Beta(0.2, 0.2) có dạng chữ U, nghĩa là lambda thường rất gần 0 hoặc rất gần 1, nên một ảnh sẽ chiếm ưu thế, ảnh kia chỉ lẫn nhẹ. MixUp giúp model không quá tự tin vào quyết định, tạo ranh giới phân loại mượt hơn.

**CutMix (50% xác suất):**
Thay vì trộn toàn bộ pixel như MixUp, CutMix cắt một vùng hình chữ nhật ngẫu nhiên từ ảnh B rồi dán đè lên ảnh A. Nhãn được trộn theo tỷ lệ diện tích thực tế của vùng cắt. CutMix buộc model phải nhận diện đúng bệnh kể cả khi chỉ nhìn thấy một phần lá, tăng khả năng nhận diện trong điều kiện thực tế khi ảnh có thể bị che khuất.

**Lý do kết hợp cả hai:** MixUp dạy model rằng các lớp bệnh có thể có đặc trưng chồng lấn. CutMix dạy model kỹ năng nhận diện cục bộ. Kết hợp cả hai giúp model tổng quát hóa tốt hơn và giảm overfitting đáng kể.

---

## SLIDE 11: STEP 4 — CLAHE

**Tiêu đề:** CLAHE — Cân bằng Histogram Thích ứng có Giới hạn Tương phản

**Vấn đề:**
Ảnh chụp lá lúa ngoài đồng thường bị ánh sáng không đều, vết bệnh mờ nhạt khó phân biệt với phần lá bình thường.

**Giải pháp — CLAHE:**
Đầu tiên chuyển ảnh từ không gian màu RGB sang LAB. Trong LAB, kênh L chứa thông tin độ sáng, kênh A và B chứa thông tin màu. Tách riêng kênh L và áp dụng CLAHE lên kênh L với tham số clipLimit bằng 2.0 và tileGridSize bằng 8×8. Ảnh được chia thành lưới 8×8 gồm 64 ô nhỏ, mỗi ô được cân bằng histogram độc lập, sau đó nội suy giữa các ô để tránh ranh giới rõ rệt. Cuối cùng ghép kênh L đã cân bằng với kênh A và B gốc, rồi chuyển lại về RGB.

**Tại sao dùng LAB thay vì cân bằng trực tiếp trên RGB?**
Nếu cân bằng trực tiếp trên R, G, B thì sẽ thay đổi màu sắc, lá xanh có thể biến thành xanh dương. Chuyển sang LAB cho phép chỉ chỉnh độ sáng mà giữ nguyên màu sắc. Kết quả là vết bệnh trở nên rõ ràng và tương phản cao hơn, giúp CNN trích xuất đặc trưng hiệu quả hơn.

---

## SLIDE 12: STEP 4 — TÁCH NỀN

**Tiêu đề:** Tách nền (Background Segmentation)

**Mục tiêu:**
Tạo thêm biến thể ảnh "leaf-only" với nền đen để so sánh hiệu năng model trên ảnh gốc và ảnh đã tách nền (ablation study).

**Phương pháp 1 — SAM (Segment Anything Model):**
Nếu có sẵn file weights của SAM từ Meta AI, hệ thống sẽ dùng SAM để segment. SAM nhận một điểm prompt ở tâm ảnh, giả định lá lúa nằm ở giữa, rồi tạo mask với độ chính xác rất cao.

**Phương pháp 2 — Fallback (GrabCut + Green Prior):**
Nếu không có SAM weights, hệ thống tự động dùng thuật toán truyền thống. Bước 1: GrabCut nhận một hình chữ nhật bao quanh 90% ảnh, dùng Gaussian Mixture Model để phân tách foreground và background qua 4 vòng lặp. Bước 2: Chuyển ảnh sang không gian HSV, lọc tất cả pixel có Hue từ 25 đến 95 độ (dải màu xanh lá), đánh dấu chúng là lá và hợp nhất với mask GrabCut. Bước 3: Áp dụng Morphology Close (giãn ra rồi co lại) với kernel 7×7 để lấp các lỗ nhỏ trong mask. Cuối cùng, tất cả pixel có mask bằng 0 được đặt thành đen, chỉ giữ lại phần lá.

Kết quả tách nền được lưu cache vào thư mục segmented_cache để tránh xử lý lại. Train giới hạn 1200 ảnh, Val và Test segment 100%.

---

## SLIDE 13: STEP 5 — CHIA TẬP DỮ LIỆU

**Tiêu đề:** Bước 5: Phân chia Tập dữ liệu (Dataset Partition)

**Chiến lược chia:**
Dữ liệu được chia thành 3 tập theo tỷ lệ 70% Train, 15% Validation, 15% Test. Sử dụng Stratified Split (phân tầng) để đảm bảo mỗi tập giữ nguyên tỷ lệ phân phối gốc của các lớp bệnh. Tập Test được khóa hoàn toàn và không được sử dụng cho đến bước đánh giá cuối cùng (Step 10), tránh tuyệt đối data leakage.

---

## SLIDE 14: STEP 5 — CONDITIONAL DCGAN

**Tiêu đề:** Sinh ảnh bằng Conditional DCGAN

**Lý do cần sinh ảnh:**
Tập Train ban đầu chưa đạt mốc 10.000 ảnh (tính riêng tập Train, không tính Val/Test) và một số lớp có số lượng ít hơn các lớp khác. Cần sinh thêm ảnh cho các lớp thiểu số để cân bằng. Ảnh chỉ được sinh vào tập Train để tránh data leakage tuyệt đối. Thiếu hụt được tính bằng `shortage = max(0, TARGET_TRAIN - len(X_train_list))`.

**DCGAN là gì?**
DCGAN là viết tắt của Deep Convolutional Generative Adversarial Network — một kiến trúc thuộc họ **CNN (Convolutional Neural Network)**, KHÔNG phải RNN. Đây là kiến trúc gồm 2 mạng CNN đối đầu nhau: Generator (mạng sinh) sử dụng Conv2DTranspose (Transposed Convolution) để tăng kích thước ảnh từ noise, và Discriminator (mạng phân biệt) sử dụng Conv2D thông thường để phân biệt ảnh thật và ảnh giả. Hai mạng huấn luyện đồng thời trong một trò chơi tổng bằng không (zero-sum game). Conditional DCGAN bổ sung thêm thông tin nhãn lớp vào cả hai mạng, cho phép kiểm soát sinh ra ảnh thuộc lớp bệnh nào.

---

## SLIDE 15: STEP 5 — KIẾN TRÚC GENERATOR

**Tiêu đề:** Generator — Mạng sinh ảnh

**Quy trình:**
Generator nhận đầu vào gồm 2 phần: một vector nhiễu ngẫu nhiên 128 chiều (noise) được lấy mẫu từ phân phối chuẩn N(0,1), và một class_id cho biết muốn sinh ảnh lớp bệnh nào.

Class_id được biến đổi thành vector 32 chiều thông qua lớp Embedding (trainable), rồi project lên 128 chiều bằng lớp Dense. Vector này được nối (concatenate) với noise tạo thành vector kết hợp 256 chiều, chứa cả tính ngẫu nhiên lẫn thông tin lớp.

Vector kết hợp đi qua lớp Dense tạo ra tensor 4×4×512, đây là bản nháp thô nhất của ảnh. Tiếp theo qua 4 lớp Conv2DTranspose (Transposed Convolution) với stride bằng 2, mỗi lớp tăng gấp đôi kích thước không gian: từ 4×4 lên 8×8, rồi 16×16, rồi 32×32, và cuối cùng 64×64. Số kênh giảm dần: 512, 256, 128, 64, và lớp cuối cùng có 3 kênh tương ứng RGB. Giữa các lớp sử dụng BatchNormalization để ổn định gradient và ReLU (theo đúng chuẩn paper DCGAN gốc, tạo sparse activation giúp ảnh sắc nét hơn). Activation cuối cùng là tanh, cho output trong khoảng âm 1 đến 1.

**Transposed Convolution là gì?**
Đây là phép toán ngược của Convolution thông thường. Trong khi Conv2D với stride 2 giảm kích thước ảnh đi một nửa, Conv2DTranspose với stride 2 tăng gấp đôi kích thước. Cơ chế hoạt động: chèn các pixel 0 giữa các pixel input (tạo khoảng trống), sau đó quét kernel lên tensor đã mở rộng để tạo ra output lớn hơn.

---

## SLIDE 16: STEP 5 — KIẾN TRÚC DISCRIMINATOR

**Tiêu đề:** Discriminator — Mạng phân biệt thật/giả

**Quy trình:**
Discriminator nhận đầu vào là ảnh 64×64×3 (có thể là ảnh thật từ dataset hoặc ảnh giả từ Generator) cùng với class_id.

Class_id được biến đổi thành một bản đồ nhãn không gian (spatial label map) có kích thước 64×64×1 thông qua Embedding rồi Reshape, sau đó nối với ảnh tạo thành tensor 64×64×4. Discriminator đóng vai trò như một mạng CNN tiêu chuẩn, sử dụng 4 lớp Conv2D với stride bằng 2 để giảm dần kích thước: từ 64×64 xuống 32×32, rồi 16×16, rồi 8×8, cuối cùng 4×4. Số kênh tăng dần: 64, 128, 256, 512. Giữa các lớp sử dụng LeakyReLU(0.2) và Dropout(0.3, để ngăn Discriminator quá mạnh). Cuối cùng Flatten thành vector 8192 chiều rồi qua Dense(1) cho ra 1 logit duy nhất: số dương nghĩa là ảnh thật, số âm nghĩa là ảnh giả.

---

## SLIDE 17: STEP 5 — QUÁ TRÌNH HUẤN LUYỆN GAN

**Tiêu đề:** Quá trình huấn luyện và sinh ảnh

**Quá trình huấn luyện:**
Mỗi bước huấn luyện gồm 2 giai đoạn. Giai đoạn 1: Huấn luyện Discriminator, cho D xem ảnh thật và muốn D dự đoán "thật" (label bằng 1), đồng thời cho D xem ảnh giả từ Generator và muốn D dự đoán "giả" (label bằng 0). Giai đoạn 2: Huấn luyện Generator, cho G sinh ảnh giả, đưa qua D, nhưng lần này G muốn D dự đoán "thật" (label bằng 1), tức là G cố đánh lừa D. Loss function sử dụng BinaryCrossentropy với from_logits bằng True. Optimizer là Adam với learning rate 2×10⁻⁴ và beta1 bằng 0.5 (giảm momentum để ổn định cho GAN, theo khuyến nghị từ paper DCGAN gốc).

**Các tham số huấn luyện:**
Noise dimension: 128; Image size: 64×64; Batch size: 64; Tổng số steps: **5000** (tăng từ 2000 để ảnh chất lượng cao hơn); Learning rate: 0.0002; Target: Train set đạt 10.000 ảnh.

**Quy trình sinh ảnh sau huấn luyện:**
Bước 1: Tính thiếu hụt dựa trên kích thước tập Train thực tế (shortage = max(0, TARGET_TRAIN - len(X_train_list))). Bước 2: Tính số ảnh cần bù cho từng lớp thiểu số dựa trên target (bằng lớp có nhiều ảnh nhất). Bước 3: Train GAN 5000 steps trên tập Train. Bước 4: Dùng Generator đã train để sinh ảnh ảo theo đúng số lượng cần bù cho mỗi lớp. Bước 5: Thêm ảnh sinh vào tập Train. Bước 6: Tính Class Weights (balanced) bằng compute_class_weight để xử lý phần mất cân bằng còn lại. Kết quả: tập Train được cân bằng gần tuyệt đối giữa các lớp, trong khi tập Val và Test giữ nguyên phân phối gốc để đánh giá trung thực.

---

## SLIDE 18: STEP 5 — PIPELINE TF.DATA

**Tiêu đề:** Pipeline tf.data siêu tốc cho GPU

**Pipeline xử lý dữ liệu:**
Toàn bộ dữ liệu được xử lý qua pipeline tf.data của TensorFlow, cho phép CPU và GPU hoạt động song song. Pipeline gồm các bước: shuffle xáo trộn dữ liệu, map load_and_preprocess đọc ảnh từ đĩa và resize về 224×224, map augment áp dụng augmentation cơ bản (chỉ cho Train), batch gom thành batch, map CutMix hoặc MixUp áp dụng augmentation nâng cao trên từng batch (chỉ cho Train), và prefetch(AUTOTUNE) nạp trước batch tiếp theo trong lúc GPU đang xử lý batch hiện tại.

**Các đặc điểm quan trọng:**
Pixel được giữ nguyên trong khoảng 0 đến 255, không chia 255, vì các model sử dụng include_preprocessing bằng True sẽ tự chuẩn hóa bên trong theo cách riêng của từng kiến trúc. AUTOTUNE cho phép TensorFlow tự động tối ưu số luồng song song. drop_remainder bằng True cho tập Train đảm bảo mỗi batch đều cùng kích thước để GPU xử lý tối ưu. Ngoài ra, hệ thống tạo song song 2 bộ dataset: ảnh gốc và ảnh đã tách nền, để thực hiện ablation study so sánh hiệu năng ở bước đánh giá.

---

## SLIDE 19: TÓM TẮT STEP 1–5

**Tiêu đề:** Tóm tắt 5 bước chuẩn bị dữ liệu

**Step 1 — Problem Understanding:** Xác định bài toán phân loại đa lớp với 7 lớp bệnh lá lúa, chọn 4 mô hình CNN đại diện cho các trường phái kiến trúc khác nhau.

**Step 2 — Data Understanding:** Gộp 4 bộ dataset từ Kaggle, chuẩn hóa nhãn, loại bỏ ảnh hỏng, ảnh quá nhỏ và ảnh trùng lặp, kiểm tra mất cân bằng.

**Step 3 — Feature Understanding:** Phân tích EDA ở 3 mức: đơn biến (phân phối lớp, kích thước ảnh), nhị biến (ảnh mẫu theo lớp), đa biến (kênh RGB theo lớp). Xác nhận kênh Green là đặc trưng phân biệt mạnh nhất.

**Step 4 — Feature Engineering:** Mã hóa nhãn, tăng cường dữ liệu bằng TF Augment cơ bản kết hợp CutMix và MixUp nâng cao, cải thiện ảnh bằng CLAHE, tách nền bằng GrabCut hoặc SAM.

**Step 5 — Dataset Partition:** Chia 70-15-15 có phân tầng, sinh ảnh bằng Conditional DCGAN (kiến trúc CNN, **5000 steps**) để cân bằng tập Train đạt 10.000 ảnh (tính shortage dựa trên kích thước tập Train thực tế), xây dựng pipeline tf.data tối ưu cho GPU.

**Lưu ý kiến trúc:** Toàn bộ dự án sử dụng **CNN (Convolutional Neural Network)** — bao gồm cả DCGAN (sinh ảnh) và 4 mô hình phân loại. Không sử dụng RNN vì bài toán phân loại ảnh không có tính tuần tự.

---

## SLIDE 20: STEP 6 — MÔ HÌNH HÓA DỮ LIỆU

**Tiêu đề:** Bước 6: Mô hình hóa Dữ liệu (Data Modelling) — CNN Transfer Learning

**Nội dung:**
Thay vì Machine Learning truyền thống (SVM, Random Forest), dự án sử dụng Học Chuyển giao Nâng cao (Advanced Transfer Learning) với 4 mô hình CNN pre-trained trên ImageNet. Tất cả 4 mô hình đều là kiến trúc CNN thuần túy, không có thành phần RNN (Recurrent Neural Network) nào.

**4 mô hình CNN được sử dụng:**

ConvNeXtSmall: CNN hiện đại SOTA 2022, dùng DepthwiseConv + LayerNorm + GELU, thiết kế cạnh tranh trực tiếp với Vision Transformer nhưng vẫn giữ bản chất CNN thuần túy. Khoảng **50M tham số**.

MobileNetV3Large: CNN siêu nhẹ của Google, dùng Depthwise Separable Convolution + SE-Block + Hard-Swish activation. Tối ưu cho thiết bị di động. Khoảng **4.2M tham số**.

EfficientNetV2S: CNN SOTA 2021, dùng Fused-MBConv + SE-Block. Cân bằng hoàn hảo giữa tốc độ và độ chính xác, tối ưu hóa parameter. Khoảng **21M tham số**.

Proposed_ConvNeXtTiny_SE (Mô hình đề xuất): Dựa trên ConvNeXtTiny (CNN), thêm SE-Block (Squeeze-and-Excitation) tùy chỉnh + Focal Loss. SE-Block hoạt động bằng cách: GlobalAveragePooling2D nén kênh, Dense(relu) giảm chiều, Dense(sigmoid) tạo attention weights, nhân ngược vào feature map gốc. Khoảng **29M tham số**.

---

## SLIDE 21: STEP 6 — KIẾN TRÚC CLASSIFIER HEAD

**Tiêu đề:** Classifier Head — Thiết kế thống nhất cho 4 mô hình CNN

**Nội dung:**
Tất cả 4 mô hình đều chia thành 2 phần: Base model (CNN backbone pre-trained trên ImageNet) và Classifier Head (phần phân loại tùy chỉnh).

Classifier Head có cấu trúc: GlobalAveragePooling2D lấy trung bình feature map 2D thành vector, Dropout(0.3) ngăn overfitting, Dense(7, softmax) phân loại ra 7 lớp bệnh. Riêng mô hình Proposed thêm: SE-Block trước GlobalAveragePooling2D, 2 lớp Dense(512, gelu) và Dense(256, gelu) với BatchNormalization và Dropout(0.4, 0.3).

Mô hình Proposed_ConvNeXtTiny_SE dùng Focal Loss thay vì CrossEntropy thông thường, giúp tập trung vào các mẫu khó phân loại và xử lý mất cân bằng hiệu quả hơn. Focal Loss nhân thêm hệ số (1 - p_t)^gamma vào cross-entropy, giảm trọng số mẫu dễ, tăng trọng số mẫu khó.

---

## SLIDE 22: STEP 6 — QUÁ TRÌNH HUẤN LUYỆN 2 GIAI ĐOẠN

**Tiêu đề:** Huấn luyện Transfer Learning 2 Giai đoạn

**Giai đoạn 1 — Warm-up (Base đóng băng):**
Đóng băng toàn bộ base CNN, chỉ huấn luyện Classifier Head. Learning rate 1e-3 (cao), 10 epochs. Mục đích: để Head học cách phân loại 7 lớp bệnh dựa trên đặc trưng sẵn có từ ImageNet.

**Giai đoạn 2 — Fine-tuning (Rã đông 30%):**
Mở khóa 30% lớp trên cùng của base CNN. Learning rate dùng **Cosine Annealing** giảm mượt từ 1e-4 → ~0 theo đường cosine (để không phá hỏng pre-trained weights). **20 epochs** (tăng từ 15). Mục đích: tinh chỉnh feature extractor CNN để chuyên biệt hóa cho ảnh lá lúa.

**Giai đoạn 3 — Full Fine-tune (Phase 3, ở Step 8):**
Mở khóa 100% layers của base CNN. Learning rate siêu nhỏ 1e-5. 5 epochs. Mục đích: tinh chỉnh toàn bộ kiến trúc để đạt hiệu suất tối đa. Biểu đồ training history hiển thị cả 3 phase với đường dọc đánh dấu: đỏ = Unfreeze 30%, tím = Full Fine-tune.

**Kỹ thuật cải thiện độ chính xác:**
**Label Smoothing (0.1):** Thay vì nhãn one-hot cứng [0,0,1,0,...], dùng nhãn mềm [0.014, 0.014, 0.914, ...]. Giảm overconfidence, giúp model tổng quát hóa tốt hơn. Áp dụng cho 3 model (ConvNeXtSmall, MobileNetV3, EfficientNetV2S); Proposed giữ Focal Loss riêng. **Cosine Annealing LR:** LR giảm mượt theo đường cosine từ 1e-4 → ~0 trong Phase 2, giúp hội tụ tốt hơn flat LR và val_loss liên tục cải thiện đến epoch cuối. **Test-Time Augmentation (TTA):** Khi đánh giá, dự đoán trên 4 biến thể (gốc + 3 flip), lấy trung bình xác suất → tăng accuracy mà không cần retrain.

**Các kỹ thuật hỗ trợ:**
EarlyStopping (patience=3) tự động dừng nếu val_loss không giảm, tránh overfitting. ReduceLROnPlateau chỉ dùng ở Phase 1 và Phase 3 (Phase 2 dùng Cosine LR nên không cần). ModelCheckpoint lưu weights tốt nhất. Giải phóng VRAM sau mỗi model để tránh OOM trên Kaggle.

---

## SLIDE 23: STEP 7 — ĐÁNH GIÁ MÔ HÌNH

**Tiêu đề:** Bước 7: Đánh giá Mô hình (Data Evaluation)

**Nội dung:**
So sánh 4 mô hình CNN trên tập Validation bằng 5 chỉ số: Accuracy, Precision, Recall, F1-Macro và F1-Weighted. **Áp dụng Test-Time Augmentation (TTA):** load best weights cho từng model, dự đoán trên 4 biến thể (gốc + 3 flip), lấy trung bình xác suất → chọn mô hình có F1-Macro cao nhất làm Best Model.

**Confusion Matrix:**
Vẽ ma trận nhầm lẫn cho cả 4 mô hình CNN, giúp nhận diện cặp lớp nào hay bị nhầm lẫn (ví dụ: Brown Spot vs Leaf Blast do triệu chứng tương tự).

**Classification Report:**
Hiển thị Precision, Recall, F1 chi tiết cho từng lớp bệnh của Best Model. Giúp xác định lớp nào model yếu nhất để cải thiện.

**Knowledge Distillation (Chưng cất Kiến thức):**
Sử dụng Best Model (CNN mạnh nhất) làm Teacher, MobileNetV3Large (CNN nhẹ) làm Student. Teacher tạo soft-label (phân phối xác suất mềm) với Temperature=4.0, Student học theo để đạt độ chính xác gần Teacher nhưng giữ kích thước nhỏ. Kỹ thuật này cho phép triển khai model nhẹ trên thiết bị di động mà không mất nhiều độ chính xác. Cả Teacher và Student đều là CNN.

---

## SLIDE 24: STEP 8 — TINH CHỈNH SIÊU THAM SỐ

**Tiêu đề:** Bước 8: Tinh chỉnh Siêu tham số (Hyper-parameter Tuning) — Phase 3

**Nội dung:**
Trong Deep Learning, thay vì GridSearchCV truyền thống, tuning được thực hiện qua Phase 3: Full Fine-tune. Tải lại Best Model CNN với weights tốt nhất từ Phase 2, mở khóa toàn bộ base layers, learning rate siêu nhỏ 1e-5 (thấp hơn 100 lần so với Phase 1). Huấn luyện thêm 5 epochs. Phase 3 dùng flat LR + ReduceLROnPlateau (không dùng CosineDecay như Phase 2). Biểu đồ training nối liền cả 3 phase với đường dọc đánh dấu: đỏ nằm dấu Phase 1→2, tím đánh dấu Phase 2→3.

**Regularization đã áp dụng:**
Dropout(0.3) tại Classifier Head ngăn overfitting. L2 regularization (1e-4) trên lớp Dense cuối cùng. EarlyStopping (patience=5) cho Phase 3 dài hơn. ReduceLROnPlateau tự động giảm LR xuống tối thiểu 1e-8. Label Smoothing (0.1) được áp dụng cho 3 model (ConvNeXtSmall, MobileNetV3, EfficientNetV2S), trong khi Proposed giữ nguyên Focal Loss.

**Ablation Study (So sánh ảnh gốc vs ảnh tách nền):**
Đánh giá Best Model CNN trên 2 bộ dữ liệu: ảnh gốc và ảnh đã tách nền (segmented). So sánh Val Accuracy để xác định xem việc tách nền có giúp CNN phân loại tốt hơn hay không.

**So sánh trước và sau Phase 3:**
Hiển thị bảng delta cho Accuracy, Precision, Recall, F1 trước và sau khi Fine-tune Phase 3 để chứng minh hiệu quả của tuning.

---

## SLIDE 25: STEP 9 — XÂY DỰNG PIPELINE SUY LUẬN

**Tiêu đề:** Bước 9: Xây dựng Pipeline với Mô hình tốt nhất

**Nội dung:**
Xây dựng class Python RiceDiseasePipeline đóng gói toàn bộ quy trình: load ảnh, tiền xử lý (resize 224×224, giữ pixel [0,255] float32), dự đoán bằng mô hình CNN tốt nhất, hiển thị Top-3 dự đoán kèm thanh confidence. Pipeline đảm bảo tiền xử lý đồng nhất giữa training và inference.

**Ensemble — Soft-voting 4 mô hình CNN:**
Kết hợp output probabilities của cả 4 mô hình CNN bằng phương pháp Soft-voting: lấy trung bình xác suất dự đoán từ ConvNeXtSmall, MobileNetV3Large, EfficientNetV2S và Proposed_ConvNeXtTiny_SE. Lớp được chọn là lớp có xác suất trung bình cao nhất. Ensemble thường cho kết quả cao hơn bất kỳ model đơn lẻ nào vì các model CNN khác nhau mắc lỗi ở các mẫu khác nhau.

---

## SLIDE 26: STEP 10 — MỞ KHÓA TEST SET

**Tiêu đề:** Bước 10: Kết luận — Mở khóa Test Set

**Nội dung:**
Đây là lần đầu tiên và duy nhất sử dụng Test Set. Tập Test đã hoàn toàn không được dùng từ Step 6 đến Step 9, đảm bảo đánh giá trung thực.

**Đánh giá đồng loạt 4 mô hình CNN trên Test:**
Tải lại lần lượt 4 mô hình CNN đã lưu, dự đoán trên Test Set, tính 6 chỉ số: Accuracy, Precision, Recall, F1-Macro, F1-Weighted và Cohen's Kappa. Vẽ biểu đồ so sánh hiệu suất.

**Đánh giá chi tiết Best Model (Đã qua Phase 3):**
Hiển thị Classification Report đầy đủ, Confusion Matrix cuối cùng, đồ thị ROC-AUC đa lớp (Multi-class ROC Curve) cho Best Model.

**Thông số phần cứng:**
So sánh 4 mô hình CNN về: Số tham số (Millions), Kích thước Model trên đĩa (MB), Tốc độ Inference (FPS).

---

## SLIDE 27: STEP 10 — GRAD-CAM

**Tiêu đề:** Grad-CAM — Giải thích Quyết định của CNN

**Nội dung:**
Grad-CAM (Gradient-weighted Class Activation Mapping) là kỹ thuật trực quan hóa cho phép nhìn thấy vùng nào trên ảnh mà CNN tập trung nhất khi đưa ra quyết định phân loại. Kỹ thuật này chỉ hoạt động với CNN (cần lớp Conv2D cuối cùng).

**Cách hoạt động:**
Lấy gradient của lớp dự đoán theo feature map của lớp Conv2D cuối cùng trong CNN. Tính trung bình gradient theo chiều không gian để được trọng số mỗi kênh. Nhân feature map với trọng số, tổng hợp thành heatmap. Áp dụng ReLU chỉ giữ vùng ảnh hưởng tích cực. Overlay heatmap lên ảnh gốc với colormap jet.

**Kết quả kỳ vọng:**
Vùng nóng (đỏ/vàng) nên trùng với vị trí vết bệnh trên lá. Nếu CNN tập trung đúng vào vết bệnh, chứng tỏ model học đúng đặc trưng. Nếu CNN tập trung vào nền, chứng tỏ có data leakage hoặc model học sai.

---

## SLIDE 28: TỔNG KẾT DỰ ÁN

**Tiêu đề:** Tổng kết 10 bước — Toàn bộ sử dụng kiến trúc CNN

**Tóm tắt kiến trúc:**
Toàn bộ dự án sử dụng CNN (Convolutional Neural Network). Không sử dụng RNN, LSTM hay GRU vì bài toán phân loại ảnh không có tính tuần tự (sequential). CNN phù hợp hoàn hảo cho bài toán ảnh nhờ khả năng trích xuất đặc trưng không gian (spatial features) qua các bộ lọc tích chập.

**Bảng tóm tắt 10 Step:**

Step 1 Problem Understanding: Xác định bài toán, 7 lớp, 4 mô hình CNN.
Step 2 Data Understanding: Gộp 4 dataset, chuẩn hóa, loại ảnh hỏng/trùng.
Step 3 Feature Understanding: EDA đơn biến, nhị biến, đa biến.
Step 4 Feature Engineering: Label Encoding, Augmentation, CLAHE, Tách nền.
Step 5 Dataset Partition: Chia 70-15-15, DCGAN (CNN, **5000 steps**) sinh ảnh cân bằng tập Train đạt 10.000 ảnh.
Step 6 Data Modelling: 4 CNN Transfer Learning, Focal Loss, SE-Block, **Label Smoothing, Cosine LR**.
Step 7 Data Evaluation: So sánh 4 CNN, **TTA**, Confusion Matrix, Knowledge Distillation.
Step 8 Hyper-parameter Tuning: Phase 3 Full Fine-tune, Ablation Study.
Step 9 Inference Pipeline: Pipeline dự đoán + Ensemble 4 CNN.
Step 10 Conclusion: Test evaluation, ROC-AUC, Grad-CAM, Tổng kết.

**Công nghệ CNN chính:**
ConvNeXtSmall/Tiny (SOTA 2022), MobileNetV3Large (Mobile), EfficientNetV2S (Efficient), SE-Block (Channel Attention), DCGAN (Data Synthesis), Grad-CAM (Interpretability), **Label Smoothing, Cosine Annealing LR, TTA**.

---
