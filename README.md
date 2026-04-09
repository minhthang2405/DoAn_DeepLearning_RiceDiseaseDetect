#  Rice Leaf Disease Prediction — Advanced Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU_T4%C3%972-20BEFF.svg)](https://www.kaggle.com/)

> **Rice Leaf Disease Classification using Deep Learning (10-Step ML Pipeline)**  
> 4 CNN Models · Ensemble Soft-Voting · Grad-CAM Explainability · Dual GPU T4×2

---

##  Overview

This project implements an automated, highly accurate classification system for **7 types of rice leaf diseases** from field images. It utilizes a comprehensive 10-step deep learning pipeline, integrating state-of-the-art Computer Vision models, advanced data augmentation techniques, and custom attention mechanisms (CBAM/SE-Blocks), optimized for distributed training on Kaggle's **Dual GPU T4×2** architecture.

**Supported Classes (7):**
*Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Sheath Blight, Hispa, Healthy*

---

##  Datasets

The dataset is a consolidated and curated collection merged from **4 different Kaggle datasets**, resulting in over **7,700 high-quality images** after strict MD5 hash deduplication and corruption filtering.

| # | Source Dataset | Link |
|---|----------------|------|
| 1 | Rice Disease (thaonguyen0712) | [ Kaggle](https://www.kaggle.com/datasets/thaonguyen0712/rice-disease) |
| 2 | Rice Disease (nurnob101) | [ Kaggle](https://www.kaggle.com/datasets/nurnob101/rice-disease) |
| 3 | Rice Disease (jonathanrjpereira) | [ Kaggle](https://www.kaggle.com/datasets/jonathanrjpereira/rice-disease) |
| 4 | Rice Disease Dataset (anshulm257) | [ Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset) |

---

##  10-Step Deep Learning Pipeline

We have designed a professional-grade notebook pipeline capable of running seamlessly on Kaggle:

1. **Problem Definition:** Multi-class classification (7 classes).
2. **Data Collection & Cleaning:** MD5 image deduplication and `tf.data` pipeline setup.
3. **Exploratory Data Analysis (EDA):** BoxPlots, class distribution, and RGB channel analysis.
4. **Advanced Augmentation:** CLAHE preprocessing, CutMix, and MixUp techniques.
5. **Handling Imbalance:** Leveraging class weights and augmentation for minority classes.
6. **Model Initialization:** Loading pre-trained SOTA models via Hugging Face/Keras (EfficientNetV2, ConvNeXt, MobileNetV3) within `tf.distribute.MirroredStrategy`.
7. **Phase 1 Training (Warm-up):** Training the top classification head with a frozen base.
8. **Phase 2 & 3 Training (Fine-tuning):** Gradual unfreezing with Cosine Decay Learning Rate and Focal Loss.
9. **Ensemble & Inference:** Soft-voting probability averaging across 4 models for robustness.
10. **Evaluation & Explainability:** Confusion Matrix, F1-Score, Cohen's Kappa, and **Grad-CAM** Heatmaps.

---

##  Models & Training Strategy

The project compares multiple architectures against a newly proposed custom model:

1. **ConvNeXtSmall:** Pure ConvNet architecture competing with Vision Transformers.
2. **EfficientNetV2S:** Balanced compound scaling for accuracy and speed.
3. **MobileNetV3Large:** Lightweight, mobile-optimized architecture.
4. **Proposed Custom Architecture:** Featuring a highly optimized backbone (EfficientNetV2 / ConvNeXt) combined with an **Attention Mechanism (CBAM / SE-Block)** and **Focal Loss (γ=2.0)** to tackle hard-to-classify samples.

###  3-Phase Transfer Learning
* **Phase 1 (Warmup):** `LR = 1e-3` — Base model frozen, train standard dense head.
* **Phase 2 (Partial Unfreeze):** `LR = 1e-4` — Unfreeze the top 30% of layers, apply Cosine Annealing.
* **Phase 3 (Full Fine-tune):** `LR = 1e-5` — Unfreeze 100% of the model to align weights precisely with the specific domain of rice diseases.

---

##  Test Set Results

Our best proposed ensemble/model achieves exceptional performance metrics on the test set, demonstrating robust applicability in real-world agricultural scenarios:

| Metric | Proposed Custom Model | EfficientNetV2S | ConvNeXtSmall | MobileNetV3Large |
|--------|-----------------------|-----------------|---------------|------------------|
| **Accuracy** | **89.5%** | 88.8% | 88.6% | 85.7% |
| **Precision** | **88.2%** | 87.1% | 86.9% | 84.5% |
| **Recall** | **88.4%** | 87.8% | 87.2% | 85.1% |
| **F1-Macro** | **88.3%** | 87.8% | 87.3% | 86.0% |
| **Kappa** | **0.875** | 0.866 | 0.862 | 0.829 |

>  **Insight:** The integration of the Attention mechanisms (CBAM/SE) significantly improves differentiation between visually similar disease states (like *Hispa* and *Early Brown Spot*), yielding a Cohen's Kappa score indicating "Almost Perfect Agreement".

---

##  Visual Explainability (Grad-CAM)

To build trust in the AI's diagnostic decisions, we leverage **Grad-CAM** (Gradient-weighted Class Activation Mapping). 
The model demonstrates exactly *where* it is looking when diagnosing a disease. For instance, the activation heatmaps successfully localize the distinctive elongated lesions of Bacterial Leaf Blight and the diamond-shaped necrotic patches of Leaf Blast. 

---

##  Quick Start & Setup

**Requirements:**
- Python 3.9+
- TensorFlow 2.12+ (or Keras 3.0)
- Kaggle Dual GPU T4×2 (recommended for training)

**Running the Code:**
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `rice_disease_prediction_t4x2.ipynb` in local Jupyter or upload it to Kaggle.
4. Ensure the Kaggle environment accelerator is set to **GPU T4×2**.
5. Run the notebook from Step 1 to Step 10. All data downloading, preprocessing, and training will execute automatically.

---

##  Repository Structure

```text
├──  rice_disease_prediction_t4x2.ipynb   ← Main training pipeline notebook (GPU T4x2)
├──  results/                             ← Saved metrics, charts, & Grad-CAM outputs
├──  sample_images/                       ← Example images for quick inference
├──  README.md                            ← Project documentation (this file)
└──  .gitignore                           ← Ignores model weights (.keras) & raw data
```
*(Note: Trained `.keras` weights and raw Kaggle datasets are excluded from this repository to save space. They will be generated/downloaded at runtime).*
