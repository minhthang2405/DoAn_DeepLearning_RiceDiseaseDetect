"""Grad-CAM visualization for 4 models."""
import os, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Custom loss for Proposed model compatibility
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha; self.gamma = gamma
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

IMG_SIZE = 224
CLASS_NAMES = ['Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy', 'Hispa',
               'Leaf_Blast', 'Leaf_Scald', 'Sheath_Blight']

SAMPLE_DIR = r"D:\DataDownloaded\Study\MonHoc\Do_An\deeplearning\CK2\sample_images"
sample_paths = []
for f in os.listdir(SAMPLE_DIR):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        sample_paths.append(os.path.join(SAMPLE_DIR, f))
sample_paths = sample_paths[:3]

print(f"Using {len(sample_paths)} sample images: {[os.path.basename(p) for p in sample_paths]}")

MODELS = [
    ("ConvNeXtSmall", "ConvNeXtSmall_best.keras"),
    ("MobileNetV3L", "MobileNetV3Large_best.keras"),
    ("EfficientNetV2S", "EfficientNetV2S_best.keras"),
    ("Proposed_SE", "Proposed_ConvNeXtTiny_SE_best_tuned.keras"),
]

def make_gradcam(model, img_array):
    last_conv = None
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    def _is_4d_conv(layer):
        return isinstance(layer, conv_types)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sl in reversed(layer.layers):
                if _is_4d_conv(sl):
                    last_conv = sl; break
            if last_conv: break
        elif _is_4d_conv(layer):
            last_conv = layer; break
    if last_conv is None:
        return None
    try:
        grad_model = tf.keras.models.Model(inputs=model.inputs,
                                           outputs=[last_conv.output, model.output])
    except Exception:
        return None
    with tf.GradientTape() as tape:
        try:
            conv_out, preds = grad_model(img_array)
        except Exception:
            return None
        pred_idx = tf.argmax(preds[0])
        class_ch = preds[:, pred_idx]
    grads = tape.gradient(class_ch, conv_out)
    if grads is None: return None
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), CLASS_NAMES[pred_idx], float(np.max(preds))

n_samples = len(sample_paths)
cols = 1 + len(MODELS)

fig, axes = plt.subplots(n_samples, cols, figsize=(4 * cols, 4 * n_samples))
if n_samples == 1: axes = axes[np.newaxis, :]
fig.suptitle('So sánh Grad-CAM Giải thích 4 Mô hình (Overlay Heatmap)', fontsize=16, fontweight='bold', y=1.02)

# Col 0: Original images
for i, img_path in enumerate(sample_paths):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr_255 = img_to_array(img).astype(np.float32)
    arr_vis = np.clip(arr_255 / 255.0, 0.0, 1.0)
    axes[i, 0].imshow(arr_vis)
    axes[i, 0].set_title('Hình Gốc (Original)', fontweight='bold', fontsize=12)
    axes[i, 0].axis('off')

# Iterate models and do Cols 1 to 4
for col_idx, (m_name, m_file) in enumerate(MODELS):
    real_col = col_idx + 1
    m_path = os.path.join(r"D:\DataDownloaded\Study\MonHoc\Do_An\deeplearning\CK2\model", m_file)
    print(f"\nLoading {m_name}...")
    try:
        model = tf.keras.models.load_model(m_path, compile=False, safe_mode=False,
                                           custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss})
        
        for i, img_path in enumerate(sample_paths):
            print(f"  {m_name} -> img {i+1}")
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            arr_255 = img_to_array(img).astype(np.float32)
            tensor = np.expand_dims(arr_255, 0)
            
            res = make_gradcam(model, tensor)
            
            arr_vis = np.clip(arr_255 / 255.0, 0.0, 1.0)
            axes[i, real_col].imshow(arr_vis)
            
            if res is not None:
                heatmap, pred_cls, conf = res
                hm = tf.image.resize(heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE), method='bicubic').numpy()[:, :, 0]
                axes[i, real_col].imshow(hm, cmap='jet', alpha=0.5, interpolation='bilinear')
                
                title = f"{m_name}\n{pred_cls} ({conf:.1%})"
                color = 'green' if conf > 0.7 else 'orange'
            else:
                title = f"{m_name}\n(Lỗi tạo Heatmap)"
                color = 'red'
                
            axes[i, real_col].set_title(title, fontweight='bold', fontsize=10, color=color)
            axes[i, real_col].axis('off')
            
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"Failed loading {m_name}: {e}")
        for i in range(n_samples):
            axes[i, real_col].text(0.5, 0.5, 'Error\nLoading', ha='center', va='center')
            axes[i, real_col].set_title(m_name, fontweight='bold')
            axes[i, real_col].axis('off')

plt.tight_layout()
out_path = r"D:\DataDownloaded\Study\MonHoc\Do_An\deeplearning\CK2\GradCAM_4Models_Compare.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✅ Đã lưu ảnh tổng hợp: {out_path}")
