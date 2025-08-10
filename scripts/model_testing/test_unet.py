# Evaluates trained U-Net models (MRI and CT) on both own and cross-modality test sets.
# Dataset produced by: scripts/data_preprocessing/preprocess.py

import os
import numpy as np
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from skimage.io import imread
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU

# ---------------------------
# Data loading
# ---------------------------
def load_dataset(image_dir, mask_dir):
    img_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    msk_paths = sorted(glob(os.path.join(mask_dir,  "*.png")))
    X = np.array([imread(p, as_gray=True) for p in img_paths], dtype=np.float32) / 255.0
    Y = np.array([imread(p, as_gray=True) for p in msk_paths], dtype=np.float32) / 255.0
    X = X[..., None]
    Y = (Y > 0.5).astype(np.float32)[..., None]
    return X, Y, img_paths

# ---------------------------
# Metrics
# ---------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# ---------------------------
# Prediction visualisation
# ---------------------------
def save_predictions_grid(model, X, Y, img_paths, out_dir, num_samples=5):
    os.makedirs(out_dir, exist_ok=True)
    idxs = np.random.choice(len(X), size=num_samples, replace=False)

    for idx in idxs:
        img = X[idx]
        mask_true = Y[idx]
        pred = model.predict(np.expand_dims(img, axis=0))[0]
        pred_bin = (pred > 0.5).astype(np.float32)

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_true.squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_bin.squeeze(), cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        fname = Path(img_paths[idx]).stem + "_pred.png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

# ---------------------------
# Evaluation function
# ---------------------------
def evaluate_model(modality_name, model_path, img_dir, msk_dir, results_root):
    # Load data
    X, Y, img_paths = load_dataset(img_dir, msk_dir)
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", MeanIoU(num_classes=2)])

    # Evaluate
    loss, acc, iou = model.evaluate(X_test, Y_test, verbose=1)
    dice = dice_coefficient(Y_test, (model.predict(X_test) > 0.5).astype(np.float32))

    # Save metrics
    out_dir = Path(results_root) / modality_name / "predictions"
    os.makedirs(out_dir, exist_ok=True)
    with open(Path(results_root) / modality_name / "test_metrics.txt", "w") as f:
        f.write(f"Loss: {loss:.4f}\nAccuracy: {acc:.4f}\nIoU: {iou:.4f}\nDice: {dice:.4f}\n")

    # Save prediction images
    save_predictions_grid(model, X_test, Y_test, img_paths, out_dir, num_samples=5)

    print(f"[OK] {modality_name} evaluation complete. Metrics saved, predictions stored in {out_dir}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    ROOT = "data/processed/JUH_MR-CT_cropped_augmented_dataset"

    MRI_IMG_DIR = os.path.join(ROOT, "MR/cropped_augmented_images")
    MRI_MSK_DIR = os.path.join(ROOT, "MR/cropped_augmented_masks")
    CT_IMG_DIR  = os.path.join(ROOT, "CT/cropped_augmented_images")
    CT_MSK_DIR  = os.path.join(ROOT, "CT/cropped_augmented_masks")

    RESULTS_ROOT = "results/unet"

    # Model paths from training output
    MRI_MODEL_PATH = os.path.join(RESULTS_ROOT, "MR/unet_model_mr_final.h5")
    CT_MODEL_PATH  = os.path.join(RESULTS_ROOT, "CT/unet_model_ct_final.h5")

    # ---- Own modality testing ----
    evaluate_model("MR_on_MR", MRI_MODEL_PATH, MRI_IMG_DIR, MRI_MSK_DIR, results_root=RESULTS_ROOT)
    evaluate_model("CT_on_CT", CT_MODEL_PATH, CT_IMG_DIR, CT_MSK_DIR, results_root=RESULTS_ROOT)

    # ---- Cross-modality testing ----
    evaluate_model("MR_on_CT", MRI_MODEL_PATH, CT_IMG_DIR, CT_MSK_DIR, results_root=RESULTS_ROOT)
    evaluate_model("CT_on_MR", CT_MODEL_PATH, MRI_IMG_DIR, MRI_MSK_DIR, results_root=RESULTS_ROOT)
