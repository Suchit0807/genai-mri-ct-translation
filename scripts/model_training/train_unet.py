# -----------------------------------------------------------
# Training two separate U-Net models for brain tumor segmentation:
#   1. MRI images + masks
#   2. CT images + masks
# Dataset: Preprocessed & cropped augmented datasets from data_preprocessing/preprocess_data.py
# -----------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import to_categorical
from skimage.io import imread

# -----------------------------------------------------------
# U-Net model definition
# -----------------------------------------------------------
def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3,3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3,3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# -----------------------------------------------------------
# Utility function to load images & masks
# -----------------------------------------------------------
def load_dataset(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    images = np.array([imread(img, as_gray=True) for img in image_paths])
    masks = np.array([imread(msk, as_gray=True) for msk in mask_paths])

    # Expand dimensions for channels-last format
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    # Normalize to [0,1]
    images = images / 255.0
    masks = masks / 255.0

    return images, masks

# -----------------------------------------------------------
# Training function for a given modality
# -----------------------------------------------------------
def train_and_save_model(modality, image_dir, mask_dir, save_dir):
    print(f"\n[INFO] Training U-Net for {modality}...")

    # Load data
    images, masks = load_dataset(image_dir, mask_dir)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Build model
    model = build_unet(input_shape=(images.shape[1], images.shape[2], 1))
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy", MeanIoU(num_classes=2)])

    # Callbacks for saving best model
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{modality}_unet_best.h5")

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=8,
        epochs=50,
        callbacks=callbacks
    )

    # Save training history plot
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{modality} U-Net Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{modality}_training_loss.png"))
    plt.close()

    print(f"[INFO] {modality} training complete. Model saved at {checkpoint_path}")

# -----------------------------------------------------------
# Paths (update these to your /data/processed structure)
# -----------------------------------------------------------
base_data_dir = "data/processed"
results_dir = "results/unet"

# MRI paths
mri_image_dir = os.path.join(base_data_dir, "MRI/images")
mri_mask_dir = os.path.join(base_data_dir, "MRI/masks")

# CT paths
ct_image_dir = os.path.join(base_data_dir, "CT/images")
ct_mask_dir = os.path.join(base_data_dir, "CT/masks")

# -----------------------------------------------------------
# Run training for both modalities
# -----------------------------------------------------------
train_and_save_model("MRI", mri_image_dir, mri_mask_dir, os.path.join(results_dir, "MRI"))
train_and_save_model("CT", ct_image_dir, ct_mask_dir, os.path.join(results_dir, "CT"))

