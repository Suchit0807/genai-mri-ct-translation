import os
import numpy as np
import pydicom
import imageio
from glob import glob
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, imsave
import albumentations as A
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
RAW_DATA_DIR = "data/raw/JUH_MR-CT_dataset"
AUGMENTED_DATA_DIR = "data/processed/JUH_MR-CT_augmented_dataset"
CROPPED_DATA_DIR = "data/processed/JUH_MR-CT_cropped_augmented_dataset"

NUM_AUGMENTS = 20
CROP_WIDTH, CROP_HEIGHT = 224, 224

# -----------------------------
# AUGMENTATION PIPELINE
# -----------------------------
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# -----------------------------
# IMAGE PROCESSING FUNCTIONS
# -----------------------------
def load_dicom_image(filepath):
    ds = pydicom.dcmread(filepath, force=True)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    img = ds.pixel_array
    return img, ds

def convert_to_grayscale(image):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    return image

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def resize_image(image, target_size=(256, 256)):
    return resize(image, target_size, anti_aliasing=True, mode='reflect')

def augment_images_and_masks(image_paths, mask_paths, output_image_folder, output_mask_folder, num_augments=1):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    for image_path, mask_path in zip(image_paths, mask_paths):
        image, _ = load_dicom_image(image_path)
        mask, _ = load_dicom_image(mask_path)

        image = normalize_image(convert_to_grayscale(image))
        mask = normalize_image(convert_to_grayscale(mask))
        image = resize_image(image)
        mask = resize_image(mask)

        for i in range(num_augments):
            augmented = augmentation_pipeline(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']

            base_image_filename = os.path.basename(image_path).replace('.dcm', f'_aug_{i}.png')
            base_mask_filename = os.path.basename(mask_path).replace('.dcm', f'_aug_{i}.png')

            imageio.imwrite(os.path.join(output_image_folder, base_image_filename), (augmented_image * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(output_mask_folder, base_mask_filename), (augmented_mask * 255).astype(np.uint8))

def crop_image(image, crop_width, crop_height):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    start_x = center_x - (crop_width // 2)
    start_y = center_y - (crop_height // 2)
    return image[start_y:start_y+crop_height, start_x:start_x+crop_width]

def process_and_save_images(image_paths, output_folder, crop_width, crop_height):
    os.makedirs(output_folder, exist_ok=True)
    for image_path in image_paths:
        image = imread(image_path, as_gray=True)
        image = normalize_image(image)
        image = crop_image(image, crop_width, crop_height)

        base_filename = os.path.basename(image_path).replace('.png', '_cropped.png')
        imsave(os.path.join(output_folder, base_filename), (image * 255).astype(np.uint8))

def visualize_images_and_masks(image_paths, mask_paths, num_samples=5):
    plt.figure(figsize=(10, num_samples * 2))
    for i in range(num_samples):
        image = imread(image_paths[i], as_gray=True)
        mask = imread(mask_paths[i], as_gray=True)

        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Image: {os.path.basename(image_paths[i])}')
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask: {os.path.basename(mask_paths[i])}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# -----------------------------
# MAIN EXECUTION PIPELINE
# -----------------------------
if __name__ == "__main__":
    # Paths to datasets
    mr_image_paths = sorted(glob(os.path.join(RAW_DATA_DIR, 'MR/image_MR/*.dcm')))
    mr_mask_paths = sorted(glob(os.path.join(RAW_DATA_DIR, 'MR/mask_MR/*.dcm')))
    ct_image_paths = sorted(glob(os.path.join(RAW_DATA_DIR, 'CT/image_CT/*.dcm')))
    ct_mask_paths = sorted(glob(os.path.join(RAW_DATA_DIR, 'CT/mask_CT/*.dcm')))

    # Augmentation output paths
    output_mr_image_folder = os.path.join(AUGMENTED_DATA_DIR, 'MR/augmented_images')
    output_mr_mask_folder = os.path.join(AUGMENTED_DATA_DIR, 'MR/augmented_masks')
    output_ct_image_folder = os.path.join(AUGMENTED_DATA_DIR, 'CT/augmented_images')
    output_ct_mask_folder = os.path.join(AUGMENTED_DATA_DIR, 'CT/augmented_masks')

    # Perform augmentations
    augment_images_and_masks(mr_image_paths, mr_mask_paths, output_mr_image_folder, output_mr_mask_folder, num_augments=NUM_AUGMENTS)
    augment_images_and_masks(ct_image_paths, ct_mask_paths, output_ct_image_folder, output_ct_mask_folder, num_augments=NUM_AUGMENTS)

    # Cropping output paths
    output_mr_image_folder_cropped = os.path.join(CROPPED_DATA_DIR, 'MR/cropped_augmented_images')
    output_mr_mask_folder_cropped = os.path.join(CROPPED_DATA_DIR, 'MR/cropped_augmented_masks')
    output_ct_image_folder_cropped = os.path.join(CROPPED_DATA_DIR, 'CT/cropped_augmented_images')
    output_ct_mask_folder_cropped = os.path.join(CROPPED_DATA_DIR, 'CT/cropped_augmented_masks')

    # Crop augmented images and masks
    process_and_save_images(glob(os.path.join(output_mr_image_folder, '*.png')), output_mr_image_folder_cropped, CROP_WIDTH, CROP_HEIGHT)
    process_and_save_images(glob(os.path.join(output_mr_mask_folder, '*.png')), output_mr_mask_folder_cropped, CROP_WIDTH, CROP_HEIGHT)
    process_and_save_images(glob(os.path.join(output_ct_image_folder, '*.png')), output_ct_image_folder_cropped, CROP_WIDTH, CROP_HEIGHT)
    process_and_save_images(glob(os.path.join(output_ct_mask_folder, '*.png')), output_ct_mask_folder_cropped, CROP_WIDTH, CROP_HEIGHT)

    # Optional visualization
    visualize_images_and_masks(
        sorted(glob(os.path.join(output_mr_image_folder_cropped, '*.png'))),
        sorted(glob(os.path.join(output_mr_mask_folder_cropped, '*.png'))),
        num_samples=5
    )

    visualize_images_and_masks(
        sorted(glob(os.path.join(output_ct_image_folder_cropped, '*.png'))),
        sorted(glob(os.path.join(output_ct_mask_folder_cropped, '*.png'))),
        num_samples=5
    )

