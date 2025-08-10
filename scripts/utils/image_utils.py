# scripts/utils/image_utils.py
"""
Minimal image I/O utilities needed:
- Converting to grayscale
- Resizing to 256x256
- Normalizing to [0, 1]
- Adding channel dimension (H, W, 1)
- Simple binary mask conversion

"""

from pathlib import Path
import numpy as np
from PIL import Image

# pydicom is optional (only needed for .dcm files)
try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False


# ---------- basic helpers ----------

def normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1]."""
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def add_channel(arr: np.ndarray) -> np.ndarray:
    """(H, W) -> (H, W, 1)."""
    if arr.ndim == 2:
        return arr[..., None]
    return arr


def binarize_mask(arr: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Binary mask (0/1) using threshold in [0,1]."""
    return (arr >= thr).astype(np.float32)


# ---------- PNG/JPG ----------

def read_png_gray(path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Read image file, convert to grayscale, resize, normalize to [0,1]."""
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def save_png_gray(arr: np.ndarray, path: str | Path) -> None:
    """Save a float32 array in [0,1] as an 8-bit grayscale PNG."""
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ---------- DICOM ----------

def read_dicom_gray(path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Read DICOM, take the single channel if needed, resize, normalize to [0,1].
    """
    if not _HAS_PYDICOM:
        raise ImportError("pydicom is not installed. Install it to read DICOM files.")
    d = pydicom.dcmread(str(path))
    arr = d.pixel_array
    if arr.ndim == 3:       # some DICOMs come with an extra channel/stack
        arr = arr[..., 0]
    img = Image.fromarray(arr).resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return normalize01(arr)


# ---------- unified loaders ----------

def load_gray(path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Unified loader:
    - .png/.jpg/.jpeg -> read_png_gray
    - .dcm -> read_dicom_gray
    Returns float32 array in [0,1], shape (H, W).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        return read_png_gray(path, size)
    if ext in [".dcm", ".dicom"]:
        return read_dicom_gray(path, size)
    raise ValueError(f"Unsupported file extension: {ext}")


# ---------- mask-specific helpers ----------

def load_mask(path: str | Path, size: tuple[int, int] = (256, 256), thr: float = 0.5) -> np.ndarray:
    """
    Load a mask file (PNG/JPG/DICOM), grayscale, resize, normalize to [0,1],
    then binarize to {0,1}.
    """
    arr = load_gray(path, size)
    return binarize_mask(arr, thr)

