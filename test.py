import tifffile as tiff
import imageio.v3 as iio
from PIL import Image
from pathlib import Path

# --- CHANGE THIS PATH TO MATCH THE FILE PATH SHOWN IN YOUR ERROR ---
FILE_PATH = Path('/Users/joebrashear/Desktop/moffitt hackathon /hackathon/train/images/P0202.tif')
# ------------------------------------------------------------------

print(f"Testing file: {FILE_PATH.name}")
print("-" * 30)

# 1. Test Tifffile (Most robust, array-based)
try:
    arr_tiff = tiff.asarray(str(FILE_PATH))
    print(f"[TIF] SUCCESS: Loaded array with shape {arr_tiff.shape} and dtype {arr_tiff.dtype}")
except Exception as e:
    print(f"[TIF] FAILED: {type(e).__name__}: {e}")

# 2. Test ImageIO
try:
    arr_iio = iio.imread(str(FILE_PATH))
    print(f"[IIO] SUCCESS: Loaded array with shape {arr_iio.shape} and dtype {arr_iio.dtype}")
except Exception as e:
    print(f"[IIO] FAILED: {type(e).__name__}: {e}")

# 3. Test PIL (The current failing fallback)
try:
    img_pil = Image.open(FILE_PATH)
    print(f"[PIL] SUCCESS: Loaded image with size {img_pil.size}")
except Exception as e:
    print(f"[PIL] FAILED (Expected): {type(e).__name__}: {e}")