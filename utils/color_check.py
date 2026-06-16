import os
import shutil
import numpy as np
from PIL import Image
import cv2

def is_grayscale_saturation(path):
    img = Image.open(path).convert("RGB")
    image = np.array(img)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:,:,1]
    
    # Low mean saturation indicates grayscale
    mean_saturation = np.mean(saturation)
    threshold = 10  # Adjust based on your data (0-255 scale)
    return mean_saturation < threshold

def is_grayscale_std(path):
    img = Image.open(path).convert("RGB")

    arr = np.array(img)

    # Calculate std deviation between channels
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    std_rg = np.std(r - g)
    std_rb = np.std(r - b)
    std_gb = np.std(g - b)
    
    # Low std indicates grayscale
    threshold = 3.0  # Adjust based on your data
    return (std_rg + std_rb + std_gb) / 3 < threshold 

def color_correlation(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    corr_rg = np.corrcoef(r, g)[0,1]
    corr_rb = np.corrcoef(r, b)[0,1]
    corr_gb = np.corrcoef(g, b)[0,1]

    return (corr_rg + corr_rb + corr_gb) / 3

def color_variance(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)

    return (var_r + var_g + var_b) / 3

def check_color_diff(path):
    img = Image.open(path).convert("RGB")

    arr = np.array(img)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    dff_rg = np.median(np.abs(r - g))
    dff_rb = np.median(np.abs(r - b))
    dff_gb = np.median(np.abs(g - b))

    mean_diff = np.max([dff_rg, dff_rb, dff_gb])

    # 255 is for extreme cases, night time photos usually have diff of 0
    if mean_diff < 3 or mean_diff == 255:
        # if datum.aid in [14, 8]:
        #     print(f"aid: {datum.aid}, img_path: {datum.img_path}, 'night', mean_dff: {mean_diff}")
        return False
    else:
        # if datum.aid in [14, 8]:
        #     print(f"aid: {datum.aid}, img_path: {datum.img_path}, 'day', mean_dff: {mean_diff}")
        return True

def classify_image(img_path, corr_thresh=0.99, color_thresh=15):
    """
    Classify image as grayscale or color.
    
    Args:
        img_path (str): Path to image.
        corr_thresh (float): Correlation threshold for grayscale detection.
        color_thresh (float): Colorfulness threshold for dull vs color.
    
    Returns:
        str: "grayscale" or "color"
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32)

    r = arr[:,:,0].flatten()
    g = arr[:,:,1].flatten()
    b = arr[:,:,2].flatten()

    # --- Step 1: Channel Correlation ---
    corr_rg = np.corrcoef(r, g)[0,1]
    corr_rb = np.corrcoef(r, b)[0,1]
    corr_gb = np.corrcoef(g, b)[0,1]
    mean_corr = (corr_rg + corr_rb + corr_gb) / 3

    if mean_corr >= corr_thresh:
        return "grayscale"

    # --- Step 2: Colorfulness Metric ---
    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)

    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    if colorfulness < color_thresh:
        return "grayscale"
    else:
        return "color"
