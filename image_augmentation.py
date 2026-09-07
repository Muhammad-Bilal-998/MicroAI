import cv2
import numpy as np
import os
import random
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"C:\Users\arif.arshad\Desktop\uglify_test"
OUTPUT_DIR = r"C:\Users\arif.arshad\Desktop\uglify_test_result"

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 720

# Choose:
# "mild"
# "realistic"
# "severe"

LEVEL = "severe"

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# 1. CIRCULAR MICROSCOPE FIELD OF VIEW
# ============================================================

def add_microscope_circle(img):

    h, w = img.shape[:2]

    size = min(h, w)

    img = cv2.resize(img, (size, size))

    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    center = (
        w//2 + np.random.randint(-5, 5),
        h//2 + np.random.randint(-5, 5)
    )

    radius = int(
        min(w, h) *
        np.random.uniform(0.43, 0.48)
    )

    cv2.circle(
        mask,
        center,
        radius,
        255,
        -1
    )

    result = np.zeros_like(img)

    result[mask == 255] = img[mask == 255]

    return result


# ============================================================
# 2. COLOR SHIFT
# ============================================================

def color_shift(img, strength):

    img = img.astype(np.float32)

    # Slight red/pink/purple variation
    r_scale = 1.0 + np.random.uniform(-strength, strength)
    g_scale = 1.0 + np.random.uniform(-strength, strength)
    b_scale = 1.0 + np.random.uniform(-strength, strength)

    img[:, :, 2] *= r_scale
    img[:, :, 1] *= g_scale
    img[:, :, 0] *= b_scale

    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


# ============================================================
# 3. UNEVEN ILLUMINATION / VIGNETTING
# ============================================================

def uneven_illumination(img, strength):

    h, w = img.shape[:2]

    cx = w // 2 + np.random.randint(-15, 15)
    cy = h // 2 + np.random.randint(-15, 15)

    y, x = np.ogrid[:h, :w]

    distance = np.sqrt(
        ((x - cx) / w) ** 2 +
        ((y - cy) / h) ** 2
    )

    illumination = 1 - strength * distance * 2

    illumination = np.clip(
        illumination,
        0.55,
        1.15
    )

    img = img.astype(np.float32)

    img *= illumination[:, :, None]

    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# 4. CAMERA NOISE
# ============================================================

def add_noise(img, sigma):

    noise = np.random.normal(
        0,
        sigma,
        img.shape
    )

    noisy = img.astype(np.float32) + noise

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 5. BLUR
# ============================================================

def add_blur(img, kernel_size):

    return cv2.GaussianBlur(
        img,
        (kernel_size, kernel_size),
        0
    )


# ============================================================
# 6. SLIGHT MOTION BLUR
# ============================================================

def motion_blur(img, kernel_size):

    kernel = np.zeros(
        (kernel_size, kernel_size)
    )

    kernel[kernel_size // 2, :] = (
        1.0 / kernel_size
    )

    return cv2.filter2D(
        img,
        -1,
        kernel
    )


# ============================================================
# 7. JPEG COMPRESSION
# ============================================================

def jpeg_compression(img, quality):

    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality
    ]

    success, encoded = cv2.imencode(
        ".jpg",
        img,
        encode_param
    )

    if not success:
        return img

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )


# ============================================================
# 8. SLIGHT CONTRAST / BRIGHTNESS CHANGE
# ============================================================

def brightness_contrast(img, brightness, contrast):

    result = img.astype(np.float32)

    result = result * contrast + brightness

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 9. SLIGHT ROTATION / PERSPECTIVE
# ============================================================

def slight_rotation(img, max_angle):

    h, w = img.shape[:2]

    angle = np.random.uniform(
        -max_angle,
        max_angle
    )

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


# ============================================================
# 10. MAIN AUGMENTATION PIPELINE
# ============================================================

def make_mobile_microscope_image(img, level):

    # ----------------------------------------
    # PARAMETERS
    # ----------------------------------------

    if level == "mild":

        color_strength = 0.04
        illumination_strength = 0.20
        noise_sigma = 2
        blur_kernel = 3
        jpeg_quality = 80
        brightness = 0
        contrast = 1.0
        rotation = 1

    elif level == "realistic":

        color_strength = 0.08
        illumination_strength = 0.30
        noise_sigma = 3
        blur_kernel = 3
        jpeg_quality = 75
        brightness = np.random.uniform(-8, 8)
        contrast = np.random.uniform(0.90, 1.05)
        rotation = 2

    elif level == "severe":

        color_strength = 0.16
        illumination_strength = 0.50
        noise_sigma = 4
        blur_kernel = 3
        jpeg_quality = 80
        brightness = np.random.uniform(-10, 10)
        contrast = np.random.uniform(0.85, 1.00)
        rotation = 2

    else:
        raise ValueError(
            "LEVEL must be mild, realistic, or severe"
        )

    # ----------------------------------------
    # PROCESSING
    # ----------------------------------------

    # Ensure square-ish image
    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Slight rotation
    img = slight_rotation(
        img,
        rotation
    )

    # Color variation
    img = color_shift(
        img,
        color_strength
    )

    # Uneven microscope illumination
    img = uneven_illumination(
        img,
        illumination_strength
    )

    # Focus loss
    img = add_blur(
        img,
        blur_kernel
    )

    img = edge_blur(img)

    # Camera noise
    img = add_noise(
        img,
        noise_sigma
    )

    # Brightness / contrast
    img = brightness_contrast(
        img,
        brightness,
        contrast
    )

    # JPEG compression
    img = jpeg_compression(
        img,
        jpeg_quality
    )

    # Microscope circular FOV
    img = add_microscope_circle(
        img
    )

    return img


def edge_blur(img):

    blurred = cv2.GaussianBlur(
        img,
        (11,11),
        0
    )

    h, w = img.shape[:2]

    y, x = np.ogrid[:h, :w]

    center_x = w // 2
    center_y = h // 2

    distance = np.sqrt(
        (x-center_x)**2 +
        (y-center_y)**2
    )

    max_dist = np.sqrt(
        center_x**2 +
        center_y**2
    )

    mask = (distance / max_dist)

    mask = np.clip(mask, 0, 1)

    mask = mask[:,:,None]

    result = (
        img * (1-mask*0.5) +
        blurred * (mask*0.5)
    )

    return result.astype(np.uint8)


# ============================================================
# 11. PROCESS ENTIRE TEST DATASET
# ============================================================

classes = [
    "colon_aca",
    "colon_n",
    "lung_aca",
    "lung_n",
    "lung_scc"
]

total = 0

for class_name in classes:

    input_class_dir = os.path.join(
        INPUT_DIR,
        class_name
    )

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        output_class_dir,
        exist_ok=True
    )

    files = [
        f for f in os.listdir(input_class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(
        f"{class_name}: {len(files)} images"
    )

    for filename in files:

        input_path = os.path.join(
            input_class_dir,
            filename
        )

        output_path = os.path.join(
            output_class_dir,
            filename
        )

        img = cv2.imread(input_path)

        if img is None:
            print(
                "Could not read:",
                input_path
            )
            continue

        augmented = make_mobile_microscope_image(
            img,
            LEVEL
        )

        cv2.imwrite(
            output_path,
            augmented
        )

        total += 1

print("\nDone!")
print("Total images generated:", total)
print("Output:", OUTPUT_DIR)