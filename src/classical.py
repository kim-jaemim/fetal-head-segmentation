
import cv2
import numpy as np


# Final hyperparameters
IMG_SIZE = (256, 256)
CANNY_LOW = 30
CANNY_HIGH = 100
KERNEL_SIZE = 3
MIN_AREA = 152


# 1. Edge detection
def detect_edges(img, threshold1=CANNY_LOW,
                 threshold2=CANNY_HIGH):

    return cv2.Canny(img, threshold1, threshold2)


# 2. Morphological closing
def apply_morphology(edges):

    kernel = np.ones(
        (KERNEL_SIZE, KERNEL_SIZE),
        np.uint8
    )

    return cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel
    )


# 3. Contour extraction
def find_contours(edges):

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


# 4. Head contour selection
def select_head_contour(contours):

    candidates = []

    for contour in contours:

        if len(contour) < 5:
            continue

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        candidates.append(contour)

    if not candidates:
        return None

    return max(candidates, key=cv2.contourArea)


# 5. Ellipse fitting
def fit_head_ellipse(contour):

    if contour is None or len(contour) < 5:
        return None

    return cv2.fitEllipse(contour)


# 6. Ellipse mask generation
def create_ellipse_mask(img_shape, ellipse):

    mask = np.zeros(
        img_shape[:2],
        dtype=np.uint8
    )

    if ellipse is not None:
        cv2.ellipse(
            mask, ellipse, 1, thickness=-1
        )

    return mask


# 7. Head ellipse prediction
def predict_ellipse(img):

    edges = detect_edges(img)

    processed = apply_morphology(edges)

    contours = find_contours(processed)

    contour = select_head_contour(contours)

    return fit_head_ellipse(contour)


# 8. Head circumference in pixels
def calculate_hc(ellipse):

    if ellipse is None:
        return None

    width, height = ellipse[1]

    a = width / 2
    b = height / 2

    return np.pi * (
        3 * (a + b)
        - np.sqrt((3*a + b) * (a + 3*b))
    )


# 9. Head circumference in millimeters
def predict_hc(img, pixel_size, original_shape):

    ellipse = predict_ellipse(img)

    if ellipse is None:
        return np.nan

    (cx, cy), (width, height), angle = ellipse

    if not np.all(np.isfinite([
        cx, cy, width, height, angle
    ])):
        return np.nan

    if width <= 0 or height <= 0:
        return np.nan

    original_h, original_w = original_shape[:2]

    scale_x = original_w / img.shape[1]
    scale_y = original_h / img.shape[0]

    theta = np.deg2rad(angle)

    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    axes = np.diag([
        width / 2,
        height / 2
    ])

    scale = np.diag([
        scale_x,
        scale_y
    ])

    transform = scale @ rotation @ axes

    a, b = np.linalg.svd(
        transform,
        compute_uv=False
    )

    hc_pixel = np.pi * (
        3 * (a + b)
        - np.sqrt((3*a + b) * (a + 3*b))
    )

    return hc_pixel * pixel_size


# 10. Final segmentation prediction
def predict(img):

    ellipse = predict_ellipse(img)

    return create_ellipse_mask(
        img.shape, ellipse
    )
