import cv2
import numpy as np
import json


# --------------------------------------------------------
# Auxiliary Function: Calculate Robust Thickness - Anti-Sawtooth
# --------------------------------------------------------
def calculate_robust_thickness(mask):
    """
    Calculate the 'visual thickness' of a contour using distance transform.
    Advantage: Very insensitive to sawtooth artifacts in the blue channel, measuring only the 'skeleton' width.
    """
    # if there is no contour, thickness is 0
    if cv2.countNonZero(mask) == 0:
        return 0.0

    # 1. Distance transform: calculate the distance of each pixel to the nearest background (0)
    # The values in dist_map represent the radius (e.g. 1.0, 2.0, 2.5...)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    # 2. Extract valid distance values inside the contour
    valid_dists = dist_map[dist_map > 0]

    if len(valid_dists) == 0:
        return 0.0

    # 3. Statistics: use 90th percentile to represent thickness
    # Do not use max (affected by noise), do not use mean (lowered by edge sawtooth)
    # Use 90% percentile to represent the thickness of the "main part", ignoring edge burrs
    radius_p90 = np.percentile(valid_dists, 90)

    # Distance transform calculates the radius, thickness = diameter
    return radius_p90 * 2.0


# --------------------------------------------------------
# Step 1: Image Enhancement and Color Channel Separation
# --------------------------------------------------------
def enhance_and_split(roi_bgr):
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    img_clahe = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    img_float = img_clahe.astype(np.float32)
    B_ch, G_ch, R_ch = cv2.split(img_float)

    red_score = R_ch - 0.6 * G_ch - 0.6 * B_ch
    blue_score = B_ch - 0.5 * G_ch - 0.5 * R_ch

    RED_NOISE_FLOOR = 40
    BLUE_NOISE_FLOOR = 20

    if np.max(red_score) > RED_NOISE_FLOOR:
        red_norm = cv2.normalize(np.maximum(red_score, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        red_norm = np.zeros_like(R_ch, dtype=np.uint8)

    if np.max(blue_score) > BLUE_NOISE_FLOOR:
        blue_norm = cv2.normalize(np.maximum(blue_score, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        blue_norm = np.zeros_like(B_ch, dtype=np.uint8)

    return red_norm, blue_norm


# --------------------------------------------------------
# Step 2: Calculate Contour Metrics (increase thickness)
# --------------------------------------------------------
def get_contour_metrics(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize metrics
    metrics = {"area": 0, "center": None, "contour": None, "thickness": 0.0}

    if not contours:
        return metrics

    best_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_cnt)

    # calculate thickness
    thickness = calculate_robust_thickness(mask)

    center = None
    if area > 0:
        M = cv2.moments(best_cnt)
        if M["m00"] != 0:
            center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        else:
            (cx, cy), _ = cv2.minEnclosingCircle(best_cnt)
            center = (cx, cy)

    return {"area": area, "center": center, "contour": best_cnt, "thickness": thickness}


# --------------------------------------------------------
# Step 3: single ROI button feature extraction
# --------------------------------------------------------
def extract_button_features(roi_bgr):
    red_norm, blue_norm = enhance_and_split(roi_bgr)

    _, red_mask = cv2.threshold(red_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, blue_mask = cv2.threshold(blue_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    r_metrics = get_contour_metrics(red_mask)
    b_metrics = get_contour_metrics(blue_mask)

    return {"red": r_metrics, "blue": b_metrics}


# --------------------------------------------------------
# Step 4: Adaptive Detection (increase thickness judgment logic)
# --------------------------------------------------------
def detect_defects_adaptively(img, roi_path="../roi_config.json"):
    with open(roi_path) as f:
        roi_data = json.load(f)

    all_features = {}

    # Data collection containers
    stats = {
        "r_area": [], "b_area": [],
        "r_thick": [], "b_thick": []
    }

    # 1. Iterate and extract
    for btn_name, (x, y, w, h) in roi_data.items():
        h_img, w_img = img.shape[:2]
        roi = img[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]

        feats = extract_button_features(roi)

        # Coordinate transformation and stats collection
        for color in ["red", "blue"]:
            if feats[color]["contour"] is not None:
                feats[color]["global_contour"] = feats[color]["contour"] + (x, y)
            else:
                feats[color]["global_contour"] = None

            if feats[color]["center"]:
                cx, cy = feats[color]["center"]
                feats[color]["global_center"] = (x + cx, y + cy)

                # Only include thickness in stats if center is detected
                stats[f"{color[0]}_area"].append(feats[color]["area"])
                stats[f"{color[0]}_thick"].append(feats[color]["thickness"])

        all_features[btn_name] = feats

    # 2. Calculate baselines (medians)
    medians = {}
    for key, values in stats.items():
        valid_vals = [v for v in values if v > 0]
        medians[key] = np.median(valid_vals) if valid_vals else 0

    # 3. Judgment Logic
    AREA_THRESH = 0.3  # Area must be > 30% of median
    THIN_WARNING_THRESH = 0.6  # Below 60% is considered THIN (previously 50%, slightly stricter)

    # Core modification: more aggressive "missing" judgment threshold
    # If thickness is below 35% of median, directly consider it Missing, no chance for Thin
    CRITICAL_MISSING_RATIO = 0.35

    # Core modification: absolute physical bottom line (pixels)
    # Anything with thickness less than 2.0px is considered noise
    ABS_MIN_THICKNESS = 2.0

    final_report = {}

    for btn_name, feats in all_features.items():
        r_area = feats["red"]["area"]
        b_area = feats["blue"]["area"]
        r_thick = feats["red"]["thickness"]
        b_thick = feats["blue"]["thickness"]

        # --- 1. More stringent existence check ---
        # Logic: must satisfy all three conditions to be considered "existent":
        # A. Area is large enough (to prevent small spots)
        # B. Relative thickness is large enough (to prevent thin scratches)
        # C. Absolute thickness is large enough (to prevent overall thin noise)
        # red check
        has_red = False
        if r_area > medians["r_area"] * AREA_THRESH:
            # Only check thickness if area is sufficient to avoid division by zero or logical confusion
            if r_thick > ABS_MIN_THICKNESS:  # Absolute bottom line
                if medians["r_thick"] > 0 and (r_thick > medians["r_thick"] * CRITICAL_MISSING_RATIO):
                    has_red = True

        # blue check
        has_blue = False
        if b_area > medians["b_area"] * AREA_THRESH:
            if b_thick > ABS_MIN_THICKNESS:  # Absolute bottom line
                if medians["b_thick"] > 0 and (b_thick > medians["b_thick"] * CRITICAL_MISSING_RATIO):
                    has_blue = True

        # --- 2. Quality check (THIN judgment) ---
        # Reaching here means has_red/has_blue are already True
        # This means they have already passed the 35% bottom line test
        # Now check if they are between 35% and 60% (i.e., "existent but thin")

        red_is_thin = False
        blue_is_thin = False

        if has_red and medians["r_thick"] > 0:
            if r_thick < medians["r_thick"] * THIN_WARNING_THRESH:
                red_is_thin = True

        if has_blue and medians["b_thick"] > 0:
            if b_thick < medians["b_thick"] * THIN_WARNING_THRESH:
                blue_is_thin = True

        # --- 3. Final status combination ---
        status = "OK"
        final_center = None

        if not has_red and not has_blue:
            status = "MISSING_BOTH"
        elif not has_red:
            status = "MISSING_RED"
            final_center = feats["blue"]["global_center"]
        elif not has_blue:
            status = "MISSING_BLUE"
            final_center = feats["red"]["global_center"]
        else:
            # Both exist, check quality
            if red_is_thin and blue_is_thin:
                status = "THIN_BOTH"
            elif red_is_thin:
                status = "THIN_RED"
            elif blue_is_thin:
                status = "THIN_BLUE"
            else:
                status = "OK"

            # Calculate final center as average of both centers
            rc = feats["red"]["global_center"]
            bc = feats["blue"]["global_center"]
            final_center = ((rc[0] + bc[0]) / 2, (rc[1] + bc[1]) / 2)

        # All black protection 
        # (if medians themselves are very small, 
        # it means the whole image might not be well captured)
        if medians["r_area"] < 10 and medians["b_area"] < 10:
            status = "ERROR_LOW_QUALITY"

        # Build return structure
        final_report[btn_name] = {
            #  keep consistent structure
            "status": status,
            "center": final_center,
            "contours": {
                "red": feats["red"]["global_contour"],
                "blue": feats["blue"]["global_contour"]
            },
            "metrics": {
                "r_area": r_area,
                "b_area": b_area,
                "r_thick": r_thick,
                "b_thick": b_thick
            }
        }

    return final_report