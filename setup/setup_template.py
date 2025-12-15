import cv2
import json
import os
import numpy as np

# --- Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Data Source: Blank Buttons
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "blank_buttons")
ROI_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "roi_config.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "config", "template_centers.json")


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def detect_robust_center(roi_bgr):
    """
    Robust center detection using Edge Detection + Ellipse Fitting.
    Includes filtering for size and position to reject noise.
    """
    h, w = roi_bgr.shape[:2]
    roi_center = np.array([w / 2, h / 2])

    # 1. Gamma Correction
    enhanced = adjust_gamma(roi_bgr, gamma=2.0)

    # 2. Preprocessing
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    # Slightly stronger blur to smooth out internal texture
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Canny Edge Detection
    # Lower threshold to catch faint rims, high to skip soft shadows
    edges = cv2.Canny(blurred, 25, 80)

    # 4. Morphology to connect gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Find Contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # 6. Filter Candidates
    best_ellipse = None
    best_score = -1  # Score based on perimeter (larger is usually the rim)

    for cnt in contours:
        if len(cnt) < 5: continue

        # Fit Ellipse
        try:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ax1, ax2), angle = ellipse

            # --- Filter A: Position ---
            # The rim center shouldn't be too far from the ROI center
            dist = np.linalg.norm(np.array([cx, cy]) - roi_center)
            if dist > w * 0.25:  # If offset is > 25% of width, it's likely noise
                continue

            # --- Filter B: Size ---
            # Major axis should be reasonable (e.g., > 1/3 of ROI width)
            major_axis = max(ax1, ax2)
            if major_axis < w * 0.3 or major_axis > w * 0.95:
                continue

            # --- Selection Strategy ---
            # We prefer the largest contour that fits these criteria
            score = cv2.arcLength(cnt, True)
            if score > best_score:
                best_score = score
                best_ellipse = ellipse

        except cv2.error:
            pass

    # 7. Final Result
    if best_ellipse:
        (cx, cy), _, _ = best_ellipse
        return (float(cx), float(cy)), best_ellipse
    else:
        # Fallback: largest contour centroid if ellipse fails
        best_cnt = max(contours, key=cv2.contourArea)
        (cx, cy), _ = cv2.minEnclosingCircle(best_cnt)
        return (float(cx), float(cy)), None


def generate_template():
    if not os.path.exists(ROI_CONFIG_PATH):
        print("❌ Error: roi_config.json not found.")
        return

    with open(ROI_CONFIG_PATH) as f:
        roi_data = json.load(f)

    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png'))])
    if not files:
        print("❌ Error: No blank images found.")
        return

    img_path = os.path.join(DATA_DIR, files[0])
    print(f"🚀 Processing Template Source: {img_path}")

    img = cv2.imread(img_path)
    if img is None: return

    template_centers = {}
    vis_img = img.copy()

    for btn_name, (x, y, w, h) in roi_data.items():
        roi = img[y:y + h, x:x + w]

        local_center, ellipse_geom = detect_robust_center(roi)

        if local_center:
            gx, gy = x + local_center[0], y + local_center[1]
            template_centers[btn_name] = (gx, gy)

            # Visualization
            cv2.drawMarker(vis_img, (int(gx), int(gy)), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

            if ellipse_geom:
                (ex, ey), esize, eangle = ellipse_geom
                global_ellipse = ((x + ex, y + ey), esize, eangle)
                cv2.ellipse(vis_img, global_ellipse, (255, 255, 0), 1)

            print(f"  {btn_name:<10}: Found Center ({gx:.1f}, {gy:.1f})")
        else:
            print(f"⚠️ {btn_name:<10}: Detection Failed")
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(template_centers, f, indent=4)

    debug_path = os.path.join(PROJECT_ROOT, "output", "template.jpg")
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    cv2.imwrite(debug_path, vis_img)

    print("-" * 60)
    print(f"✅ Template centers saved to: {OUTPUT_FILE}")
    print(f"👀 Check visualization at: {debug_path}")


if __name__ == "__main__":
    generate_template()