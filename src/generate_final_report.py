import cv2
import json
import os
import numpy as np

# Configure paths

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "filmed_buttons")
RESULT_JSON_PATH = os.path.join(PROJECT_ROOT, "output", "detection_results.json")
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "config", "template_centers.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "final_unified_view")
ROI_PATH = os.path.join(PROJECT_ROOT, "config", "roi_config.json")

# Set misalignment threshold
MISALIGNMENT_THRESHOLD = 20.0


def load_json(path):
    with open(path, 'r') as f: return json.load(f)


def draw_hud(img, btn_name, info, roi, t_orig):
    rx, ry, rw, rh = roi
    status = info['status']
    center = info['center']
    contours = info['contours']
    metrics = info['metrics']
    alignment = info.get('alignment', {})  # Obtain alignment info safely

    g_offset = alignment.get('global_offset', [0, 0])
    misalignment_px = alignment.get('misalignment_px', 0.0)

    # --- 1. Basic box ---
    color_status = (0, 255, 0) if status == "OK" else (0, 0, 255)
    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color_status, 1)

    # --- 2. Contours ---
    if contours['red']:
        cv2.drawContours(img, [np.array(contours['red'], dtype=np.int32)], -1, (255, 255, 0), 1)
    if contours['blue']:
        cv2.drawContours(img, [np.array(contours['blue'], dtype=np.int32)], -1, (0, 215, 255), 1)

    # --- 3. Center & Offset ---
    is_shift = False

    # Only show offset info when status is OK
    if status == "OK" and center and t_orig:
        cx, cy = int(center[0]), int(center[1])
        # Dynamic template position
        ax, ay = int(t_orig[0] + g_offset[0]), int(t_orig[1] + g_offset[1])

        # Determine if misalignment exceeds threshold
        is_shift = misalignment_px > MISALIGNMENT_THRESHOLD
        arrow_color = (255, 0, 255) if is_shift else (0, 255, 255)  # Purple=exceeds, Yellow=pass

        # Draw
        cv2.drawMarker(img, (ax, ay), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)  # Green cross (reference)
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)  # Red dot (measured)

        if misalignment_px > 2.0:
            cv2.arrowedLine(img, (ax, ay), (cx, cy), arrow_color, 1, tipLength=0.3)

    # --- 4. Text HUD ---
    labels = []

    # A. Status/Offset Warning
    if status != "OK":
        labels.append(status.replace("MISSING_", "NO ").replace("THIN_", "THIN "))
    elif is_shift:
        labels.append("SHIFT")

    # B. Thickness Info
    thick_str = ""
    if metrics['r_area'] > 0: thick_str += f"R:{metrics.get('r_thick', 0):.1f} "
    if metrics['b_area'] > 0: thick_str += f"B:{metrics.get('b_thick', 0):.1f}"

    # C. Offset Info
    offset_str = ""
    if status == "OK":
        offset_str = f"Diff:{misalignment_px:.1f}px"

    # Plot text
    text_x = rx + 5
    text_y_center = ry + rh // 2

    # Plot status (centered)
    for i, label in enumerate(labels):
        cv2.putText(img, label, (text_x, text_y_center + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Plot thickness (bottom)
    if thick_str:
        cv2.putText(img, thick_str, (rx + 2, ry + rh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Plot offset value (top) - red if exceeds threshold, yellow if not
    if offset_str:
        color = (0, 0, 255) if is_shift else (0, 255, 255)
        cv2.putText(img, offset_str, (rx + 2, ry + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def generate_report():
    if not os.path.exists(RESULT_JSON_PATH):
        print("Error: JSON not found. Run batch_process_final.py first.")
        return

    det_data = load_json(RESULT_JSON_PATH)
    tpl_data = load_json(TEMPLATE_PATH)
    with open(ROI_PATH) as f:
        roi_data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 Generating Unified Report with Metrics...")

    count_defect = 0
    for fname, buttons in det_data.items():
        img_path = os.path.join(DATA_DIR, fname)
        img = cv2.imread(img_path)
        if img is None: continue

        vis_img = img.copy()
        has_issue = False

        for btn, info in buttons.items():
            t_orig = tpl_data.get(btn)
            draw_hud(vis_img, btn, info, roi_data[btn], t_orig)

            # Count defects (non-OK or misalignment exceeds threshold)
            if info['status'] != 'OK': has_issue = True
            if info.get('alignment', {}).get('misalignment_px', 0) > MISALIGNMENT_THRESHOLD:
                has_issue = True

        if has_issue: count_defect += 1
        prefix = "FAIL_" if has_issue else "PASS_"
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}{fname}"), vis_img)

    print(f"✅ Done! Defect images: {count_defect}")
    print(f"   Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_report()