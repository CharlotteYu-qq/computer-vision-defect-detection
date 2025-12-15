import cv2
import os
import json
import csv
import numpy as np
import time
import math

# Introduce core detection functions
from geometry_utils_adaptive import detect_defects_adaptively

# Configuring paths
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "filmed_buttons")
OUTPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "output", "batch_visuals")
OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "output", "detection_results.json")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "output", "detection_report.csv")
ROI_PATH = os.path.join(PROJECT_ROOT, "config", "roi_config.json")
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "config", "template_centers.json")


# Auxiliary Functions 
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def calculate_global_offset(report, template_data):
    """
    Calculate Global Offset (Global Registration)
    Use Median to filter out individual severe offset interference
    """
    dx_list = []
    dy_list = []

    for btn, info in report.items():
        # only consider buttons with valid status (OK) and center
        if info['status'] == 'OK' and info['center']:
            d_pos = info['center']
            t_pos = template_data.get(btn)
            if t_pos:
                dx_list.append(d_pos[0] - t_pos[0])
                dy_list.append(d_pos[1] - t_pos[1])

    # if there are not enough valid offsets, return zero
    if len(dx_list) < 3:
        return 0.0, 0.0

    return float(np.median(dx_list)), float(np.median(dy_list))


def visualize_and_save(img, report, roi_data, save_name):
    # here only visualize for single image 
    # full view by generate_final_report.py
    vis = img.copy()
    for btn_name, info in report.items():
        rx, ry, rw, rh = roi_data[btn_name]
        status = info["status"]
        contours = info["contours"]

        if contours["red"] is not None:
            cv2.drawContours(vis, [contours["red"]], -1, (255, 255, 0), 1)
        if contours["blue"] is not None:
            cv2.drawContours(vis, [contours["blue"]], -1, (0, 215, 255), 1)

        color = (0, 255, 0) if status == "OK" else (0, 0, 255)
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 1)

    cv2.imwrite(save_name, vis)

# main function
def run_batch_processing():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # load ROI and template data
    with open(ROI_PATH) as f:
        roi_data = json.load(f)
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH) as f:
            template_data = json.load(f)
    else:
        print("Warning: Template centers not found. Misalignment calculation will be skipped.")
        template_data = {}

    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png'))])

    print(f"🚀 Starting Batch Processing & Data Calculation...")
    
    # Initialize containers
    dataset_results = {}
    csv_rows = []

    for idx, fname in enumerate(all_files):
        img_path = os.path.join(DATA_DIR, fname)
        img = cv2.imread(img_path)
        if img is None: continue

        # 1. Core detection (Status, Center, Thickness)
        report = detect_defects_adaptively(img, ROI_PATH)

        # 2. Calculate Global Offset
        g_dx, g_dy = calculate_global_offset(report, template_data)

        # 3. Organize data (Calculate net error for each button)
        clean_report = {}

        for btn, info in report.items():
            metrics = info["metrics"]
            status = info["status"]
            center = info["center"]

            # Calculate net error (Misalignment Distance)
            misalignment_px = 0.0
            if status == "OK" and center and template_data.get(btn):
                t_orig = template_data[btn]
                # Dynamic template position
                t_adj_x = t_orig[0] + g_dx
                t_adj_y = t_orig[1] + g_dy
                # Distance
                misalignment_px = math.sqrt((center[0] - t_adj_x) ** 2 + (center[1] - t_adj_y) ** 2)

            # Store in JSON (for visualization)
            clean_report[btn] = {
                "status": status,
                "center": center,
                "contours": info["contours"],
                "metrics": metrics,
                "alignment": {
                    "global_offset": (g_dx, g_dy),
                    "misalignment_px": misalignment_px
                }
            }

            # save in CSV
            csv_rows.append([
                fname,
                btn,
                status,
                round(metrics.get('r_area', 0), 1),
                round(metrics.get('b_area', 0), 1),
                round(metrics.get('r_thick', 0), 2),
                round(metrics.get('b_thick', 0), 2),
                round(g_dx, 2),  # global offset X
                round(g_dy, 2),  # global offset Y
                round(misalignment_px, 2)  # net error (placement accuracy)
            ])

        dataset_results[fname] = clean_report

        # Visualization (basic version)
        visualize_and_save(img, report, roi_data, os.path.join(OUTPUT_IMG_DIR, f"res_{fname}"))

        print(f"[{idx + 1}/{len(all_files)}] {fname:<15} | Offset: ({g_dx:+.1f}, {g_dy:+.1f})")

    # 4. Save JSON
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(dataset_results, f, indent=None, separators=(',', ':'), default=convert_to_serializable)

    # 5. Save CSV
    header = ["Image", "Button", "Status",
              "Red_Area", "Blue_Area", "Red_Thickness", "Blue_Thickness",
              "Global_Offset_X", "Global_Offset_Y", "Misalignment_Px"]

    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    print("-" * 60)
    print(f"🎉 Processing Complete!")
    print(f"Data saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    run_batch_processing()