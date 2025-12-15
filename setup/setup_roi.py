import cv2
import json
import os

# === CONFIG ===
IMAGE_PATH = "data/blank_buttons/1.png"
OUTPUT_JSON = "config/roi_config.json"

# Global variables
drawing = False
ix, iy = -1, -1
rois = []
current_img = None


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_img, rois

    img_copy = current_img.copy()

    # When mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # When dragging mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("ROI Marking", img_copy)

    # When mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("ROI Marking", img_copy)

        # Save ROI
        x1, y1 = min(ix, x), min(iy, y)
        w = abs(ix - x)
        h = abs(iy - y)
        rois.append([x1, y1, w, h])

        print(f"ROI {len(rois)} = [x={x1}, y={y1}, w={w}, h={h}]")

        # If 8 ROIs are done, auto-save
        if len(rois) == 8:
            save_rois()


def save_rois():
    global rois

    roi_dict = {}
    for i, box in enumerate(rois):
        roi_dict[f"button_{i+1}"] = box

    with open(OUTPUT_JSON, "w") as f:
        json.dump(roi_dict, f, indent=4)

    print("\n=== ROI SAVED TO roi_config.json ===")
    print(json.dumps(roi_dict, indent=4))
    print("Close window to finish.")


def main():
    global current_img

    if not os.path.exists(IMAGE_PATH):
        print("ERROR: IMAGE_PATH not found:", IMAGE_PATH)
        return

    current_img = cv2.imread(IMAGE_PATH)
    cv2.namedWindow("ROI Marking")
    cv2.setMouseCallback("ROI Marking", draw_rectangle)

    print("Instructions:")
    print("1. Use mouse to DRAW 8 rectangles (one for each button).")
    print("2. When done, roi_config.json will be saved automatically.")
    print("3. Close the window when finished.\n")

    cv2.imshow("ROI Marking", current_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()