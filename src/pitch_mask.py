import cv2
import numpy as np


def build_pitch_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def filter_on_pitch(detections, mask):
    keep = []
    for box in detections.xyxy:
        x1, y1, x2, y2 = box.astype(int)
        foot_x = min(max((x1 + x2) // 2, 0), mask.shape[1] - 1)
        foot_y = min(max(y2, 0), mask.shape[0] - 1)
        keep.append(mask[foot_y, foot_x] > 0)
    return detections[np.array(keep)]