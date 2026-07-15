import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_shirt_color(image, box):
    bx1, by1, bx2, by2 = box.astype(int)
    h = by2 - by1
    w = bx2 - bx1
    top = by1 + int(h * 0.15)
    bottom = by1 + int(h * 0.40)
    left = bx1 + int(w * 0.3)
    right = bx2 - int(w * 0.3)
    torso = image[top:bottom, left:right]
    if torso.size < 9:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    grass_mask = (hue > 25) & (hue < 95) & (sat > 40) & (val > 40)
    pixels = torso[~grass_mask]
    if len(pixels) < 5:
        return None
    return np.median(pixels, axis=0)


def classify_teams(detections, balanced_image):
    valid_idx = []
    colors = []
    for i, box in enumerate(detections.xyxy):
        color = get_shirt_color(balanced_image, box)
        if color is None:
            continue
        valid_idx.append(i)
        colors.append(color)

    if len(colors) < 4:
        return {}

    colors = np.array(colors)
    b_vals, g_vals, r_vals = colors[:, 0], colors[:, 1], colors[:, 2]
    ref_mask = (r_vals > g_vals - 10) & (g_vals > b_vals + 15)

    non_ref_colors = colors[~ref_mask]
    non_ref_positions = np.array(valid_idx)[~ref_mask]

    final_map = {}
    if len(non_ref_colors) >= 2:
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
        cluster_labels = kmeans.fit_predict(non_ref_colors)
        for pos, cl in zip(non_ref_positions, cluster_labels):
            final_map[pos] = "teamA" if cl == 0 else "teamB"

    for pos in np.array(valid_idx)[ref_mask]:
        final_map[pos] = "ref"

    return final_map