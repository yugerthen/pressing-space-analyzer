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


def classify_teams(detections, balanced_image, reference_colors=None):
    valid_idx = []
    colors = []
    for i, box in enumerate(detections.xyxy):
        color = get_shirt_color(balanced_image, box)
        if color is None:
            continue
        valid_idx.append(i)
        colors.append(color)

    if len(colors) < 4:
        return {}, reference_colors

    colors = np.array(colors)
    b_vals, g_vals, r_vals = colors[:, 0], colors[:, 1], colors[:, 2]
    ref_mask = (r_vals > g_vals - 10) & (g_vals > b_vals + 15)

    non_ref_colors = colors[~ref_mask]
    non_ref_positions = np.array(valid_idx)[~ref_mask]

    final_map = {}
    if len(non_ref_colors) >= 2:
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
        cluster_labels = kmeans.fit_predict(non_ref_colors)
        centers = kmeans.cluster_centers_

        if reference_colors is None:
            reference_colors = {"teamA": centers[0], "teamB": centers[1]}
        else:
            dist_00 = np.linalg.norm(centers[0] - reference_colors["teamA"])
            dist_01 = np.linalg.norm(centers[0] - reference_colors["teamB"])
            if dist_00 <= dist_01:
                cluster_to_team = {0: "teamA", 1: "teamB"}
            else:
                cluster_to_team = {0: "teamB", 1: "teamA"}
            for team in ["teamA", "teamB"]:
                idx = 0 if cluster_to_team[0] == team else 1
                reference_colors[team] = 0.8 * reference_colors[team] + 0.2 * centers[idx]

        if reference_colors is not None and "teamA" in reference_colors:
            dist_to_A = [np.linalg.norm(c - reference_colors["teamA"]) for c in centers]
            cluster_to_team = {int(np.argmin(dist_to_A)): "teamA", int(np.argmax(dist_to_A)): "teamB"}

        for pos, cl in zip(non_ref_positions, cluster_labels):
            final_map[pos] = cluster_to_team[cl]

    for pos in np.array(valid_idx)[ref_mask]:
        final_map[pos] = "ref"

    return final_map, reference_colors