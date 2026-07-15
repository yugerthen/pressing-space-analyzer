from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv
import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
import os

load_dotenv()
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("ROBOFLOW_API_KEY"))
client.configure(InferenceConfiguration(confidence_threshold=0.12))

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = client.infer(image_slice, model_id="football-players-detection-3zvbc/10")
    return sv.Detections.from_inference(result)

slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))

cap = cv2.VideoCapture("data/raw/match_clip.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ret, frame = cap.read()

h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
x1 = int(w_full * 0.28)
y1 = int(h_full * 0.40)
x2 = int(w_full * 0.95)
y2 = int(h_full * 0.82)
cropped = frame[y1:y2, x1:x2]
ch = cropped.shape[0]

b, g, r = cv2.split(cropped.astype(np.float32))
avg = (b.mean() + g.mean() + r.mean()) / 3
b = b * (avg / b.mean())
g = g * (avg / g.mean())
r = r * (avg / r.mean())
balanced = cv2.merge([b, g, r])
balanced = balanced.clip(0, 255).astype(np.uint8)

detections = slicer(cropped)
detections = detections.with_nms(threshold=0.5)
keep = detections.xyxy[:, 1] > ch * 0.08
detections = detections[keep]

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

valid_idx = []
colors = []
for i, box in enumerate(detections.xyxy):
    color = get_shirt_color(balanced, box)
    if color is None:
        continue
    valid_idx.append(i)
    colors.append(color)

colors = np.array(colors)
b_vals, g_vals, r_vals = colors[:, 0], colors[:, 1], colors[:, 2]
ref_mask = (r_vals > g_vals - 10) & (g_vals > b_vals + 15)

non_ref_colors = colors[~ref_mask]
non_ref_positions = np.array(valid_idx)[~ref_mask]

kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
cluster_labels = kmeans.fit_predict(non_ref_colors)

final_map = {}
for pos, cl in zip(non_ref_positions, cluster_labels):
    final_map[pos] = "teamA" if cl == 0 else "teamB"
for pos in np.array(valid_idx)[ref_mask]:
    final_map[pos] = "ref"

def foot_point(box):
    bx1, by1, bx2, by2 = box.astype(int)
    return np.array([(bx1 + bx2) // 2, by2])

team_points = {"teamA": [], "teamB": []}
for pos, label in final_map.items():
    if label in team_points:
        team_points[label].append(foot_point(detections.xyxy[pos]))

annotated = cropped.copy()
overlay = cropped.copy()
palette = {"teamA": (0, 0, 255), "teamB": (255, 0, 0)}

results = {}
for team, points in team_points.items():
    pts = np.array(points)
    if len(pts) < 3:
        continue
    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)
    cv2.fillPoly(overlay, [hull], palette[team])
    cv2.polylines(annotated, [hull], True, palette[team], 2)

    dists = []
    for p in pts:
        opp = np.array(team_points["teamB" if team == "teamA" else "teamA"])
        if len(opp) == 0:
            continue
        d = np.min(np.linalg.norm(opp - p, axis=1))
        dists.append(d)
    avg_nearest_opp = np.mean(dists) if dists else None
    results[team] = {"hull_area_px": area, "avg_nearest_opponent_px": avg_nearest_opp}

annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)

for pos, label in final_map.items():
    box = detections.xyxy[pos]
    bx1, by1, bx2, by2 = box.astype(int)
    cx = (bx1 + bx2) // 2
    cy2 = by2
    color = palette.get(label, (0, 255, 255))
    cv2.ellipse(annotated, (cx, cy2), (18, 8), 0, -45, 235, color, 2)

cv2.imwrite("data/tactical_metrics.jpg", annotated)

print("=== Métriques tactiques ===")
for team, m in results.items():
    print(f"{team}: aire hull = {m['hull_area_px']:.0f} px², "
          f"distance moy. à l'adversaire le plus proche = {m['avg_nearest_opponent_px']:.0f} px")

if "teamA" in results and "teamB" in results:
    tighter = "teamA" if results["teamA"]["avg_nearest_opponent_px"] < results["teamB"]["avg_nearest_opponent_px"] else "teamB"
    print(f"→ {tighter} subit un pressing plus intense (adversaires plus proches en moyenne)")