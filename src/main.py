from dotenv import load_dotenv
import cv2
import csv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import numpy as np
import supervision as sv
from team_classifier import classify_teams
from tactical_metrics import compute_team_points, compute_metrics

load_dotenv()
import os


def build_client():
    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("ROBOFLOW_API_KEY"))
    client.configure(InferenceConfiguration(confidence_threshold=0.1))
    return client


def build_slicer(client):
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = client.infer(image_slice, model_id="football-players-detection-3zvbc/10")
        return sv.Detections.from_inference(result)
    return sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))


def gray_world_balance(image):
    b, g, r = image[:, :, 0].astype(np.float32), image[:, :, 1].astype(np.float32), image[:, :, 2].astype(np.float32)
    avg = (b.mean() + g.mean() + r.mean()) / 3
    b = b * (avg / b.mean())
    g = g * (avg / g.mean())
    r = r * (avg / r.mean())
    balanced = np.stack([b, g, r], axis=-1)
    return balanced.clip(0, 255).astype(np.uint8)


def main():
    client = build_client()
    slicer = build_slicer(client)

    cap = cv2.VideoCapture("../data/raw/match_full.mkv")
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_time_s = 15 * 60
    end_time_s = 20 * 60
    sample_every_n_seconds = 5
    min_detections = 14
    min_per_team = 7

    start_frame = int(start_time_s * fps)
    end_frame = int(end_time_s * fps)
    step = int(fps * sample_every_n_seconds)

    rows = []
    skipped = 0
    reference_colors = None

    for frame_idx in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        upscaled = cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        balanced = gray_world_balance(upscaled)

        detections = slicer(upscaled)
        detections = detections.with_nms(threshold=0.5)

        if len(detections) < min_detections:
            print(f"t={timestamp:.0f}s ({timestamp/60:.1f} min) ignoré — plan non exploitable ({len(detections)} détections)")
            skipped += 1
            continue

        team_map, reference_colors = classify_teams(detections, balanced, reference_colors)
        team_points = compute_team_points(detections, team_map)

        n_a = len(team_points.get("teamA", []))
        n_b = len(team_points.get("teamB", []))
        if n_a < min_per_team or n_b < min_per_team:
            print(f"t={timestamp:.0f}s ({timestamp/60:.1f} min) ignoré — équipes déséquilibrées (A={n_a}, B={n_b})")
            skipped += 1
            continue

        metrics = compute_metrics(team_points)

        if "teamA" not in metrics or "teamB" not in metrics:
            print(f"t={timestamp:.0f}s ({timestamp/60:.1f} min) ignoré — hull incomplet")
            skipped += 1
            continue

        row = {"timestamp_s": round(timestamp, 1)}
        for team in ["teamA", "teamB"]:
            row[f"{team}_hull_area"] = metrics[team]["hull_area_px"]
            row[f"{team}_avg_nearest_opp"] = metrics[team]["avg_nearest_opponent_px"]

        rows.append(row)
        print(f"t={timestamp:.0f}s ({timestamp/60:.1f} min) traité — {len(detections)} détections (A={n_a}, B={n_b})")

    cap.release()

    with open("../data/metrics_timeseries.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} échantillons exploitables, {skipped} plans ignorés")


if __name__ == "__main__":
    main()