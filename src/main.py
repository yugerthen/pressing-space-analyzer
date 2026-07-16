from dotenv import load_dotenv
import cv2
import os
from detection import build_client, build_slicer, detect_and_filter, crop_pitch, gray_world_balance
from team_classifier import classify_teams
from tactical_metrics import compute_team_points, compute_metrics, draw_tactical_overlay

load_dotenv()


def process_frame(slicer, frame):
    cropped = crop_pitch(frame)
    balanced = gray_world_balance(cropped)

    detections = detect_and_filter(slicer, cropped)
    team_map = classify_teams(detections, balanced)
    team_points = compute_team_points(detections, team_map)
    metrics = compute_metrics(team_points)

    annotated = draw_tactical_overlay(cropped, detections, team_map, metrics)
    return annotated, metrics


def main():
    client = build_client()
    slicer = build_slicer(client)

    cap = cv2.VideoCapture("../data/raw/match_clip.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)

    os.makedirs("../data/gif_frames", exist_ok=True)

    n_frames = int(fps * 4)
    step = 2

    for i in range(0, n_frames, step):
        ret, frame = cap.read()
        if not ret:
            break
        annotated, metrics = process_frame(slicer, frame)
        cv2.imwrite(f"../data/gif_frames/frame_{i:04d}.jpg", annotated)
        print(f"frame {i+1}/{n_frames} traitée")

    cap.release()


if __name__ == "__main__":
    main()