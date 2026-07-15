from dotenv import load_dotenv
import cv2
from detection import build_client, build_slicer, detect_and_filter, crop_pitch, gray_world_balance
from team_classifier import classify_teams
from tactical_metrics import compute_team_points, compute_metrics, draw_tactical_overlay

load_dotenv()


def main():
    client = build_client()
    slicer = build_slicer(client)

    cap = cv2.VideoCapture("../data/raw/match_clip.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erreur : impossible de lire la vidéo")
        return

    cropped = crop_pitch(frame)
    balanced = gray_world_balance(cropped)

    detections = detect_and_filter(slicer, cropped)
    team_map = classify_teams(detections, balanced)
    team_points = compute_team_points(detections, team_map)
    metrics = compute_metrics(team_points)

    annotated = draw_tactical_overlay(cropped, detections, team_map, metrics)
    cv2.imwrite("../data/tactical_metrics.jpg", annotated)

    print("=== Métriques tactiques ===")
    for team, m in metrics.items():
        area = m["hull_area_px"]
        dist = m["avg_nearest_opponent_px"]
        print(f"{team}: aire hull = {area:.0f} px², distance moy. adversaire = {dist:.0f} px")


if __name__ == "__main__":
    main()