import cv2
import numpy as np


def foot_point(box):
    bx1, by1, bx2, by2 = box.astype(int)
    return np.array([(bx1 + bx2) // 2, by2])


def compute_team_points(detections, team_map):
    team_points = {"teamA": [], "teamB": []}
    for pos, label in team_map.items():
        if label in team_points:
            team_points[label].append(foot_point(detections.xyxy[pos]))
    return team_points


def compute_metrics(team_points):
    results = {}
    for team, points in team_points.items():
        pts = np.array(points)
        if len(pts) < 5:
            continue
        hull = cv2.convexHull(pts)
        area = cv2.contourArea(hull)

        opponent = "teamB" if team == "teamA" else "teamA"
        opp_pts = np.array(team_points[opponent])
        dists = []
        if len(opp_pts) > 0:
            for p in pts:
                d = np.min(np.linalg.norm(opp_pts - p, axis=1))
                dists.append(d)

        avg_nearest_opp = np.mean(dists) if dists else None
        results[team] = {
            "hull_area_px": area,
            "avg_nearest_opponent_px": avg_nearest_opp,
            "hull_points": hull,
        }
    return results


def draw_tactical_overlay(image, detections, team_map, metrics):
    palette = {"teamA": (0, 0, 255), "teamB": (255, 0, 0), "ref": (0, 255, 255)}
    overlay = image.copy()
    annotated = image.copy()

    for team, m in metrics.items():
        cv2.fillPoly(overlay, [m["hull_points"]], palette[team])
        cv2.polylines(annotated, [m["hull_points"]], True, palette[team], 2)

    annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)

    for pos, label in team_map.items():
        box = detections.xyxy[pos]
        bx1, by1, bx2, by2 = box.astype(int)
        cx = (bx1 + bx2) // 2
        cy2 = by2
        color = palette.get(label, (0, 255, 255))
        cv2.ellipse(annotated, (cx, cy2), (18, 8), 0, -45, 235, color, 2)

    return annotated