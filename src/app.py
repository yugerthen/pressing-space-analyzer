import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import supervision as sv
from team_classifier import classify_teams
from tactical_metrics import compute_team_points, compute_metrics, draw_tactical_overlay
from pitch_mask import build_pitch_mask, filter_on_pitch

load_dotenv()

st.set_page_config(page_title="Pressing & Space Analyzer", layout="wide")
st.title("⚽ Pressing & Space Analyzer")
st.caption("Analyse tactique automatique : détection des joueurs, classification par équipe, métriques de pressing et d'occupation spatiale.")


@st.cache_resource
def get_client():
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


uploaded_file = st.file_uploader("Dépose une vidéo de match (vue large)", type=["mp4", "mkv", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    st.write(f"Durée détectée : {duration/60:.1f} minutes")

    timestamp = st.slider("Instant à analyser (secondes)", 0.0, duration, min(60.0, duration / 2))

    if st.button("Analyser cette frame"):
        with st.spinner("Détection et classification en cours..."):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
            ret, frame = cap.read()

            if not ret:
                st.error("Impossible de lire cette frame.")
            else:
                h, w = frame.shape[:2]
                if w < 800:
                    frame = cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

                balanced = gray_world_balance(frame)

                client = get_client()
                slicer = build_slicer(client)
                detections = slicer(frame)
                detections = detections.with_nms(threshold=0.3)

                pitch_mask = build_pitch_mask(frame)
                detections = filter_on_pitch(detections, pitch_mask)

                n_detections = len(detections)

                if n_detections > 26:
                    st.warning(f"⚠️ {n_detections} détections — probablement des doublons dus au découpage en tuiles. Résultat à interpréter avec prudence.")
                elif n_detections < 10:
                    st.error(f"⚠️ Seulement {n_detections} joueurs détectés sur cette frame — plan non exploitable (gros plan, ralenti, ou angle inhabituel). Essayez un autre instant.")
                elif n_detections < 16:
                    st.warning(f"⚠️ Seulement {n_detections} joueurs détectés — résultat possible mais moins fiable. Un instant avec plus de joueurs visibles donnerait un résultat plus précis.")
                else:
                    st.success(f"✅ {n_detections} joueurs détectés — bon plan pour l'analyse.")

                team_map, _ = classify_teams(detections, balanced, None)
                team_points = compute_team_points(detections, team_map)
                metrics = compute_metrics(team_points)

                annotated = draw_tactical_overlay(frame, detections, team_map, metrics)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb, caption="Résultat annoté", use_container_width=True)

                st.caption("💡 Les distances et aires sont exprimées en pixels de l'image, pas en mètres réels. Elles ne sont comparables qu'entre les deux équipes sur cette même frame, pas dans l'absolu — le zoom de la caméra affecte l'échelle.")

                col1, col2 = st.columns(2)
                for i, (team, m) in enumerate(metrics.items()):
                    col = col1 if i == 0 else col2
                    col.metric(f"{team} — Aire hull", f"{m['hull_area_px']:.0f} px²")
                    col.metric(f"{team} — Pressing subi", f"{m['avg_nearest_opponent_px']:.0f} px")

    cap.release()
    os.unlink(video_path)
else:
    st.info("Dépose une vidéo pour commencer.")