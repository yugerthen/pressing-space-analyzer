from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import numpy as np
import supervision as sv
import os


def build_client():
    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("ROBOFLOW_API_KEY"))
    client.configure(InferenceConfiguration(confidence_threshold=0.12))
    return client


def build_slicer(client):
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = client.infer(image_slice, model_id="football-players-detection-3zvbc/10")
        return sv.Detections.from_inference(result)
    return sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))


def detect_and_filter(slicer, image, pitch_top_ratio=0.08):
    ch = image.shape[0]
    detections = slicer(image)
    detections = detections.with_nms(threshold=0.5)
    keep = detections.xyxy[:, 1] > ch * pitch_top_ratio
    return detections[keep]


def crop_pitch(frame, w_ratio=(0.28, 0.95), h_ratio=(0.40, 0.82)):
    h_full, w_full = frame.shape[:2]
    x1 = int(w_full * w_ratio[0])
    x2 = int(w_full * w_ratio[1])
    y1 = int(h_full * h_ratio[0])
    y2 = int(h_full * h_ratio[1])
    return frame[y1:y2, x1:x2]


def gray_world_balance(image):
    b, g, r = image[:, :, 0].astype(np.float32), image[:, :, 1].astype(np.float32), image[:, :, 2].astype(np.float32)
    avg = (b.mean() + g.mean() + r.mean()) / 3
    b = b * (avg / b.mean())
    g = g * (avg / g.mean())
    r = r * (avg / r.mean())
    balanced = np.stack([b, g, r], axis=-1)
    return balanced.clip(0, 255).astype(np.uint8)