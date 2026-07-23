from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv
import cv2
import numpy as np
import supervision as sv
import os

load_dotenv()
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("ROBOFLOW_API_KEY"))
client.configure(InferenceConfiguration(confidence_threshold=0.1))

cap = cv2.VideoCapture("../data/raw/match_full.mkv")
cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
ret, frame = cap.read()

upscaled = cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = client.infer(image_slice, model_id="football-players-detection-3zvbc/10")
    return sv.Detections.from_inference(result)

slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))
detections = slicer(upscaled)
detections = detections.with_nms(threshold=0.5)

annotator = sv.EllipseAnnotator(thickness=2)
annotated = annotator.annotate(upscaled.copy(), detections)

cv2.imwrite("../data/laliga_test.jpg", annotated)
print(f"{len(detections)} détections")