import cv2

cap = cv2.VideoCapture("../data/raw/match_full.mkv")
fps = cap.get(cv2.CAP_PROP_FPS)

for t in [350, 380, 425, 440, 495]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    cv2.imwrite(f"../data/check_t{t}.jpg", frame)

print("frames sauvegardées")