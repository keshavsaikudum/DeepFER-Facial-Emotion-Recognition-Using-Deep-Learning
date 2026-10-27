"""Live webcam emotion recognition. Press q to quit. Usage: python realtime.py [--ckpt runs/resnet18.pt] [--cam 0]"""
import argparse
import time

import cv2

from emotion import EmotionRecognizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/resnet18.pt")
ap.add_argument("--cam", type=int, default=0)
a = ap.parse_args()

rec = EmotionRecognizer(a.ckpt)
cap = cv2.VideoCapture(a.cam)
if not cap.isOpened():
    raise SystemExit("Could not open camera (on macOS, allow camera access for your terminal app).")
fps, t0 = 0.0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame, _ = rec.annotate(cv2.flip(frame, 1))
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / (now - t0)
    t0 = now
    cv2.putText(frame, f"{fps:.0f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("DeepFER", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
