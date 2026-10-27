"""Face detection + emotion prediction shared by the webcam script and the app."""
import cv2
import numpy as np
import torch

from common import CLASSES, get_device, load_checkpoint

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class EmotionRecognizer:
    def __init__(self, ckpt="runs/resnet18.pt"):
        self.device = get_device()
        self.model, self.size = load_checkpoint(ckpt, self.device)

    def detect(self, gray):
        return _cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))

    @torch.no_grad()
    def predict_faces(self, gray, boxes):
        """Return a (n, 7) probability array for the given face boxes."""
        if len(boxes) == 0:
            return np.zeros((0, len(CLASSES)))
        crops = []
        for x, y, w, h in boxes:
            c = cv2.resize(gray[y:y + h, x:x + w], (self.size, self.size), interpolation=cv2.INTER_AREA)
            crops.append((c.astype(np.float32) / 255 - 0.5) / 0.5)
        t = torch.from_numpy(np.stack(crops))[:, None].to(self.device)
        return torch.softmax(self.model(t), 1).cpu().numpy()

    def annotate(self, bgr):
        """Draw boxes + labels on a BGR frame; returns (frame, [(label, conf)])."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detect(gray)
        probs = self.predict_faces(gray, boxes)
        results = []
        for (x, y, w, h), p in zip(boxes, probs):
            k = int(p.argmax())
            results.append((CLASSES[k], float(p[k])))
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(bgr, f"{CLASSES[k]} {p[k]:.0%}", (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        return bgr, results
