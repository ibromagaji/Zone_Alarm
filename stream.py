import cv2
import threading
import time
from ultralytics import YOLO

model = YOLO("/home/ju5ti5/zone/yolo_320.tflite")  # Load the YOLO model

class LatestFrameReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame  # always overwrite, never queue

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()


reader = LatestFrameReader("http://10.103.29.16:8080/video")

while True:
    frame = reader.read()
    if frame is None:
        time.sleep(0.005)
        continue

    results = model(frame)          # process whatever is currentsrent
    annotated = results[0].plot()
    cv2.imshow("YOLO", annotated)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader.stop()
cv2.destroyAllWindows()