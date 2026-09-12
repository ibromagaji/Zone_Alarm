import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import tensorflow as tf
import os
from dotenv import load_dotenv
import requests
import pygame 
import threading
import time

HOME = os.getcwd()
load_dotenv()

bot_token = os.getenv('bot_token')
chat_id = os.getenv('chat_id')


pygame.mixer.init()

ALARM_SOUND_PATH = "/home/ju5ti5/detection/alarm.wav" 

try:
    pygame.mixer.music.load(ALARM_SOUND_PATH)
except Exception as e:
    print(f"Warning: Could not load sound file: {e}. Running without audio.")


def send_telegram_alert(frame,caption="Object detected!"):
    _, img_encoded = cv2.imencode('.jpg', frame)
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': ('detection.jpg', img_encoded.tobytes())}
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        response =requests.post(url, files=files, data=data)
        print(response.status_code)
    except Exception as e:
        print(f'Failed to send Telegram Alert: {e}')

def send_alert(frame,caption='Object Detected'):
    thread = threading.Thread(target=send_telegram_alert,args=(frame.copy(),caption))
    thread.daemon=True
    thread.start()

def trigger_alarm():
    print("ALARM: object entered the zone!")
    try:
        # -1 loops the sound indefinitely until stopped
        pygame.mixer.music.play(-1) 
    except Exception as e:
        print(f"Failed to play sound: {e}")


def clear_alarm():
    print("Zone clear.")
    pygame.mixer.music.stop()  # Stop the audio when the zone is clear


def draw_alarm_overlay(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 15)
    cv2.putText(
        frame, "ALARM", (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA
    )
    return frame

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
                self.frame = frame  

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()


reader = LatestFrameReader("http://10.103.29.16:8080/video")


time.sleep(0.005) 
ret, test_frame = reader.read()
if not ret or test_frame is None:
    print("Error: Could not open the IP camera stream. Verify network connection and URL.")
    reader.stop()
    exit()

print("Live stream active. Press 'q' to exit.")

model = YOLO("/home/ju5ti5/detection/yolo_320.tflite")
tracker = sv.ByteTrack()

polygon = np.array([[342, 413],
                    [185, 564],
                    [703, 607],
                    [929, 400]])
zone = sv.PolygonZone(polygon=polygon)

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, color=sv.Color.RED, thickness=2, text_thickness=2, text_scale=1
)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

alarm_active = False  


while True:
    ret, frame = reader.read()
    frame_spawn = time.time()
    if not ret or frame is None:
        print("Waiting for camera frame...")
        continue

    results_generator = model(frame, stream=True)
    results = next(results_generator)
    
    detections = sv.Detections.from_ultralytics(results)
    
    detections = detections[detections.class_id == 0]
    detections = tracker.update_with_detections(detections)

    mask = zone.trigger(detections=detections)

    if zone.current_count > 0 and not alarm_active:
        trigger_alarm()
        send_telegram_alert()
        alarm_active = True
    elif zone.current_count == 0 and alarm_active:
        clear_alarm()
        alarm_active = False

    frame = box_annotator.annotate(scene=frame, detections=detections)
    
    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        frame = label_annotator.annotate(
            scene=frame, detections=detections,
            labels=[f"#{tid}" for tid in detections.tracker_id]
        )
        
    frame = zone_annotator.annotate(scene=frame)

    if alarm_active:
        frame = draw_alarm_overlay(frame)

    end_time = time.time()
    latency_ms = (end_time - frame_spawn) * 1000
    print(f"Pipeline Latency: {latency_ms:.1f} ms | Effective FPS: {1000/latency_ms:.1f}")
    
    cv2.imshow("IP Camera - Continuous Stream Zone Alarm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader.stop() 
cv2.destroyAllWindows()
pygame.mixer.quit()
print("System disarmed and closed.")