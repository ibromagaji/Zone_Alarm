import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import time
import pygame 
import requests
import threading
from dotenv import load_dotenv

HOME = os.getcwd()
load_dotenv()

# Telegram configuration from .env
bot_token = os.getenv('bot_token')
chat_id = os.getenv('chat_id')

pygame.mixer.init()

alarm_path = "/home/ju5ti5/zone/alarm.wav" 

try:
    pygame.mixer.music.load(alarm_path)
except Exception as e:
    print(f"Warning: Could not load sound file: {e}. Running without audio.")


def send_telegram_alert(frame, caption="Object detected in zone!"):
    _, img_encoded = cv2.imencode('.jpg', frame)
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': ('detection.jpg', img_encoded.tobytes())}
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Telegram status: {response.status_code}")
    except Exception as e:
        print(f"Failed to send Telegram Alert: {e}")


def send_alert(frame, caption="Object Detected in Zone!"):
    # Non-blocking threaded call to prevent frame processing latency
    thread = threading.Thread(target=send_telegram_alert, args=(frame.copy(), caption))
    thread.daemon = True
    thread.start()


def trigger_alarm():
    print("ALARM: object entered the zone!")
    try:
        pygame.mixer.music.play(-1) 
    except Exception as e:
        print(f"Failed to play sound: {e}")


def clear_alarm():
    print("Zone clear.")
    pygame.mixer.music.stop()


def draw_alarm_overlay(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 15)
    cv2.putText(
        frame, "ALARM", (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA
    )
    return frame


# --- VIDEO SOURCE SETUP ---
VIDEO_PATH = "/home/ju5ti5/zone/passing.mp4"  # Replace with your video file path

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video loaded successfully. Press 'q' to exit.")

# --- MODEL AND TRACKING INITIALIZATION ---
model = YOLO("/home/ju5ti5/zone/yolo_320.tflite")
tracker = sv.ByteTrack()

# Zone corner points
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
frame_count = 0
stride = 3


# --- MAIN PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    frame_spawn = time.time()
    
    if not ret or frame is None:
        print("End of video file reached.")
        break

    frame_count += 1

    if frame_count % stride == 0:
        results_generator = model(frame, stream=True)
        results = next(results_generator)
        
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for class 0 (e.g. person)
        detections = detections[detections.class_id == 0]
        detections = tracker.update_with_detections(detections)

        mask = zone.trigger(detections=detections)

        if zone.current_count > 0 and not alarm_active:
            trigger_alarm()
            send_alert(frame, caption="Intruder detected in active zone!")
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

    print(f"Pipeline Latency: {latency_ms:.1f} ms | Effective FPS: {1000/latency_ms if latency_ms > 0 else 0:.1f}")

    cv2.imshow("Video File - Continuous Zone Alarm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("System disarmed and closed.")