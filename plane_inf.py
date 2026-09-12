import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
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

ALARM_SOUND_PATH = "/home/ju5ti5/zone/alarm.wav" 

try:
    pygame.mixer.music.load(ALARM_SOUND_PATH)
except Exception as e:
    print(f"Warning: Could not load sound file: {e}. Running without audio.")


def send_telegram_alert(frame, caption="Person detected!"):
    _, img_encoded = cv2.imencode('.jpg', frame)
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': ('detection.jpg', img_encoded.tobytes())}
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Telegram status: {response.status_code}")
    except Exception as e:
        print(f'Failed to send Telegram Alert: {e}')


def send_alert(frame, caption='Person Detected'):
    thread = threading.Thread(target=send_telegram_alert, args=(frame.copy(), caption))
    thread.daemon = True
    thread.start()


def trigger_alarm():
    print("ALARM: Person detected in frame!")
    try:
        pygame.mixer.music.play(-1) 
    except Exception as e:
        print(f"Failed to play sound: {e}")


def clear_alarm():
    print("Frame clear.")
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
VIDEO_PATH = "/home/ju5ti5/zone/passing.mp4"  # Update this to your video file path
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

print("Video stream active. Press 'q' to exit.")

# --- MODEL AND TRACKER INITIALIZATION ---
model = YOLO("/home/ju5ti5/zone/yolo_320.tflite")
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

alarm_active = False  
frames_in_zone = 0
frame_threshold = 60  # Require consecutive detection frames before triggering alarm


# --- MAIN PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    frame_spawn = time.time()
    
    if not ret or frame is None:
        print("End of video stream reached.")
        break

    results_generator = model(frame, stream=True)
    results = next(results_generator)
    
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter for class 0 (person)
    detections = detections[detections.class_id == 0]
    detections = tracker.update_with_detections(detections)

    # Check if any person is in the frame
    person_count = len(detections)

    if person_count > 0:
        if not alarm_active:
            frames_in_zone += 1
            if frames_in_zone >= frame_threshold:
                trigger_alarm()
                send_alert(frame, caption="Alert: Person continuously detected in video frame!")
                alarm_active = True
    else:
        frames_in_zone = 0
        if alarm_active:
            clear_alarm()
            alarm_active = False

    # Annotate bounding boxes and IDs
    frame = box_annotator.annotate(scene=frame, detections=detections)
    
    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        frame = label_annotator.annotate(
            scene=frame, detections=detections,
            labels=[f"#{tid}" for tid in detections.tracker_id]
        )

    if alarm_active:
        frame = draw_alarm_overlay(frame)

    end_time = time.time()
    latency_ms = (end_time - frame_spawn) * 1000
    fps = 1000 / latency_ms if latency_ms > 0 else 0
    print(f"Pipeline Latency: {latency_ms:.1f} ms | Effective FPS: {fps:.1f}")
    
    cv2.imshow("Video File - Full Frame Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("System disarmed and closed.")