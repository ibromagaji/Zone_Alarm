import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import tensorflow as tf
import os
import time
import pygame 

HOME = os.getcwd()

pygame.mixer.init()

alarm_path = "/home/ju5ti5/detection/alarm.wav" 

try:
    pygame.mixer.music.load(alarm_path)
except Exception as e:
    print(f"Warning: Could not load sound file: {e}. Running without audio.")



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



camera_url = "http://10.147.207.79:8080/video"

cap = cv2.VideoCapture(camera_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not open the IP camera stream. Verify network connection and URL.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Live stream active. Press 'q' to exit.")

model = YOLO("/home/ju5ti5/detection/yolo_320.tflite")
tracker = sv.ByteTrack()

#zone corner points
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


while True:
    ret, frame = cap.read()
    frame_spawn = time.time()
    if not ret:
        print("Failed to grab frame from stream.")
        break
    frame_count += 1


    if frame_count % stride ==0:

        results_generator = model(frame,stream=True)
        results = next(results_generator)
        
        detections = sv.Detections.from_ultralytics(results)
        
       
        detections = detections[detections.class_id == 0]
        detections = tracker.update_with_detections(detections)

        mask = zone.trigger(detections=detections)

        if zone.current_count > 0 and not alarm_active:
            trigger_alarm()
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

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("System disarmed and closed.")
