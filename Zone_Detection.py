from urllib import response
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import requests
import os
from dotenv import load_dotenv
import pygame
import threading
import time
import json

load_dotenv()
HOME = os.getcwd()
CONFIG_FILE = os.path.join(HOME, "zone_config.json")

bot_token = os.getenv('bot_token')
chat_id = os.getenv('chat_id')

#audio set up
pygame.mixer.init()
ALARM_SOUND_PATH = "/home/ju5ti5/zone/alarm.wav" 

try:
    pygame.mixer.music.load(ALARM_SOUND_PATH)
except Exception as e:
    print(f"Warning: Could not load sound file: {e}. Running without audio.")

def _send_telegram_worker(frame, caption="Object detected!"):
    _, img_encoded = cv2.imencode('.jpg', frame)
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': ('detection.jpg', img_encoded.tobytes())}
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        response = requests.post(url, files=files, data=data)
        print(response.status_code, response.text)  
    except Exception as e:
        print(f'Failed to send Telegram Alert: {e}')

def send_telegram_alert(frame,caption='Object Detected'):
    t = threading.Thread(target=_send_telegram_worker,args=(frame.copy(),caption))
    t.deamon = True
    t.start()
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


#region selection
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"Point registered: [{x}, {y}]")


def interactive_zone_picker(stream_url):
    global clicked_points
    clicked_points = []
    
    print("\n--- ZONE CALIBRATION MODE ---")
    print("1. Click 4 points on the window to draw your detection zone.")
    print("2. Press 'ENTER' to save and start the system.")
    print("3. Press 'r' to reset points if you make a mistake.")
    
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    ret, calibration_frame = cap.read()
    cap.release()
    
    if not ret or calibration_frame is None:
        print("Error: Could not grab calibration frame from camera stream.")
        return None

    window_name = "Calibrate Detection Zone - Click 4 points"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_frame = calibration_frame.copy()
        
        # Draw the points and lines dynamically as the user clicks
        for i, pt in enumerate(clicked_points):
            cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if i > 0:
                cv2.line(display_frame, tuple(clicked_points[i-1]), tuple(pt), (0, 255, 0), 2)
        
        if len(clicked_points) == 4:
            cv2.line(display_frame, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Press Enter (13) to confirm
        if key == 13 and len(clicked_points) == 4:
            break
        # Press 'r' to reset coordinates
        elif key == ord('r'):
            clicked_points = []
            print("Points reset. Start clicking again.")
            
    cv2.destroyWindow(window_name)
    
    # Save coordinates locally to JSON file
    with open(CONFIG_FILE, 'w') as f:
        json.dump(clicked_points, f)
    print(f"Zone configurations successfully saved to {CONFIG_FILE}")
    
    return np.array(clicked_points)


def load_zone_coordinates(stream_url):
    # Check if config file exists, if not, force interactive setup
    if os.path.exists(CONFIG_FILE):
        print(f"Loading existing configurations from {CONFIG_FILE}")
        with open(CONFIG_FILE, 'r') as f:
            points = json.load(f)
        print("Loaded Points:", points)
        return np.array(points)
    else:
        print("No configuration file found. Launching setup wizard...")
        return interactive_zone_picker(stream_url)


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

camera_url = "http://10.210.75.210:8080/video"

reader = LatestFrameReader(camera_url)
time.sleep(1.0)

polygon = load_zone_coordinates(camera_url)
if polygon is None:
    print("Failed to initialize zone coordinates. Exiting.")
    exit()


model = YOLO("/home/ju5ti5/detection/yolo_320.tflite")
tracker = sv.ByteTrack()

zone = sv.PolygonZone(polygon=polygon)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

alarm_active = False  
frames_in_zone = 0
frame_threshold = 60

print("\nSystem Armed. Controls:\nPress 'q' to Quit\nPress 'c' to Re-Calibrate Zone coordinates on the fly.")

while True:
    frame = reader.read()
    if frame is None:
        continue

    results_generator = model(frame,stream=True) 
    results = next(results_generator)
    
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]
    detections = tracker.update_with_detections(detections)

    mask = zone.trigger(detections=detections)

    if zone.current_count > 0:
        if not alarm_active:
            frames_in_zone += 1
            if frames_in_zone >= frame_threshold:
                trigger_alarm()
                send_telegram_alert(frame, caption="Alert: Object detected in the zone!")
                alarm_active = True
    else :
        frames_in_zone=0
        if alarm_active:
            clear_alarm()
            alarm_active = False

    # Annotations
    frame = box_annotator.annotate(scene=frame, detections=detections)
    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        frame = label_annotator.annotate(
            scene=frame, detections=detections,
            labels=[f"#{tid}" for tid in detections.tracker_id]
        )
    frame = zone_annotator.annotate(scene=frame)

    if alarm_active:
        frame = draw_alarm_overlay(frame)

    cv2.imshow("IP Camera - Continuous Stream Zone Alarm", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Secret Key c Let's the user reset the zone live without code adjustments
    elif key == ord('c'):
        clear_alarm()
        alarm_active = False
        reader.stop()
        cv2.destroyAllWindows()
        
        polygon = interactive_zone_picker(camera_url)
        
        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)
        
        # Restart background video stream thread
        reader = LatestFrameReader(camera_url)
        time.sleep(1.0)

reader.stop()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("System disarmed and closed.")
