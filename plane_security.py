from urllib import response
import alarm
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
home = os.getcwd()
config_file = os.path.join(home, 'zone_config.json')

bot_token = os.getenv('bot_token')
chat_id = os.getenv('chat_id')

#audio set up
pygame.mixer.init()
alarm_sound_path = '/home/ju5ti5/zone/alarm.wav'

try:
    pygame.mixer.music.load(alarm_sound_path)
except Exception as e:
    print(f'Error loading Alarm sound: {e}')

def _telegram_worker(frame,caption='Object Detected!!'):
    _, img_encoded = cv2.imencode('jpg', frame)
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    files = {'photo': ('detection.jpg', img_encoded.tobytes())}
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        response= requests.post(url, files=files, data=data)
        print(response.status_code)

    except Exception as e:
        print(f'Failed to send Telegram Alert: {e}')

def send_telegram_alert(frame,caption= 'Object Detected'):
    thread = threading.Thread(target=_telegram_worker,args=(frame.copy(),caption))
    thread.deamon = True
    thread.start()

def trigger_alarm():
    print('ALARM: Object In zone')
    try:
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f'Failed to play sound: {e}')

def clear_alarm():
    print('Zone clear.')
    pygame.mixer.music.stop()


class LatestFrameReader:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()

    def _read_frames(self):
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video stream.")
                break

            with self.lock:
                self.latest_frame = frame

        cap.release()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.stop_event.set()
        self.thread.join()

camera_url = ''

reader = LatestFrameReader(camera_url)
time.sleep(1.0)

# polygon = load_zone_coordinates(camera_url)
# if polygon is None:
#     print('Failed to initialize zone coordinates, Exiting.')
#     exit()

model = YOLO('')

box_anotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


alarm_active = False
frames_in_zone = 0
frame_threshold = 30

while True:
    frame = reader.read()
    if frame is None:
        continue

    results_generator = model(frame,stream=True)
    results = next(results_generator)

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]

    object_count = len(detections)

    if object_count > 0:
        if not alarm_active:
            frames_in_zone +=1
            if frames_in_zone >= frame_threshold and not alarm_active:
                trigger_alarm()
                send_telegram_alert(frame, caption='Alert: Object Detected!!')
                alarm_active = True

    else:
        frames_in_zone=0
        if alarm_active:
            clear_alarm()
            alarm_active = False

    #ANNOTATIONS
    frame = box_anotator.annotate(scene=frame, detections=detections)
    # if detections.tracker_id is not None and len(detections.tracker_id) > 0:
    #     frame = label_annotator.annotate(
    #         scene =frame, detections=detections,
    #         labels = [f'#{tid}' for tid in detections.tracker_id]
        #)
    #frame = zone_annotator.annotate(scene=frame)

    cv2.imshow('IP Camera',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('c'):
        clear_alarm()
        alarm_active = False
        reader.stop()
        cv2.destroyAllWindows()

        reader = LatestFrameReader(camera_url)
        time.sleep(1.0)
reader.stop()
cv2.destroyAllWindows()
pygame.mixer.quite()
print('System closed')

    




