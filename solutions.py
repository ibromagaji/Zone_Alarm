##this is the ultraltics version of my solution.py file. it uses the ultralytics solutions module to detect objects in a defined region and trigger an alarm sound when an object is detected.
import cv2
import pygame
from ultralytics import solutions

# Audio Setup
pygame.mixer.init()
try:
    pygame.mixer.music.load("/home/ju5ti5/detection/alarm.wav")
except:
    print("Sound load failed.")

# Initialize Security Alarm with Region
security = solutions.SecurityAlarm(
    model="/home/ju5ti5/detection/yolo_320.tflite",
    classes = [0], 
    region=[(342,413),(185, 564),(703,607),(929, 400)],
    records=1,
    show=True
)
security.authenticate("ibrahimdahirumagajii@gmail.com", "pkjk qxgd nmnf bwdi", "motogate105@gmail.com")

cap = cv2.VideoCapture("http://10.207.1.241:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

alarm_active = False

frame_count = 0
stride = 3

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame_count += 1

    if frame_count % stride == 0:

        results = security(frame)

    # if results is not None:
    #     cv2.imshow('Smooth AI Result', results)
    # else:
    #     cv2.imshow('Smooth AI Result', frame)

    # if cv2.waitkey(1) & 0xFF == ord('q'):
    #     break

    # Logic: Play sound if total_tracks > 0 (or use email_sent for one-time alert)
    if results.total_tracks >= 1 and not alarm_active:
        pygame.mixer.music.play(-1)
        alarm_active = True
    elif results.total_tracks == 0 and alarm_active:
        pygame.mixer.music.stop()
        alarm_active = False

cap.release()
cv2.destroyAllWindows()