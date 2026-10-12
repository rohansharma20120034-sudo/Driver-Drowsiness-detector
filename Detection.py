import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from pygame import mixer

mixer.init()
try:
    alarm_sound = mixer.Sound("alarm.wav")
except:
    alarm_sound = None

class EyeClassifier(nn.Module):
    def __init__(self):
        super(EyeClassifier, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc_net(x)
        return x

model = EyeClassifier()
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
COUNTER = 0
DROWSY_LIMIT = 15
STATUS = "Active"

print("[INFO] Starting Drowsiness Detection System. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            eyes_detected = True
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            
            try:
                eye_resized = cv2.resize(eye_roi, (24, 24))
                eye_normalized = eye_resized / 255.0
                tensor_eye = torch.FloatTensor(eye_normalized).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(tensor_eye)
                    _, predicted = torch.max(outputs, 1)
                    prediction = predicted.item()

                color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)

                if prediction == 0:
                    COUNTER += 1
                    if COUNTER >= DROWSY_LIMIT:
                        STATUS = "DROWSY ALERT!"
                        if alarm_sound:
                            alarm_sound.play()
                else:
                    COUNTER = max(0, COUNTER - 1)
                    if COUNTER == 0:
                        STATUS = "Active"

            except Exception as e:
                pass

    color_status = (0, 0, 255) if STATUS == "DROWSY ALERT!" else (0, 255, 0)
    cv2.putText(frame, f"Status: {STATUS}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
    cv2.putText(frame, f"Closed Frames Counter: {COUNTER}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection (PyTorch)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
