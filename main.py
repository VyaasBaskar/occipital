print("Starting imports...")
import cv2
import numpy as np
print("cv2, numpy - done.")
import time
import math
import os
print("utils - done.")
from ultralytics import YOLO
print("ultralytics - done.")
from V5comm import Detection, V5SerialComms, AIRecord
print("V5comm - done.")
print("Imports complete.\n")

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)

GPIO.setup(17, GPIO.OUT)

GPIO.output(17, GPIO.HIGH)
time.sleep(1.0)
GPIO.output(17, GPIO.LOW)

print("Constructing camera instance.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Done.")

print("Constructing Model.")
model = YOLO("./best_ncnn_model", task="detect")
print("Done.")

HFOV = 67.5
VFOV = 55

def HDISTORTION_CORRECTION(angle):
    return (1.0 / math.cos(0.5 * angle)) ** 1.5

comms = V5SerialComms()
time.sleep(1)
comms.start()

GPIO.output(17, GPIO.HIGH)

ctr = 0
tctr = time.time()
print("\nStarting...")
while True:
    ctr += 1
    t = time.time()
    ret, frame = cap.read()
    get_frame_time = time.time()

    if not ret:
        print("Error obtaining frame, retrying...")
        time.sleep(0.25)
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                break
        print("Reconstructing capture.")
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue
    frame = cv2.resize(frame, (256, 256))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(frame)
    # v = np.clip(16.0 * np.sqrt(v), 0, 255).astype(np.uint8)
    # s = np.clip(s * 1.1, 0, 255).astype(np.uint8)

    # frame = cv2.merge([h, s, v])

    # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    GPIO.output(17, GPIO.HIGH)
    results = model(frame, conf=0.25, imgsz=256, half=True)
    GPIO.output(17, GPIO.LOW)

    mgdets = []
    detections = []

    for result in results:
        for xywh, cls, conf in zip(result.boxes.xywh, result.boxes.cls, result.boxes.conf):
            conf = float(conf)
            if cls != 2 and conf < 0.5: continue

            x, y, w, h = map(float, xywh)
            # print(x, y, w, h)

            tx_l = (-128.0 + (x)) / 256.0 * HFOV
            tx_r = (-128.0 + (x + w)) / 256.0 * HFOV
            tx_l_rad = math.radians(tx_l)
            tx_r_rad = math.radians(tx_r)

            objsize = 7.0 if (int(cls) != 2) else 11.0

            d_ground = objsize / (math.tan(tx_r_rad) - math.tan(tx_l_rad))
            tx_c_rad = (tx_r_rad + tx_l_rad) / 2.0
            d_ground *= HDISTORTION_CORRECTION(tx_c_rad)

            ty_b_rad = -math.radians((-128.0 + (y)) / 256.0 * VFOV)
            height = d_ground * math.tan(ty_b_rad)

            print(f"Class {int(cls)}, Dist: {d_ground:.1f}, Height: {height:.1f}, Pxh: {y}")

            det = Detection(int(cls), tx_l, tx_r, d_ground, height, 0.0)
            if cls == 2:
                mgdets.append(det)
            else:
                detections.append(det)

    detections_parsed = []

    for det in detections:
        ign = False
        for mgdet in mgdets:
            if (mgdet.tx_l < det.tx_l < mgdet.tx_r or mgdet.tx_r > det.tx_r > mgdet.tx_l) and (mgdet.d < det.d + 12.0):
                print("Removing detection, overlapped by MG.")
                ign = True
                break
        if not ign:
            detections_parsed.append(det)

    detections_parsed += mgdets

    for det in detections_parsed:
        det.cam_latency = time.time() - t

    record = AIRecord(detections_parsed)
    comms.setDetectionData(record)

    print(detections_parsed)

    if ctr % 15 == 0:
        t = time.time()
        print("EFPS ", 15 / (t - tctr))
        tctr = t

cap.release()