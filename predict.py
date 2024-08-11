from fastapi import FastAPI, UploadFile, File
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

app = FastAPI()

# Initiate YOLO model
model = YOLO("models/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

limitLine = [(247, 387), (674, 633)]

@app.get("/")
def read_root():
    return {"message": "People Counting API is running!"}

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    cap = cv2.VideoCapture(file.file)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('videos/output/video_output.mp4', fourcc, 20.0, (1280, 720))

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    totalCountIn = 0
    totalCountOut = 0

    person_status = {}

    def is_point_in_line(line, point):
        dist = cv2.pointPolygonTest(np.array(line, np.int32), point, True)
        return abs(dist) < 5

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (1280, 720))

        results = model(img, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "person" and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        cv2.line(img, limitLine[0], limitLine[1], (0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if id not in person_status:
                person_status[id] = {'in': False, 'out': False, 'initial_pos': (cx, cy)}

            previous_pos = person_status[id].get('current_pos', person_status[id]['initial_pos'])
            person_status[id]['current_pos'] = (cx, cy)

            if is_point_in_line(limitLine, (cx, cy)):
                if previous_pos[0] < cx:
                    if not person_status[id]['in']:
                        totalCountIn += 1
                        person_status[id]['in'] = True
                    person_status[id]['out'] = False
                    cv2.line(img, limitLine[0], limitLine[1], (0, 255, 0), 5)
                elif previous_pos[0] > cx:
                    if not person_status[id]['out']:
                        totalCountOut += 1
                        person_status[id]['out'] = True
                    person_status[id]['in'] = False
                    cv2.line(img, limitLine[0], limitLine[1], (0, 255, 0), 5)

        cv2.putText(img, f"{totalCountIn} people", (874, 415), cv2.FONT_HERSHEY_PLAIN, 1.5, (139, 195, 75), 2)
        cv2.putText(img, f"{totalCountOut} people", (1071, 415), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 50, 230), 2)

        out.write(img)

    cap.release()
    out.release()

    return {"message": "Video processed successfully", "total_in": totalCountIn, "total_out": totalCountOut}
