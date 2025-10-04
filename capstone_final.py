import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import time
import threading
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import numpy as np
import os
import random
import psutil
import pandas as pd

# YOLOv8 모델 로드
yolo_model = YOLO("final.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# InsightFace 로드
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

frame_count = 0
lock = threading.Lock()
last_results = []

ad_playing = False
ad_end_time = 0
current_ad_frame = np.zeros((480, 640, 3), dtype=np.uint8)

ad_images = {}
ad_root = "ads"

def load_ad_images():
    for folder in os.listdir(ad_root):
        full_path = os.path.join(ad_root, folder)
        if not os.path.isdir(full_path):
            continue
        ad_images[folder] = []
        for file in os.listdir(full_path):
            if file.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(full_path, file))
                if img is not None:
                    ad_images[folder].append(img)

load_ad_images()

def get_age_group(age):
    if age < 13:
        return "0~12"
    elif age < 20:
        return "13~19"
    elif age < 40:
        return "20~39"
    elif age < 60:
        return "40~59"
    else:
        return "60+"

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def insightface_analysis(frame, yolo_boxes):
    global last_results
    try:
        faces = app.get(frame)
        results = [{} for _ in range(len(yolo_boxes))]

        for i, yolo_box in enumerate(yolo_boxes):
            best_iou = 0
            best_match = None
            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                iou = compute_iou(yolo_box, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = face

            if best_iou > 0.3 and best_match is not None:
                age = best_match.age
                gender = "female" if best_match.gender == 0 else "male"

                # 나이 보정
                if gender == "male" and 40 <= age <= 44:
                    age -= 4
                elif gender == "female" and 30 <= age <= 59:
                    age += 10

                age_group = get_age_group(age)
                results[i] = {"gender": gender, "age_group": age_group}

        with lock:
            last_results = results
    except Exception as e:
        print("[Error] InsightFace failed:", e)

def fade_in_ad(next_ad_img):
    global current_ad_frame

    steps = 10
    delay = 0.03

    prev_img = current_ad_frame.copy()
    next_img = cv2.resize(next_ad_img, (640, 480))

    for i in range(1, steps + 1):
        alpha = i / steps
        blended = cv2.addWeighted(next_img, alpha, prev_img, 1 - alpha, 0)
        current_ad_frame = blended.copy()
        cv2.imshow("Advertisement", current_ad_frame)
        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
            break

def play_ad(age_group=None, gender=None, is_common=False):
    global ad_playing, ad_end_time, current_ad_frame, last_ad_start

    folder_name = "common" if is_common else f"{age_group}_{gender}"
    candidates = ad_images.get(folder_name, [])
    if not candidates:
        print(f"[경고] 광고 이미지 없음: {folder_name}")
        return

    next_ad_img = random.choice(candidates)
    fade_in_ad(next_ad_img)
    ad_playing = True
    ad_end_time = time.time() + 4
    last_ad_start = time.time()

measurement_data = []
start_time = time.time()
prev_time = time.time()
ad_active_time = 0.0
last_ad_start = None
cpu_energy_Wh = 0.0

cv2.namedWindow("Advertisement", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Advertisement", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    delta_sec = current_time - prev_time
    prev_time = current_time

    frame_count += 1

    results = yolo_model(frame, conf=0.4, iou=0.5, verbose=False)
    yolo_boxes = [(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                  for r in results for box in r.boxes.xyxy]

    if frame_count % 10 == 0 and len(yolo_boxes) > 0:
        threading.Thread(target=insightface_analysis, args=(frame.copy(), yolo_boxes)).start()

    with lock:
        for i, (x1, y1, x2, y2) in enumerate(yolo_boxes):
            info = last_results[i] if i < len(last_results) else {}
            gender = info.get("gender", "Unknown")
            age_group = info.get("age_group", "??")
            label = f"{gender}, {age_group}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    now = time.time()


    if ad_playing and now < ad_end_time:
        pass
    elif ad_playing and now >= ad_end_time:
        if last_ad_start:
            ad_active_time += (now - last_ad_start)
        ad_playing = False
        last_ad_start = None
        current_ad_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    elif len(yolo_boxes) > 0:
        with lock:
            if len(yolo_boxes) >= 4: # 4명이상 공용광고 출력
                play_ad(is_common=True)
            elif len(last_results) > 0:
                person = last_results[0]
                if "gender" in person and "age_group" in person:
                    play_ad(person["age_group"], person["gender"])

    cpu_usage = psutil.cpu_percent()
    cpu_power_estimate = 65 * (cpu_usage / 100)
    total_power = cpu_power_estimate

    cpu_energy_Wh += cpu_power_estimate * (delta_sec / 3600)

    measurement_data.append({
        "timestamp": now - start_time,
        "cpu_usage_percent": cpu_usage,
        "estimated_cpu_power_w": cpu_power_estimate,
        "estimated_cpu_energy_Wh": cpu_energy_Wh,
        "estimated_total_power_w": total_power
    })

    #cv2.imshow("YOLOv8 + InsightFace", frame) #yolo 화면출력
    ad_resized = cv2.resize(current_ad_frame, (640, 480))
    cv2.imshow("Advertisement", ad_resized) # 광고화면 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(measurement_data)
df.to_csv("power_measurement_log.csv", index=False)
print("[저장 완료] power_measurement_log.csv")

# 최종 요약 출력
total_runtime = time.time() - start_time
ad_percent = (ad_active_time / total_runtime) * 100
avg_cpu = df["cpu_usage_percent"].mean()
total_energy = df["estimated_cpu_energy_Wh"].iloc[-1]

avg_cpu_power = 65 * (avg_cpu / 100)
ad_energy_Wh = avg_cpu_power * (ad_active_time / 3600)

print(f"\n[실행 요약]")
print(f"전체 실행 시간: {total_runtime:.2f}초")
print(f"광고 재생 시간: {ad_active_time:.2f}초 ({ad_percent:.2f}%)")
print(f"평균 CPU 사용률: {avg_cpu:.2f}%")
print(f"총 CPU 에너지 소비량 (전체 실행 기준): {total_energy:.4f} Wh")
print(f"광고 재생 시간 기준 CPU 에너지 소비량: {ad_energy_Wh:.4f} Wh")
