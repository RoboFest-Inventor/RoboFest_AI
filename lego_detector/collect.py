import cv2
import os
import json
import time
import random

PATCH_SIZE = 300  # 기본 상자 크기를 300으로 대폭 확대
TARGET_COUNT = 300

os.makedirs("dataset/lego", exist_ok=True)
os.makedirs("dataset/no_lego", exist_ok=True)

def save_config(size):
    with open("config.json", "w") as f:
        json.dump({"patch_size": size}, f)

save_config(PATCH_SIZE)

mouse_x, mouse_y = 640, 360
current_frame = None

def save_crop(frame, cx, cy, folder, prefix, use_scale_jitter=True):
    h, w, _ = frame.shape
    scale_size = random.randint(int(PATCH_SIZE * 0.92), int(PATCH_SIZE * 1.08)) if use_scale_jitter else PATCH_SIZE
    scale_size = min(scale_size, h, w)
    half = scale_size // 2

    x1 = max(0, min(w - scale_size, cx - half))
    y1 = max(0, min(h - scale_size, cy - half))
    crop = frame[y1:y1 + scale_size, x1:x1 + scale_size]
    crop = cv2.resize(crop, (224, 224))

    filename = f"{folder}/{prefix}_{int(time.time() * 1000)}_{random.randint(100,999)}.jpg"
    cv2.imwrite(filename, crop)

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, current_frame
    mouse_x, mouse_y = x, y
    if current_frame is None:
        return

    if event == cv2.EVENT_RBUTTONDOWN:
        for _ in range(5):
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            save_crop(current_frame, x + dx, y + dy, "dataset/no_lego", "bg")
        print(f"[저장] 특정 배경 +5장 (현재 {len(os.listdir('dataset/no_lego'))}장)")

print("맥북 카메라를 켜는 중입니다...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for _ in range(50):
    ret, frame = cap.read()
    if ret and frame is not None:
        break
    time.sleep(0.1)
else:
    print("\n[오류] 카메라 화면을 불러오지 못했습니다.")
    cap.release()
    exit()

cv2.namedWindow("Data Collector")
cv2.setMouseCallback("Data Collector", on_mouse)

print("\n=== 카메라 연결 완료! (입력기를 영문 ABC로 맞춰주세요) ===")
print("1. '+'(또는 ']') 키: 초록 상자 더 크게 / '-'(또는 '[') 키: 상자 작게")
print("2. 레고 위에 커서를 대고 'v' 키: 5장씩 저장")
print("3. 빈 보드판에서 'b' 키: 배경 10장씩 저장")
print("4. 'q' 키: 수집 종료\n")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.05)
        continue

    current_frame = frame.copy()
    display = frame.copy()
    h, w, _ = display.shape

    half = PATCH_SIZE // 2
    cv2.rectangle(display, (mouse_x - half, mouse_y - half), (mouse_x + half, mouse_y + half), (0, 255, 0), 2)

    lego_cnt = len(os.listdir("dataset/lego"))
    bg_cnt = len(os.listdir("dataset/no_lego"))

    lego_status = f"LEGO: {lego_cnt}/{TARGET_COUNT}+" + (" (OK)" if lego_cnt >= TARGET_COUNT else "")
    bg_status = f"BG: {bg_cnt}/{TARGET_COUNT}+" + (" (OK)" if bg_cnt >= TARGET_COUNT else "")

    cv2.putText(display, f"{lego_status} | {bg_status} | Box: {PATCH_SIZE}px", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0) if (lego_cnt >= TARGET_COUNT and bg_cnt >= TARGET_COUNT) else (0, 255, 255), 2)
    cv2.putText(display, "'+' / '-': Box Size | 'v': +5 LEGO | 'b': +10 BG | 'q': Quit", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Data Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("="), ord("+"), ord("]")):
        PATCH_SIZE = min(min(h, w) - 20, PATCH_SIZE + 20)
        save_config(PATCH_SIZE)
        print(f"상자 크기 확대: {PATCH_SIZE}px")

    elif key in (ord("-"), ord("_"), ord("[")):
        PATCH_SIZE = max(120, PATCH_SIZE - 20)
        save_config(PATCH_SIZE)
        print(f"상자 크기 축소: {PATCH_SIZE}px")

    elif key in (ord("v"), ord("V")):
        for _ in range(5):
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            save_crop(current_frame, mouse_x + dx, mouse_y + dy, "dataset/lego", "lego", use_scale_jitter=True)
        print(f"[저장] 레고 +5장 -> 총 {len(os.listdir('dataset/lego'))}장")

    elif key in (ord("b"), ord("B")):
        for _ in range(10):
            rx = random.randint(half, w - half)
            ry = random.randint(half, h - half)
            save_crop(current_frame, rx, ry, "dataset/no_lego", "bg", use_scale_jitter=True)
        print(f"[저장] 배경 +10장 -> 총 {len(os.listdir('dataset/no_lego'))}장")

    elif key in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
