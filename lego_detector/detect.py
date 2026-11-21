import cv2
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

CONF_THRESHOLD = 0.90  # 레고 판별 기준 (90% 이상)
MERGE_RATIO = 0.90     # 상자 크기 대비 이 거리 안의 검출 결과는 한 명으로 병합

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load("lego_classifier.pth", map_location=device)
class_names = checkpoint["classes"]
patch_size = checkpoint["patch_size"]
input_size = checkpoint.get("input_size", 224)
lego_idx = class_names.index("lego")

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device).eval()

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def try_open_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        time.sleep(0.05)
    cap.release()
    return None

def merge_nearby_detections(hits, patch_w, merge_ratio):
    """같은 레고 주변에서 여러 칸이 동시에 반응할 때 중심을 평균 내어 1개의 박스로 통합"""
    if not hits:
        return []

    # 신뢰도가 높은 순서대로 정렬
    hits = sorted(hits, key=lambda item: item[2], reverse=True)
    merge_dist = patch_w * merge_ratio
    clusters = []

    for cx, cy, prob in hits:
        weight = prob ** 2
        matched_cluster = None
        for cluster in clusters:
            dist = math.hypot(cx - cluster["cx"], cy - cluster["cy"])
            if dist <= merge_dist:
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            total_w = matched_cluster["weight"] + weight
            matched_cluster["cx"] = (matched_cluster["cx"] * matched_cluster["weight"] + cx * weight) / total_w
            matched_cluster["cy"] = (matched_cluster["cy"] * matched_cluster["weight"] + cy * weight) / total_w
            matched_cluster["weight"] = total_w
            matched_cluster["max_prob"] = max(matched_cluster["max_prob"], prob)
        else:
            clusters.append({
                "cx": cx,
                "cy": cy,
                "weight": weight,
                "max_prob": prob
            })

    return clusters

current_cam = 1
cap = None
for idx in [1, 0, 2]:
    print(f"{idx}번 카메라 연결 시도 중...")
    cap = try_open_camera(idx)
    if cap is not None:
        current_cam = idx
        print(f"-> {current_cam}번 카메라 연결 성공!")
        break

if cap is None:
    print("[오류] 연결된 카메라를 찾지 못했습니다.")
    exit()

print("=== 실행 완료! ('c': 카메라 전환 / 'q': 종료) ===")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.05)
        continue

    h, w, _ = frame.shape
    eff_patch = min(patch_size, h, w)
    stride = max(48, eff_patch // 3)  # 더 촘촘하게 훑어서 레고 정중앙을 정확히 포착
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    patches = []
    centers = []
    for y in range(0, h - eff_patch + 1, stride):
        for x in range(0, w - eff_patch + 1, stride):
            crop = rgb[y:y + eff_patch, x:x + eff_patch]
            crop = cv2.resize(crop, (input_size, input_size))
            patches.append(crop)
            centers.append((x + eff_patch / 2.0, y + eff_patch / 2.0))

    batch = np.stack(patches).astype(np.float32) / 255.0
    batch = (batch - mean) / std
    tensor_batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        logits = model(tensor_batch)
        probs = torch.softmax(logits, dim=1)[:, lego_idx].cpu().numpy()

    hits = [(cx, cy, float(prob)) for (cx, cy), prob in zip(centers, probs) if prob >= CONF_THRESHOLD]
    merged = merge_nearby_detections(hits, eff_patch, MERGE_RATIO)

    detected_count = len(merged)
    half = eff_patch // 2
    for item in merged:
        bx = int(max(0, min(w - eff_patch, item["cx"] - half)))
        by = int(max(0, min(h - eff_patch, item["cy"] - half)))
        cv2.rectangle(frame, (bx, by), (bx + eff_patch, by + eff_patch), (0, 255, 0), 3)
        cv2.putText(frame, f"LEGO {item['max_prob']*100:.0f}%", (bx, max(25, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    status = f"DETECTED: {detected_count} LEGO" if detected_count > 0 else "EMPTY: No LEGO"
    color = (0, 255, 0) if detected_count > 0 else (0, 0, 255)
    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, f"Cam #{current_cam} | 'c': Switch Cam | 'q': Quit", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("PyTorch Sliding-Window LEGO Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("c"), ord("C")):
        cap.release()
        for step in range(1, 4):
            next_idx = (current_cam + step) % 3
            new_cap = try_open_camera(next_idx)
            if new_cap is not None:
                cap = new_cap
                current_cam = next_idx
                break
    elif key in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
