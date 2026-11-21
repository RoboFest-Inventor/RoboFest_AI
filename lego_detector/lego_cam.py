import os
import cv2
import time
import math
import base64
import threading
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

CONF_THRESHOLD = 0.88  # 레고 판별 확신도 기준

class LegoCameraDetector:
    def __init__(self, model_path="lego_classifier.pth"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = model_path if os.path.isabs(model_path) else os.path.join(base_dir, model_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        checkpoint = torch.load(full_path, map_location=self.device)
        self.class_names = checkpoint["classes"]
        self.patch_size = checkpoint["patch_size"]
        self.input_size = checkpoint.get("input_size", 224)
        self.lego_idx = self.class_names.index("lego")

        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 단일 박스 흔들림 방지(시간축 안정화) 변수
        self.pos_streak = 0
        self.neg_streak = 0
        self.active_box = None

        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cam_idx = 1
        self.latest = {
            "ok": False,
            "frame": "",
            "w": 0,
            "h": 0,
            "fps": 0,
            "count": 0,
            "boxes": [],
            "cam_idx": 1,
            "error": ""
        }

    def _open_camera(self, idx):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(25):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            time.sleep(0.05)
        cap.release()
        return None

    def _stabilize_single_box(self, raw_box):
        """감지 시 1개 박스만 출력하고, 3초 정지 중 깜빡이지 않도록 유지"""
        if raw_box is not None:
            self.pos_streak += 1
            self.neg_streak = 0
            if self.active_box is None:
                self.active_box = raw_box
            else:
                # 박스 위치가 미세하게 떨리지 않도록 부드럽게 보정
                self.active_box = {
                    "x": int(self.active_box["x"] * 0.6 + raw_box["x"] * 0.4),
                    "y": int(self.active_box["y"] * 0.6 + raw_box["y"] * 0.4),
                    "w": raw_box["w"],
                    "h": raw_box["h"],
                    "prob": raw_box["prob"]
                }
            if self.pos_streak >= 2:
                return [self.active_box]
            return []
        else:
            self.neg_streak += 1
            self.pos_streak = 0
            # 잠깐 1~4프레임 놓치더라도 감지 상태를 안정적으로 유지
            if self.active_box is not None and self.neg_streak <= 4:
                return [self.active_box]
            self.active_box = None
            return []

    def _infer(self, frame):
        h, w, _ = frame.shape
        eff_patch = min(self.patch_size, h, w)
        stride = max(48, eff_patch // 3)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        patches, centers = [], []
        for y in range(0, h - eff_patch + 1, stride):
            for x in range(0, w - eff_patch + 1, stride):
                crop = cv2.resize(rgb[y:y + eff_patch, x:x + eff_patch], (self.input_size, self.input_size))
                patches.append(crop)
                centers.append((x + eff_patch / 2.0, y + eff_patch / 2.0))

        if not patches:
            return self._stabilize_single_box(None)

        batch = np.stack(patches).astype(np.float32) / 255.0
        batch = (batch - self.mean) / self.std
        tensor_batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_batch)
            probs = torch.softmax(logits, dim=1)[:, self.lego_idx].cpu().numpy()

        hits = [(cx, cy, float(p)) for (cx, cy), p in zip(centers, probs) if p >= CONF_THRESHOLD]
        if not hits:
            return self._stabilize_single_box(None)

        # 가장 확신도가 높은 지점을 기준으로 인접 반응을 하나로 합쳐 1개 박스만 생성
        best_cx, best_cy, max_prob = max(hits, key=lambda item: item[2])
        nearby = [item for item in hits if math.hypot(item[0] - best_cx, item[1] - best_cy) <= eff_patch * 0.9]

        total_w = sum(p ** 2 for _, _, p in nearby)
        avg_cx = sum(cx * (p ** 2) for cx, _, p in nearby) / total_w
        avg_cy = sum(cy * (p ** 2) for _, cy, p in nearby) / total_w

        half = eff_patch // 2
        bx = int(max(0, min(w - eff_patch, avg_cx - half)))
        by = int(max(0, min(h - eff_patch, avg_cy - half)))
        raw_box = {
            "x": bx,
            "y": by,
            "w": int(eff_patch),
            "h": int(eff_patch),
            "prob": round(float(max_prob), 3)
        }
        return self._stabilize_single_box(raw_box)

    def _loop(self, preferred_idx):
        order = [preferred_idx] + [i for i in [1, 0, 2] if i != preferred_idx]
        cap = None
        active_idx = preferred_idx
        for idx in order:
            cap = self._open_camera(idx)
            if cap is not None:
                active_idx = idx
                break

        if cap is None:
            with self.lock:
                self.latest.update({"ok": False, "error": "연결된 카메라를 찾지 못했습니다."})
            self.running = False
            return

        self.cam_idx = active_idx
        self.pos_streak = 0
        self.neg_streak = 0
        self.active_box = None
        frames = 0
        fps_t = time.time()
        fps = 0

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frames += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps = round(frames / (now - fps_t))
                frames = 0
                fps_t = now

            boxes = self._infer(frame)
            h, w, _ = frame.shape
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

            with self.lock:
                self.latest = {
                    "ok": True,
                    "frame": b64,
                    "w": int(w),
                    "h": int(h),
                    "fps": fps,
                    "count": len(boxes),
                    "boxes": boxes,
                    "cam_idx": active_idx,
                    "error": ""
                }
            time.sleep(0.02)

        cap.release()
        with self.lock:
            self.latest["ok"] = False

    def cam_list(self):
        return [
            {"id": "1", "label": "외장 웹캠 (카메라 #1)"},
            {"id": "0", "label": "기본 카메라 (카메라 #0)"},
            {"id": "2", "label": "보조 카메라 (카메라 #2)"},
        ]

    def cam_start(self, cam_idx=1):
        self.cam_stop()
        try:
            idx = int(cam_idx) if cam_idx not in (None, "") else 1
        except ValueError:
            idx = 1
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(idx,), daemon=True)
        self.thread.start()
        return {"ok": True}

    def cam_stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        return {"ok": True}

    def cam_poll(self):
        with self.lock:
            return dict(self.latest)
