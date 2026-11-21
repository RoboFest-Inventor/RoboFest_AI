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

CONF_THRESHOLD = 0.90  # 레고 판별 확신도 기준
MERGE_RATIO = 0.65     # 1명은 하나로 합치고, 떨어진 2명은 별도 인원으로 분리

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

    def _merge_detections(self, hits, patch_w):
        if not hits:
            return []
        hits = sorted(hits, key=lambda item: item[2], reverse=True)
        merge_dist = patch_w * MERGE_RATIO
        clusters = []

        for cx, cy, prob in hits:
            weight = prob ** 2
            matched = None
            for c in clusters:
                if math.hypot(cx - c["cx"], cy - c["cy"]) <= merge_dist:
                    matched = c
                    break
            if matched is not None:
                tw = matched["weight"] + weight
                matched["cx"] = (matched["cx"] * matched["weight"] + cx * weight) / tw
                matched["cy"] = (matched["cy"] * matched["weight"] + cy * weight) / tw
                matched["weight"] = tw
                matched["max_prob"] = max(matched["max_prob"], prob)
            else:
                clusters.append({"cx": cx, "cy": cy, "weight": weight, "max_prob": prob})
        return clusters

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
            return []

        batch = np.stack(patches).astype(np.float32) / 255.0
        batch = (batch - self.mean) / self.std
        tensor_batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_batch)
            probs = torch.softmax(logits, dim=1)[:, self.lego_idx].cpu().numpy()

        hits = [(cx, cy, float(p)) for (cx, cy), p in zip(centers, probs) if p >= CONF_THRESHOLD]
        merged = self._merge_detections(hits, eff_patch)

        half = eff_patch // 2
        boxes = []
        for item in merged:
            bx = int(max(0, min(w - eff_patch, item["cx"] - half)))
            by = int(max(0, min(h - eff_patch, item["cy"] - half)))
            boxes.append({
                "x": bx,
                "y": by,
                "w": int(eff_patch),
                "h": int(eff_patch),
                "prob": round(float(item["max_prob"]), 3)
            })
        return boxes

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

    # --- pywebview JS API 노출 메서드 ---
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


# 단독 실행 테스트용 (기존 로봇 백엔드 없이 바로 UI + 카메라를 띄울 때 사용)
if __name__ == "__main__":
    import webview

    class StandaloneApi(LegoCameraDetector):
        def __init__(self):
            super().__init__()
            self.robots = [{"serial": "RX200-001", "name": "로봇 A", "model": "ROBO-X200",
                            "battery": 85, "voltage": 12.1, "moving": False, "age": None,
                            "gen": 0, "odo": {"x": 0, "y": 0, "th": 0, "l": 0, "r": 0}, "hist": []}]

        def list_ports(self): return []
        def connect(self, port, baud): return {"ok": False, "error": "단독 카메라 테스트 모드"}
        def disconnect(self): return {"ok": True}
        def set_channel(self, ch): return {"ok": True}
        def remove_robot(self, sn): return {"ok": True}
        def start_test(self, sn): return {"ok": False, "error": "단독 모드"}
        def register(self, m, s, n, memo): return {"ok": False, "error": "단독 모드"}
        def set_params(self, t, d, b): return {"ok": True}
        def inject(self, sn, l, r, add): return {"ok": True}
        def reset_pose(self, sn): return {"ok": True}
        def get_state(self):
            return {
                "now": time.time(),
                "rx": {"state": "off", "age": None, "error": "", "mode": "standalone"},
                "robots": self.robots,
                "log": [],
                "params": {"tpr": 360, "dia": 0.065, "base": 0.16}
            }

    api = StandaloneApi()
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui2.html")
    win = webview.create_window("Robofest - LEGO Detector", html_path, js_api=api, width=1280, height=820)
    webview.start()
