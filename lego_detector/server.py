import os
import json
import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import cv2
from lego_cam import LegoCameraDetector

detector = LegoCameraDetector("lego_classifier.pth")
latest_jpeg = None
jpeg_lock = threading.Lock()

# lego_cam.py의 루프를 로컬 스트리밍용으로 확장
def stream_loop(preferred_idx):
    global latest_jpeg
    order = [preferred_idx] + [i for i in [1, 0, 2] if i != preferred_idx]
    cap = None
    active_idx = preferred_idx
    for idx in order:
        cap = detector._open_camera(idx)
        if cap is not None:
            active_idx = idx
            break

    if cap is None:
        with detector.lock:
            detector.latest.update({"ok": False, "error": "연결된 카메라를 찾지 못했습니다."})
        detector.running = False
        return

    detector.cam_idx = active_idx
    frames, fps_t, fps = 0, time.time(), 0

    while detector.running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        frames += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = round(frames / (now - fps_t))
            frames, fps_t = 0, now

        boxes = detector._infer(frame)
        h, w, _ = frame.shape
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with jpeg_lock:
            latest_jpeg = buf.tobytes()

        with detector.lock:
            detector.latest = {
                "ok": True,
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
    with detector.lock:
        detector.latest["ok"] = False

def start_cam(idx):
    stop_cam()
    detector.running = True
    detector.thread = threading.Thread(target=stream_loop, args=(idx,), daemon=True)
    detector.thread.start()

def stop_cam():
    detector.running = False
    if detector.thread and detector.thread.is_alive():
        detector.thread.join(timeout=1.5)
    detector.thread = None

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/ui2.html"):
            html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui2.html")
            with open(html_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while detector.running:
                    with jpeg_lock:
                        jpg = latest_jpeg
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.04)
            except Exception:
                pass

        elif path == "/api/cam_start":
            qs = parse_qs(parsed.query)
            idx = int(qs.get("idx", ["1"])[0])
            start_cam(idx)
            self._send_json({"ok": True})

        elif path == "/api/cam_stop":
            stop_cam()
            self._send_json({"ok": True})

        elif path == "/api/cam_status":
            self._send_json(detector.cam_poll())

        elif path == "/api/shutdown":
            self._send_json({"ok": True})
            stop_cam()
            threading.Timer(0.3, lambda: os._exit(0)).start()

        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    port = 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"로컬 서버 실행 중: {url}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    server.serve_forever()
