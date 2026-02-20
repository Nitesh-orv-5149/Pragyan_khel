import cv2 as cv
from ultralytics import YOLO
import numpy as np
import math
import threading
import queue
import time


# ================= CONFIG =================
MODEL_PATH = "yolo26l-seg-opt.onnx"
CAMERA_ID = 0
BLUR_KERNEL = (15, 15)

FRAME_QUEUE = 5
RESULT_QUEUE = 5

MAX_DIST = 120
SMOOTH = 0.7


# ================= QUEUES =================
frame_queue = queue.Queue(maxsize=FRAME_QUEUE)
result_queue = queue.Queue(maxsize=RESULT_QUEUE)


# ================= INIT =================
model = YOLO(MODEL_PATH, task="segment")


cap = cv.VideoCapture(CAMERA_ID)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


# ================= STATE =================
locked = False
last_pos = None

current_tracks = []
running = True


# ================= CAMERA THREAD =================
def capture_worker():
    global running

    while running:
        ret, frame = cap.read()

        if not ret:
            continue

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass

        frame_queue.put(frame)


# ================= YOLO THREAD =================
def inference_worker():
    global running

    while running:

        if frame_queue.empty():
            time.sleep(0.002)
            continue

        frame = frame_queue.get()

        results = model.predict(
            frame,
            conf=0.25,
            iou=0.4,
            device=0,
            verbose=False
        )

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except:
                pass

        result_queue.put((frame, results))


# ================= MOUSE =================
def mouse_callback(event, x, y, flags, param):
    global locked, last_pos

    if event == cv.EVENT_LBUTTONDOWN:

        for (poly, box, centroid) in current_tracks:

            if cv.pointPolygonTest(
                poly, (float(x), float(y)), False
            ) >= 0:

                last_pos = np.array(centroid, dtype=float)
                locked = True
                print("TARGET LOCKED")
                return

        locked = False
        last_pos = None
        print("TARGET CLEARED")


# ================= WINDOW =================
cv.namedWindow("Pro-Precision Tracker")
cv.setMouseCallback("Pro-Precision Tracker", mouse_callback)


# ================= START THREADS =================
threading.Thread(target=capture_worker, daemon=True).start()
threading.Thread(target=inference_worker, daemon=True).start()


# ================= MAIN LOOP =================
prev = time.time()

while True:

    if result_queue.empty():
        time.sleep(0.002)
        continue

    frame, results = result_queue.get()
    current_tracks.clear()


    # ================= PARSE =================
    if results and results[0].masks is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks.xy

        for box, poly_pts in zip(boxes, masks):

            poly = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))

            M = cv.moments(poly)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(box[0]), int(box[1])

            current_tracks.append((poly, box, (cx, cy)))


    # ================= TRACK =================
    if locked and last_pos is not None and current_tracks:

        best = None
        best_d = float("inf")

        for (poly, box, centroid) in current_tracks:

            d = math.hypot(
                last_pos[0] - centroid[0],
                last_pos[1] - centroid[1]
            )

            if d < best_d and d < MAX_DIST:
                best_d = d
                best = centroid

        if best is not None:
            new_pos = np.array(best, dtype=float)
            last_pos = SMOOTH * last_pos + (1 - SMOOTH) * new_pos


    # ================= MASK =================
    display = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if locked and last_pos is not None:

        for (poly, box, centroid) in current_tracks:

            if math.hypot(
                last_pos[0] - centroid[0],
                last_pos[1] - centroid[1]
            ) < 40:
                cv.fillPoly(mask, [poly], 255)

        blurred = cv.GaussianBlur(frame, BLUR_KERNEL, 0)

        m3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        fg = cv.bitwise_and(frame, m3)
        bg = cv.bitwise_and(blurred, 255 - m3)

        display = cv.add(fg, bg)


    # ================= DRAW =================
    for (poly, box, centroid) in current_tracks:
        cv.polylines(display, [poly], True, (255, 255, 255), 1)

    if locked and last_pos is not None:

        px = int(last_pos[0])
        py = int(last_pos[1])

        cv.circle(display, (px, py), 6, (0, 0, 255), -1)

        cv.putText(
            display,
            "LOCKED",
            (px + 10, py),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ================= FPS =================
    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now

    cv.putText(
        display,
        f"FPS: {int(fps)}",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ================= SHOW =================
    cv.imshow("Pro-Precision Tracker", display)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


# ================= CLEANUP =================
running = False
cap.release()
cv.destroyAllWindows()
