import cv2 as cv
from ultralytics import YOLO
import numpy as np
import math

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11s-seg.pt"   # Segmentation model
CAMERA_ID = 0
BLUR_KERNEL = (15, 15)         # Increase for stronger blur

# ---------------- INIT ----------------
model = YOLO(MODEL_PATH)   # Auto uses GPU if available, else CPU
cap = cv.VideoCapture(CAMERA_ID)

model.to("cuda")

# Reduce resolution for speed (important on CPU)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- STATE ----------------
selected_track_id = None
last_centroid = None
current_tracks = []   # (id, poly, box, centroid)


# ---------------- MOUSE ----------------
def mouse_callback(event, x, y, flags, param):
    global selected_track_id, last_centroid

    if event == cv.EVENT_LBUTTONDOWN:

        for (track_id, poly, bbox, centroid) in current_tracks:

            inside = cv.pointPolygonTest(
                poly, (float(x), float(y)), False
            )

            if inside >= 0:
                selected_track_id = track_id
                last_centroid = centroid
                print(f"TARGET LOCKED → ID {track_id}")
                return

        # If click on empty area
        selected_track_id = None
        last_centroid = None
        print("TARGET CLEARED")


cv.namedWindow("Pro-Precision Tracker")
cv.setMouseCallback("Pro-Precision Tracker", mouse_callback)



while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    # ---------------- YOLO TRACK ----------------
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.25,     # detect smaller objects
        iou=0.4,       # allow more overlap
        verbose=False
    )

    current_tracks.clear()

    # ---------------- EXTRACT TRACKS ----------------
    if (
        results
        and results[0].boxes.id is not None
        and results[0].masks is not None
    ):

        ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        masks = results[0].masks.xy

        for tid, box, poly_pts in zip(ids, boxes, masks):

            poly = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))

            # Centroid
            M = cv.moments(poly)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = box[0], box[1]

            current_tracks.append(
                (tid, poly, box, (cx, cy))
            )


    # ---------------- ID RECOVERY ----------------
    if selected_track_id is not None and last_centroid is not None:

        id_found = False
        candidates = []

        for (tid, poly, bbox, centroid) in current_tracks:

            if tid == selected_track_id:
                last_centroid = centroid
                id_found = True
                break

            dist = math.hypot(
                last_centroid[0] - centroid[0],
                last_centroid[1] - centroid[1]
            )

            if dist < 50:
                candidates.append((dist, tid, centroid))

        if not id_found and candidates:

            candidates.sort()
            selected_track_id = candidates[0][1]
            last_centroid = candidates[0][2]


    # ---------------- BLUR BACKGROUND ----------------
    display = frame.copy()

    # Create mask
    object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for (tid, poly, bbox, centroid) in current_tracks:

        if tid == selected_track_id:
            cv.fillPoly(object_mask, [poly], 255)


    if selected_track_id is not None:

        # Blur whole frame
        blurred = cv.GaussianBlur(frame, BLUR_KERNEL, 0)

        # Convert mask
        mask_3ch = cv.cvtColor(
            object_mask, cv.COLOR_GRAY2BGR
        )

        # Foreground
        fg = cv.bitwise_and(frame, mask_3ch)

        # Background
        bg = cv.bitwise_and(
            blurred, 255 - mask_3ch
        )

        display = cv.add(fg, bg)


    # ---------------- DRAW OUTLINES ----------------
    for (tid, poly, bbox, centroid) in current_tracks:

        x1, y1, x2, y2 = bbox

        if tid == selected_track_id:

            cv.polylines(display, [poly], True,
                         (0, 255, 0), 3)

            cv.putText(
                display,
                "LOCKED",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv.polylines(display, [poly], True,
                         (255, 255, 255), 1)


    # ---------------- SHOW ----------------
    cv.imshow("Pro-Precision Tracker", display)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------- CLEANUP ----------------
cap.release()
cv.destroyAllWindows()
