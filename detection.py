import cv2 as cv
import numpy as np
from ultralytics import YOLO
import torch

class YoloTracker:
    def __init__(self, default_model="yolo26x-seg.pt"):
        self.device = 'cuda'
        self.model_name = default_model
        self.model = YOLO(default_model).to(self.device)
        self.selected_track_id = None
        self.blur_kernel = (25, 25)

    def switch_model(self, model_path):
        """Swaps the model (e.g., from X to Nano)"""
        if self.model_name != model_path:
            print(f"🔄 Swapping to {model_path}...")
            self.model = YOLO(model_path).to(self.device)
            self.model_name = model_path

    def process_frame(self, frame):
        results = self.model.track(
            frame, persist=True, tracker="botsort.yaml", 
            conf=0.25, iou=0.4, verbose=False
        )

        display = frame.copy()
        current_tracks_data = []

        if results and results[0].boxes.id is not None and results[0].masks is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            masks = results[0].masks.xy
            object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for tid, poly_pts in zip(ids, masks):
                poly = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
                current_tracks_data.append((tid, poly))

                if tid == self.selected_track_id:
                    cv.fillPoly(object_mask, [poly], 255)
                    cv.polylines(display, [poly], True, (0, 255, 0), 3)

            if self.selected_track_id is not None and np.any(object_mask):
                blurred = cv.GaussianBlur(frame, self.blur_kernel, 0)
                mask_3ch = cv.cvtColor(object_mask, cv.COLOR_GRAY2BGR)
                fg = cv.bitwise_and(frame, mask_3ch)
                bg = cv.bitwise_and(blurred, 255 - mask_3ch)
                display = cv.add(fg, bg)

        return display, current_tracks_data