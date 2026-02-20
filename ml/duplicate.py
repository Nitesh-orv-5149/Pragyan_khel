import cv2
import os 
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def points_collection(file_name):
    video_source = r"C:\\Users\\PBuva\\Downloads\\Hackathon_prgy\\ui\\{}".format(file_name)

    click_point = None 
    clicked_id = None 

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    def mouse_action(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x, y)
            print(f"Clicked at: {click_point}")

    cap = cv2.VideoCapture(video_source)

    if not os.path.exists(video_source):
        print(f"ERROR: I can't find the file '{video_source}'")
        print(f"Current Folder: {os.getcwd()}")
        return None
    else:
        print("File found! Opening video...")

    cv2.namedWindow("My Video")
    cv2.setMouseCallback("My Video", mouse_action)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        results = model(frame)
        
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                w = x2 - x1
                h = y2 - y1
                detections.append(([[float(x1), float(y1), float(w), float(h)], conf, cls]))

    
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:                    
            if not track.is_confirmed():
                continue
            
            l, t, r, b = track.to_ltrb()
            l, t, r, b = int(l), int(t), int(r), int(b)
            track_id = track.track_id

    
            if click_point is not None:
                cx, cy = click_point
                if l <= cx <= r and t <= cy <= b:
                    clicked_id = track_id
                    print(f"✅ Selected ID: {clicked_id}")
                    click_point = None  

            color = (100, 100, 100)
            if clicked_id == track_id:
                color = (0, 255, 0)

            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("My Video", frame)          

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return clicked_id 


if __name__ == "__main__":
    selected_id = points_collection('my_video.mp4')
    print(f"THE CLICKED ID IS: {selected_id}")