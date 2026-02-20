import cv2 as cv
from ultralytics import YOLO

model = YOLO("ml/yolov8n.pt") 

vid_path = "assets/sample-inp-1.mp4"
cap = cv.VideoCapture(vid_path)

if not cap.isOpened():
    print("Error: Could not open video file. Check your path!")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("End of video or cannot read frame.")
        break



    results = model.predict(source=frame, conf=0.25, verbose=False)
    

    annotated_frame = results[0].plot()


    cv.imshow("YOLO Live Detection", annotated_frame)


    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()