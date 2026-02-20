import cv2
import os 



def points_collection(file_name):
    video_source = r"C:\\Users\\PBuva\\Downloads\\Hackathon_prgy\\ui\\{}".format(file_name)

    clicked_points=[]

    def mouse_action(event, x, y, flags, param):

       if event == cv2.EVENT_LBUTTONDOWN:
          clicked_points.append((x,y))

    cap = cv2.VideoCapture(video_source)

    if not os.path.exists(video_source):
        print(f" ERROR: I can't find the file '{video_source}'")
        print(f"Current Folder: {os.getcwd()}")
    else:
        print("File found! Opening video...")

    cv2.namedWindow("My Video")
    cv2.setMouseCallback("My Video", mouse_action)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            cv2.imshow("My Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return clicked_points


if __name__=="__main__":
    points_lst=points_collection('my_video.mp4')
    print(points_lst)

