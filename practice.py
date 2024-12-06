from ultralytics import YOLO
import cv2

model = YOLO("firearm.pt")

video_path = "./videos/gun.mp4"
cap = cv2.VideoCapture(video_path)
results = []

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.predict(frame, stream=True, device="0", show=True)
        # cv2.imshow("Frame", results.imgs[0])
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    else:
        break

cap.release()




   