import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

# Load the YOLO model
model = YOLO("firearm.pt")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Run YOLO inference on the frame
        results = model.predict(img, stream=True, device="cpu")
        for result in results:
            img = result.plot()
        return img

# Define the display window size
display_width = 800

# Streamlit UI elements

st.title("Object Detection with YOLO")
st.write("This is a simple object detection application using YOLOv11.")
col1, col2 = st.columns([1, 1])

col1.title("Select a Model")

selectedDevice = col1.radio(
    "What Object Detection Model would you like to use?",
    ["Firearm", "YoloV11"],
    help="Select the model for object detection."
)

if selectedDevice == "Firearm":
    model = YOLO("firearm.pt")
else:
    model = YOLO("yolo11n.pt")

col2.title("Select a Video Source")
selectedDevice = col2.radio(
    "What input video source would you like to use?",
    ["Upload a video", "Sample Video"],
    help="Select the video source for object detection."
)



if selectedDevice == "Upload a video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

btnCol1, btnCol2 = st.columns([1, 1])

quit = btnCol1.button("Quit", type="secondary", use_container_width=True)

if selectedDevice != "Webcam":
    start = btnCol2.button("Start", type="primary", use_container_width=True)

    if start:
        start_time = time.time()
        video_path = None
        if selectedDevice == "Upload a video" and uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                video_path = temp_file.name
        elif selectedDevice == "Sample Video":
            video_path = "./gun2.mp4"
        
        # Process video if available
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                display_height = int(display_width * frame_height / frame_width)

                # Use an empty slot to overwrite frames continuously
                frame_display = st.empty()
                count=0
                while cap.isOpened():
                    success, frame = cap.read()
                    if success:
                        # Run YOLO inference on the frame
                        results = model.predict(frame, stream=True, device="cpu", )
                        for result in results:
                            annotated_frame = result.plot()
                            if result.boxes:
                                print("Found {} objects".format(len(result.boxes)))
                                
                                if (len(result.boxes) > 0):
                                    if (time.time() - start_time > 5):
                                        print("Found a gun!")
                                        count += len(result.boxes)
                                        start_time = time.time()
                                    
                            # Display the frame in the same slot for continuous updates
                            frame_display.image(annotated_frame, channels="BGR", width=display_width)

                        # Check for quit button to exit
                        if quit:
                            break
                    else:
                        break

            if (count > 0):
                print(f"Guns detected: {count}")
                st.error("Found a gun!")

                # Release resources
                cap.release()
                cv2.destroyAllWindows()
            else:
                st.error("Failed to open video source.")
        else:
            st.warning("Please upload a video file to start.")

st.caption("Made By Zailee Brooks & Isabell Ceja 11/19/2024 ❤")