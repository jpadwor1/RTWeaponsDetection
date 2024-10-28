import cv2
from ultralytics import YOLO
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the YOLO model
model = YOLO("best.pt")

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
    model = YOLO("best.pt")
else:
    model = YOLO("yolo11n.pt")

col2.title("Select a Video Source")
selectedDevice = col2.radio(
    "What input video source would you like to use?",
    ["Webcam", "Upload a video", "Sample Video"],
    help="Select the video source for object detection."
)

if selectedDevice == "Upload a video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

if selectedDevice == "Webcam":
    st.info("Please grant the required permissions to access the webcam.")
    webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

btnCol1, btnCol2 = st.columns([1, 1])

quit = btnCol1.button("Quit", type="secondary", use_container_width=True)

if selectedDevice != "Webcam":
    start = btnCol2.button("Start", type="primary", use_container_width=True)

    if start:
        video_path = None
        if selectedDevice == "Upload a video" and uploaded_file is not None:
            video_path = f"./videos/{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif selectedDevice == "Sample Video":
            video_path = "./gun.mp4"
        
        # Process video if available
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                display_height = int(display_width * frame_height / frame_width)

                # Use an empty slot to overwrite frames continuously
                frame_display = st.empty()

                while cap.isOpened():
                    success, frame = cap.read()
                    if success:
                        # Run YOLO inference on the frame
                        results = model.predict(frame, stream=True, device="cpu")
                        for result in results:
                            annotated_frame = result.plot()

                            # Display the frame in the same slot for continuous updates
                            frame_display.image(annotated_frame, channels="BGR", width=display_width)

                        # Check for quit button to exit
                        if quit:
                            break
                    else:
                        break

                # Release resources
                cap.release()
                cv2.destroyAllWindows()
            else:
                st.error("Failed to open video source.")
        else:
            st.warning("Please upload a video file to start.")
