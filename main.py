import cv2
from ultralytics import YOLO
import torch
import streamlit as st
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Check if CUDA is available
print(torch.cuda.is_available())

# Load the YOLO model
model = YOLO("best.pt")


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference on the frame
        results = model.predict(img, stream=True, device="cpu")

        # Iterate over the generator to get individual results
        for result in results:
            # Visualize the results on the frame
            img = result.plot()

        return img


# Define the display window size
display_width = 800

# Streamlit UI elements
st.title("Object Detection with YOLO")
selectedDevice = st.radio(
    "What input video source would you like to use?",
    ["Webcam", "Upload a video", "Sample Video"],
    help="Select the video source for object detection."
)

if selectedDevice == "Upload a video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

if selectedDevice == "Webcam":
    st.info("Please grant the required permissions to access the webcam.")
    st.info("Once permissions are granted hit start")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

quit = st.button("Quit")
if selectedDevice != "Webcam":
    start = st.button("Start")

    if start:
        if selectedDevice == "Upload a video" and uploaded_file is not None:
            video_path = f"./videos/{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            cap = cv2.VideoCapture(video_path)
        elif selectedDevice == "Sample Video":
            video_path = "./videos/gun.mp4"
            cap = cv2.VideoCapture(video_path)
        elif selectedDevice == "Webcam":
            webrtc_streamer(
                key="source", video_transformer_factory=VideoTransformer)
        else:
            cap = cv2.VideoCapture(0)  # Use webcam

        if selectedDevice != "Webcam":
            if cap.isOpened():
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                display_height = int(
                    display_width * frame_height / frame_width)

                # Create a temporary file to save the output video
                temp_video_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.mp4')
                out = cv2.VideoWriter(temp_video_file.name, cv2.VideoWriter_fourcc(
                    *'mp4v'), 20, (frame_width, frame_height))

                while cap.isOpened():
                    success, frame = cap.read()
                    if success:
                        # Run YOLO inference on the frame
                        results = model.predict(
                            frame, stream=True, device="cpu")

                        # Iterate over the generator to get individual results
                        for result in results:
                            # Visualize the results on the frame
                            annotated_frame = result.plot()

                            # Write the annotated frame to the video file
                            out.write(annotated_frame)

                            # Break the loop if 'q' is pressed or quit button is clicked
                            if cv2.waitKey(1) & 0xFF == ord("q") or quit:
                                break
                    else:
                        break

                # Release the video capture and writer objects
                cap.release()
                out.release()
                cv2.destroyAllWindows()

                # Display the saved video file using Streamlit
                st.video(temp_video_file.name)

                # Clean up the temporary video file
                os.remove(temp_video_file.name)
            else:
                st.error("Failed to open video source.")
