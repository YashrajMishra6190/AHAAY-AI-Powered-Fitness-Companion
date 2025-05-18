import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from utils import calculate_angle, detection_body_part
import tempfile

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Streamlit UI
st.title("AI AHAAY Fitness Exercise Tracker")

# User input for exercise type
exercise_type = st.selectbox("Select Exercise Type", [
    "push-up", "pull-up", "squat", "walk", "sit-up"
])

# User input for video source
video_source = st.radio("Choose Video Source", ("Webcam", "Upload Video"))

# Pause and resume functionality
pause = st.checkbox("Pause Analysis")

# Initialize variables
counter = 0
status = True

# Function to process frames
def process_frame(frame, exercise_type, counter, status):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            return frame, counter, status

        # Calculate body part angles
        body_angles = BodyPartAngle(landmarks)

        # Calculate exercise
        counter, status = TypeOfExercise(landmarks).calculate_exercise(
            exercise_type, counter, status
        )

        # Render detections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Overlay counter and status on the frame
        cv2.putText(image, f"Counter: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Status: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image, counter, status

# Main loop for video processing
def main_loop():
    global counter, status  # Ensure counter and status are accessible

    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                return
        else:
            st.warning("Please upload a video file.")
            return

    stframe = st.image([])

    while cap.isOpened() and not pause:
        ret, frame = cap.read()
        if not ret:
            break

        frame, counter, status = process_frame(frame, exercise_type, counter, status)
        stframe.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main_loop()
