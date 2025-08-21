import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=30):
    """Extract frames from a video at a given frame rate."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS

    # Debug: Print FPS
    print(f"Video: {video_path}, FPS: {fps}")

    if fps == 0:
        print("Error: Could not read FPS. Check the video file or codec.")
        return  # Exit function to prevent division by zero

    frame_interval = max(1, int(fps / frame_rate))  # Avoid division by zero

    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when the video ends

        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1  # Increment frame number

        count += 1

    cap.release()
    print(f" Extracted {frame_id} frames from {video_path}")
    

