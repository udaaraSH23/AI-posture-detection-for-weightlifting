import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # static_image_mode=True for images
mp_drawing = mp.solutions.drawing_utils

def detect_keypoints(input_folder, output_folder):
    """Detects keypoints in images and saves annotated frames."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Could not read image {img_path}, skipping.")
            continue  # Skip if image is corrupted or not read properly

        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            # Draw keypoints on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save the output image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, image)

    print(f"✅ Keypoint detection completed. Output saved in {output_folder}")

# Example usage
# detect_keypoints("path/to/input_frames", "path/to/output_annotated_frames")
