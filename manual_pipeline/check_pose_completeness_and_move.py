import cv2
import mediapipe as mp
import os
import shutil

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def check_pose_completeness_and_move(input_folder, bad_folder, min_visible=25, visibility_threshold=0.5):
    """
    Checks pose completeness and moves frames with low keypoint visibility to a separate folder.
    """
    if not os.path.exists(bad_folder):
        os.makedirs(bad_folder)

    print(f"🔍 Checking pose completeness in: {input_folder}")
    incomplete_frames = []

    for img_name in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        visible_count = 0
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                if landmark.visibility >= visibility_threshold:
                    visible_count += 1

        if visible_count < min_visible:
            incomplete_frames.append((img_name, visible_count))
            print(f"⚠️ Incomplete pose in {img_name}: {visible_count}/33 visible")

            # Move the bad frame
            shutil.move(img_path, os.path.join(bad_folder, img_name))

    print(f"\n📉 Moved {len(incomplete_frames)} incomplete frames to '{bad_folder}'")
    return incomplete_frames
