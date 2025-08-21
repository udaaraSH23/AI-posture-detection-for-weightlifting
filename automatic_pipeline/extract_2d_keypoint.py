import cv2
import mediapipe as mp
import os
import json

# =========================
# MediaPipe Pose Setup
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# =========================
# Landmark Index to Use
# =========================
# Keep 11–32 (body, hands, feet), skip 0–10 (face)
USEFUL_LANDMARKS = list(range(11, 33))  # Keep shoulder to foot landmarks only

# =========================
# Normalize Function
# =========================
def normalize_keypoints(landmarks):
    """
    Normalize keypoints by centering at mid-hip and scaling by shoulder width.
    Only useful keypoints are returned.
    """
    try:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    except IndexError:
        return []  # Skip if any required landmarks are missing

    # Center = Midpoint between hips
    center_x = (left_hip.x + right_hip.x) / 2
    center_y = (left_hip.y + right_hip.y) / 2

    # Scale = Shoulder width
    scale = ((left_shoulder.x - right_shoulder.x) ** 2 +
             (left_shoulder.y - right_shoulder.y) ** 2) ** 0.5
    if scale < 1e-6:
        scale = 1.0  # Prevent division by zero

    # Normalize and filter only useful keypoints
    normalized = []
    for idx in USEFUL_LANDMARKS:
        lm = landmarks[idx]
        x_norm = (lm.x - center_x) / scale
        y_norm = (lm.y - center_y) / scale
        normalized.append((x_norm, y_norm))

    return normalized

# =========================
# Main Extraction Function
# =========================
def extract_2d_keypoints(input_folder, output_json):
    """
    Extracts and normalizes 2D keypoints from each image in a folder,
    using only useful body-related landmarks. Saves result to JSON.
    """
    keypoints_data = {}

    for img_name in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue  # Skip if unreadable or corrupted

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            normalized_keypoints = normalize_keypoints(results.pose_landmarks.landmark)
            keypoints_data[img_name] = normalized_keypoints
        else:
            keypoints_data[img_name] = []  # No keypoints detected

    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(keypoints_data, f, indent=4)

    print(f"✅ 2D normalized keypoints extracted and saved to {output_json}")
