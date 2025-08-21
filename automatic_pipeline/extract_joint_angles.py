import numpy as np
import json
import os

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

POSE_LANDMARKS = {
    "front": {
        "right_shoulder": 12, "right_elbow": 14, "right_wrist": 16,
        "left_shoulder": 11, "left_elbow": 13, "left_wrist": 15
    },
    "side": {
        "shoulder": 12, "elbow": 14, "wrist": 16,
        "hip": 24, "knee": 26, "ankle": 28,
        "back": 11, "foot": 32
    },
    "top": {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_wrist": 15, "right_wrist": 16
    }
}

def get_view_from_path(path):
    folder_name = os.path.basename(os.path.dirname(path)).lower()
    if "front" in folder_name:
        return "front"
    elif "side" in folder_name:
        return "side"
    elif "top" in folder_name:
        return "top"
    return "unknown"

def extract_joint_angles(input_path, output_path):
    view = get_view_from_path(input_path)
    if view == "unknown":
        print(f"⚠️ Skipping: Unknown view in path: {input_path}")
        return

    with open(input_path, "r") as file:
        keypoints_data = json.load(file)

    landmarks = POSE_LANDMARKS[view]
    angles_per_frame = {}

    for frame_name, keypoints in keypoints_data.items():
        angles = {}

        def get_landmark(name):
            index = landmarks.get(name)
            if index is None or index >= len(keypoints):
                return None
            return keypoints[index]

        if view == "front":
            r_s, r_e, r_w = get_landmark("right_shoulder"), get_landmark("right_elbow"), get_landmark("right_wrist")
            l_s, l_e, l_w = get_landmark("left_shoulder"), get_landmark("left_elbow"), get_landmark("left_wrist")
            if None not in [r_s, r_e, r_w]:
                angles["right_elbow_angle"] = calculate_angle(r_s, r_e, r_w)
            if None not in [l_s, l_e, l_w]:
                angles["left_elbow_angle"] = calculate_angle(l_s, l_e, l_w)

        elif view == "side":
            s, e, w = get_landmark("shoulder"), get_landmark("elbow"), get_landmark("wrist")
            h, k, a = get_landmark("hip"), get_landmark("knee"), get_landmark("ankle")
            b, f = get_landmark("back"), get_landmark("foot")
            if None not in [h, k, a]:
                angles["knee_angle"] = calculate_angle(h, k, a)
            if None not in [s, h, k]:
                angles["hip_angle"] = calculate_angle(s, h, k)
            if None not in [k, a, f]:
                angles["ankle_angle"] = calculate_angle(k, a, f)
            if None not in [s, b, f]:
                angles["back_angle"] = calculate_angle(s, b, f)
            if None not in [s, e, w]:
                angles["elbow_angle"] = calculate_angle(s, e, w)

        elif view == "top":
            l_s, r_s = get_landmark("left_shoulder"), get_landmark("right_shoulder")
            l_h, r_h = get_landmark("left_hip"), get_landmark("right_hip")
            l_w, r_w = get_landmark("left_wrist"), get_landmark("right_wrist")
            if None not in [l_s, r_s, r_h]:
                angles["shoulder_symmetry"] = calculate_angle(l_s, r_s, r_h)
            if None not in [l_h, r_h, r_s]:
                angles["hip_symmetry"] = calculate_angle(l_h, r_h, r_s)
            if None not in [l_w, r_w, r_s]:
                angles["wrist_alignment"] = calculate_angle(l_w, r_w, r_s)

        if angles:
            angles_per_frame[frame_name] = angles

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(angles_per_frame, f, indent=4)
    print(f"✅ Saved joint angles to {output_path}")
