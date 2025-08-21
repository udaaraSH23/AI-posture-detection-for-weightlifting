import numpy as np
import json
import os

# Barbell width in pixels and meters
PIXEL_WIDTH = 1296
REAL_WORLD_WIDTH = 2.2  # meters
PIXEL_TO_METER = REAL_WORLD_WIDTH / PIXEL_WIDTH  # meters per pixel

# Landmarks to extract
SIDE_VIEW_LANDMARKS = {
    "shoulder": 12,
    "elbow": 14,
    "wrist": 16,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
    "back": 11,
    "foot": 32
}

def extract_velocity_side_view(input_path, output_path, fps=30):
    """
    Extract joint velocities from keypoints and convert to meters/second (m/s).
    """
    with open(input_path, "r") as f:
        keypoints_data = json.load(f)

    sorted_frames = sorted(keypoints_data.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
    velocities_per_frame = {}

    def get_landmark(frame_kpts, landmark_idx):
        if landmark_idx < len(frame_kpts):
            return frame_kpts[landmark_idx]
        return None

    for i in range(len(sorted_frames) - 1):
        frame_curr = sorted_frames[i]
        frame_next = sorted_frames[i + 1]

        curr_kpts = keypoints_data[frame_curr]
        next_kpts = keypoints_data[frame_next]

        frame_velocities = {}

        for joint, idx in SIDE_VIEW_LANDMARKS.items():
            curr_point = get_landmark(curr_kpts, idx)
            next_point = get_landmark(next_kpts, idx)

            if curr_point is not None and next_point is not None:
                pixel_dist = np.linalg.norm(np.array(next_point) - np.array(curr_point))
                velocity_m_per_s = pixel_dist * PIXEL_TO_METER * fps  # meters/second
                frame_velocities[f"{joint}_velocity"] = velocity_m_per_s
            else:
                frame_velocities[f"{joint}_velocity"] = None

        velocities_per_frame[frame_curr] = frame_velocities

    # Last frame: set all velocities to 0.0
    last_frame = sorted_frames[-1]
    velocities_per_frame[last_frame] = {f"{joint}_velocity": 0.0 for joint in SIDE_VIEW_LANDMARKS.keys()}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f_out:
        json.dump(velocities_per_frame, f_out, indent=4)

    print(f"✅ Saved side view velocities (in m/s) to {output_path}")
