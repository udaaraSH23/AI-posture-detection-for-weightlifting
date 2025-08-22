# AI-Driven Weightlifting Posture Analysis Pipeline

This repository contains a complete pipeline for analyzing Olympic weightlifting videos (Snatch and Clean & Jerk) to extract joint angles, velocities, phases, generate feedback, and train a GRU regression model for posture correction.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Step 1 - Frame Extraction](#step-1---frame-extraction)  
3. [Step 2 - Keypoint Detection](#step-2---keypoint-detection)  
4. [Step 3 - Feature Extraction](#step-3---feature-extraction)  
5. [Step 4 - Dataset Creation](#step-4---dataset-creation)  
6. [Step 5 - Model Architecture and Training](#step-5---model-architecture-and-training)  
7. [Step 6 - Using the Model](#step-6---using-the-model)  
8. [Folder Structure](#folder-structure)  
9. [References](#references)  

---

## Prerequisites

Install required Python packages:

```bash
pip install opencv-python-headless mediapipe pandas scikit-learn tensorflow matplotlib
```

---

## Step 1 - Frame Extraction

```python
import cv2
import os
from automatic_pipeline.extract_frames import extract_frames
from automatic_pipeline.crop_black_margins import crop_black_margins

# Example: Extract frames from a video
extract_frames("video.mp4", "Output/frames/video_name")
crop_black_margins("Output/frames/video_name", "Output/frames/video_name_cropped")
```

---

## Step 2 - Keypoint Detection

```python
from automatic_pipeline.annotate_frames import detect_keypoints
from automatic_pipeline.extract_2d_keypoint import extract_2d_keypoints

# Detect keypoints and save to JSON
detect_keypoints("Output/frames/video_name_cropped", "Output/annotations/video_name")
extract_2d_keypoints("Output/frames/video_name_cropped", "Output/keypoints/video_name/keypoints.json")
```

- Normalize keypoints according to hip distance.

---

## Step 3 - Feature Extraction

```python
from automatic_pipeline.extract_joint_angles import extract_joint_angles
from automatic_pipeline.extract_velocities import extract_velocity_side_view

# Extract angles
extract_joint_angles("Output/keypoints/video_name/keypoints.json", "Output/angles/video_name/joint_angles.json")

# Extract velocities (side view)
extract_velocity_side_view("Output/keypoints/video_name/keypoints.json", "Output/velocities/video_name/velocities.json")
```

---

## Step 4 - Dataset Creation

### Find Complete Data Sets (Quintuplets)

```python
from automatic_pipeline.dataset_utils import find_joint_angle_velocity_phase_quintuplets

quintuplets = find_joint_angle_velocity_phase_quintuplets(
    angle_root="Output/angles",
    velocity_root="Output/velocities",
    phase_root="Output/phase_data"
)
```

### Generate Sequences

```python
from automatic_pipeline.sequence_utils import process_joint_angle_velocity_phase_triplet

for front, side, top, vel, phase in quintuplets:
    sequences = process_joint_angle_velocity_phase_triplet(front, side, top, vel, phase, seq_len=10)
```

### Generate Feedback for Sequences

```python
from automatic_pipeline.feedback_generation import generate_feedback_for_sequence

all_feedback = []
for seq in sequences:
    feedback = generate_feedback_for_sequence(seq)
    all_feedback.append(feedback)

# Save feedback to JSON
import json
with open("feedback_output.json", "w") as f:
    json.dump(all_feedback, f, indent=4)
```

---

## Step 5 - Model Architecture and Training

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load dataset
data = np.load("feedback_dataset.npz")
X, y = data["X"], data["y"]

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build GRU regression model
def build_gru_regression_model(input_shape, output_dim):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

model = build_gru_regression_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])

# Train model
history = model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_val, y_val))

# Save model
model.save("gru_posture_model_snatch_regression.h5")
```

### Plot Training Metrics

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("MSE Loss Over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["mean_absolute_error"], label="Train MAE")
plt.plot(history.history["val_mean_absolute_error"], label="Val MAE")
plt.title("MAE Over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Step 6 - Using the Model

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model("gru_posture_model_snatch_regression.h5")

# Predict
data = np.load("regression_dataset_snatch.npz")
X_test, y_test = data["X"], data["y"]
y_pred = model.predict(X_test)

# Compare first prediction
print("Predicted:", y_pred[0])
print("Actual   :", y_test[0])
```

---

## Folder Structure

```
Source/                # Input videos
Output/
├─ frames/             # Extracted frames
├─ annotations/        # Annotated frames
├─ keypoints/          # 2D keypoints JSON
├─ angles/             # Joint angles JSON
├─ velocities/         # Velocity JSON
├─ phase_data/         # Phase label JSON
feedback_output.json   # Generated feedback
regression_dataset.npz # Training dataset
gru_posture_model_snatch_regression.h5
```

---

## References

- [MediaPipe](https://mediapipe.dev/) for keypoint extraction  
- TensorFlow GRU documentation for sequence regression models  
- Olympic weightlifting biomechanics literature for phase definitions and thresholds  

