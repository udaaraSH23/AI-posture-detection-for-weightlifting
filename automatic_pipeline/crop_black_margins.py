import os
import cv2
import numpy as np

def crop_black_margins(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in sorted(os.listdir(input_folder)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to remove black background (tweak threshold if needed)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours of non-black areas
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = img[y:y+h, x:x+w]
        else:
            cropped = img  # If all black, leave as is

        # Save cropped image
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, cropped)

    print(f"✅ Cropped images saved to: {output_folder}")
