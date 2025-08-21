import cv2
import os

def check_orientation(folder_path):
    """
    Automatically rotates portrait images in the given folder to landscape.
    Overwrites the original image if rotated.
    """
    for img_name in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_name}")
            continue

        height, width, _ = img.shape

        # Rotate only if it's in portrait orientation
        if height > width:
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(img_path, img_rotated)
            print(f"🔄 Rotated {img_name} to landscape ({height}x{width}) → ({img_rotated.shape[1]}x{img_rotated.shape[0]})")
        else:
            print(f"✅ {img_name} already landscape ({width}x{height})")
