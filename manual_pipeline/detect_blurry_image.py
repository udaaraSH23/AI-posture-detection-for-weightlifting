import os
import cv2
import shutil

def detect_and_move_blurry_images(folder_path, default_threshold=120.0):
    """
    Analyze blurriness of images using Laplacian variance.
    Prompts for threshold input and moves blurry images to a separate folder.
    """
    print(f"\n🔍 Analyzing blurriness in: {folder_path}")
    blur_data = []

    # Step 1: Collect and show blurriness variance
    for img_name in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_data.append((img_name, variance))

    # Display all variance scores
    print("\n📊 Blurriness Index (Higher is sharper):")
    for name, var in blur_data:
        print(f"{name}: Variance = {var:.2f}")

    # Step 2: Ask user for threshold
    try:
        threshold_input = input(f"\nEnter blurriness threshold [Default = {default_threshold}]: ")
        threshold = float(threshold_input) if threshold_input.strip() else default_threshold
    except ValueError:
        print("⚠️ Invalid input. Using default threshold.")
        threshold = default_threshold

    # Step 3: Move blurry images
    blurry_folder = os.path.join(os.path.dirname(folder_path), f"blurry_{os.path.basename(folder_path)}")
    os.makedirs(blurry_folder, exist_ok=True)

    moved = 0
    for name, var in blur_data:
        if var < threshold:
            src_path = os.path.join(folder_path, name)
            dst_path = os.path.join(blurry_folder, name)
            shutil.move(src_path, dst_path)
            print(f"📁 Moved blurry image: {name} (Variance = {var:.2f})")
            moved += 1

    print(f"\n✅ Done. {moved} blurry images moved to '{blurry_folder}'.")

# Example usage
# detect_and_move_blurry_images("Output/extracted_frames/top")
