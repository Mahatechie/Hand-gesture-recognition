import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "D:/Downloads/hand_gesture_recognition/leapGestRecog"
IMG_SIZE = 64

def load_images():
    X = []
    y = []
    class_names = []
    class_id = 0

    print(f"🔍 Looking inside: {DATA_DIR}")
    for user_folder in os.listdir(DATA_DIR):
        user_path = os.path.join(DATA_DIR, user_folder)
        if not os.path.isdir(user_path):
            continue

        print(f"📁 Reading user folder: {user_path}")
        for gesture_folder in os.listdir(user_path):
            gesture_path = os.path.join(user_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            print(f"    ➤ Reading gesture folder: {gesture_path}")
            class_names.append(gesture_folder)

            for filename in os.listdir(gesture_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(gesture_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(class_id)

            class_id += 1

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array(y)
    print(f"✅ Loaded {len(X)} images across {len(class_names)} gesture classes.")
    return X, y, class_names

def prepare_data():
    X, y, class_names = load_images()
    if len(X) == 0:
        raise ValueError("❌ No images loaded. Check DATA_DIR or image format.")
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print("✅ Data preparation complete. Ready for training.")



