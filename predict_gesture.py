import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("gesture_cnn_model.h5")

# Define the image path you want to test
image_path = r"D:\Downloads\hand_gesture_recognition\leapGestRecog\00\01_palm\frame_00_01_0010.png"

# Define the label map (you can expand this up to class 99 if needed)
label_map = {
    0: "Palm",
    1: "Fist",
    2: "Thumbs Up",
    3: "OK Sign",
    4: "Peace",
    5: "Stop",
    6: "L Shape",
    7: "Point",
    8: "Rock",
    9: "Call Me"
    # Add more if you have more classes...
}

# Preprocess the image (resize, grayscale, normalize)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

img = cv2.resize(img, (64, 64))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=-1)  # Add channel dimension (64,64,1)
img = np.expand_dims(img, axis=0)   # Add batch dimension (1,64,64,1)

# Predict
pred = model.predict(img)
predicted_class = np.argmax(pred)
confidence = np.max(pred) * 100

# Get label from map
gesture_label = label_map.get(predicted_class, "Unknown")

# Output
print(f"🧠 Predicted Class ID: {predicted_class}")
print(f"🔍 Confidence: {confidence:.1f} %")
print(f"🖐 Gesture: {gesture_label}")
