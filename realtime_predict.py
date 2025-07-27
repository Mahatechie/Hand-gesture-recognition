import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("gesture_cnn_model.h5")

# Define gesture classes (based on your folder names 00 to 09, for example)
gesture_classes = {
    0: "Palm", 1: "L", 2: "Fist", 3: "Fist_moved", 4: "Thumb", 
    5: "Index", 6: "OK", 7: "Palm_moved", 8: "C", 9: "Down"
    # Add up to 99 classes if needed
}

# Initialize webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Define region of interest (ROI) for hand
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Adjust based on training input size
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 64, 64, 1))  # Shape should match model input

    # Predict
    predictions = model.predict(reshaped)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]

    # Show prediction
    gesture_name = gesture_classes.get(class_id, "Unknown")
    label = f"{gesture_name} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
