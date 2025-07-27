import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from preprocess import prepare_data

# Load preprocessed data
X_train, X_test, y_train, y_test = prepare_data()

# Convert labels to one-hot vectors
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("🚀 Training started...")
history = model.fit(X_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"✅ Test accuracy: {acc:.4f}")

# Save model
model.save("gesture_cnn_model.h5")
print("✅ Model saved as gesture_cnn_model.h5")
