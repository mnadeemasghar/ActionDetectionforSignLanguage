import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define function to process each frame and extract landmarks
def extract_landmarks(frame, hands_module):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_module.process(frame_rgb)
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
    return landmarks

# Build your neural network model
def build_model():
    model = Sequential([
        Dense(128, activation='relu', input_dim=63),  # 21 landmarks, each with x, y, z (63 values)
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(26, activation='softmax')  # Assuming 26 classes (A-Z in ASL)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare your training data (features and labels)
def prepare_data(landmarks_list, labels):
    X = np.array(landmarks_list)
    y = np.array(labels)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model (assuming you have preprocessed data)
landmarks_list = []  # List of all the landmark features
labels = []  # Corresponding labels for the landmarks (e.g., "A", "B", "C", etc.)

# For each image in your dataset, extract landmarks and labels
# You should do this for all your images and store the landmarks in landmarks_list
# and labels in the labels list.

X_train, X_test, y_train, y_test = prepare_data(landmarks_list, labels)

# Train the model
model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('sign_language_model.h5')
