# Load the trained model
import tensorflow as tf


model = tf.keras.models.load_model('sign_language_model.h5')

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Initialize hand detection with Mediapipe
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a selfie view
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks = extract_landmarks(frame, hands)
        
        if len(landmarks) > 0:  # If hand landmarks are detected
            landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Flatten landmarks to match model input
            prediction = model.predict(landmarks)  # Get the predicted sign
            predicted_label = np.argmax(prediction)  # Get the class with the highest probability
            
            # Map the predicted label to a sign (e.g., A-Z)
            sign = chr(predicted_label + 65)  # Mapping 0 -> A, 1 -> B, ..., 25 -> Z
            
            # Display the predicted sign on the frame
            cv2.putText(
                frame, f"Predicted Sign: {sign}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

        # Display the video feed with the predicted sign
        cv2.imshow("Sign Language Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
