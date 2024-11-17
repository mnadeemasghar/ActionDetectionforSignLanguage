import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Initialize hand detection
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a selfie view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)

        # Draw landmarks and classify gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Get landmark positions
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Example Gesture Recognition
                if (
                    thumb_tip.y < wrist.y
                    and index_tip.y < wrist.y
                    and middle_tip.y < wrist.y
                    and ring_tip.y < wrist.y
                    and pinky_tip.y < wrist.y
                ):
                    gesture = "Open Hand"
                elif (
                    index_tip.y < wrist.y
                    and middle_tip.y > wrist.y
                    and ring_tip.y > wrist.y
                    and pinky_tip.y > wrist.y
                ):
                    gesture = "Pointing"
                elif (
                    index_tip.y < wrist.y
                    and middle_tip.y < wrist.y
                    and ring_tip.y > wrist.y
                    and pinky_tip.y > wrist.y
                ):
                    gesture = "Peace Sign"
                else:
                    gesture = "Unknown Gesture"

                # Display the gesture on the screen
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Display the video feed with gestures
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
