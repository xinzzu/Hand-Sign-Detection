import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Initialize Mediapipe Hand model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Placeholder for storing training data
    X = []
    y = []

    # Collect training data (This would normally be replaced with pre-recorded data loading)
    def collect_training_data(label):
        cap = cv2.VideoCapture(0)
        print(f"Collecting data for '{label}'. Press 'q' to stop.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = extract_hand_landmarks(results)
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(label)
            
            cv2.imshow('Collecting Data', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    # Extract hand landmarks
    def extract_hand_landmarks(results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                return np.array(landmarks).flatten()
        return None

    # Train the model
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print("Model accuracy:", accuracy_score(y_test, y_pred))
        
        # Save the model
        with open('sign_language_model.pkl', 'wb') as f:
            pickle.dump(clf, f)

        return clf

    # Real-time prediction
    def real_time_prediction(clf):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = extract_hand_landmarks(results)
                if landmarks is not None:
                    prediction = clf.predict([landmarks])
                    sign = prediction[0]
                    
                    cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    # Collect training data for different signs
    signs = ["hello", "i love you", "thank you", "fuck you"]
    for sign in signs:
        collect_training_data(sign)

    # Train the model
    model = train_model(X, y)

    # Real-time prediction
    real_time_prediction(model)

if __name__ == "__main__":
    main()
