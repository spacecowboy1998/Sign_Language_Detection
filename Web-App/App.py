import cv2
import time
import copy
import csv
from flask import Flask, render_template, Response
from mediapipe import solutions as mp
from Model import HandGestureClassifier
from Utilities.Utils import calc_landmark_list, pre_process_landmark, calc_bounding_rect, draw_landmarks

# Initialize Flask app
app = Flask(__name__)

# Mediapipe Hands Initialization
mp_hands = mp.hands.Hands(static_image_mode=False, max_num_hands=1)

# Hand Gesture Classifier Initialization
keypoint_classifier = HandGestureClassifier()

# Load gesture labels
with open('../Model/hand_gesture/hand_gesture_label.csv', encoding='utf-8') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# Global variables for predictions
PROBABILITY_THRESHOLD = 0.5
LAST_PREDICTION_TIME = time.time()
last_prediction = ""

def generate_frames():
    global LAST_PREDICTION_TIME, last_prediction
    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate landmarks and bounding box
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                # Predict gesture every 2.5 seconds
                current_time = time.time()
                if current_time - LAST_PREDICTION_TIME > 2.5:
                    LAST_PREDICTION_TIME = current_time
                    hand_sign_id, predicted_probability = keypoint_classifier(pre_processed_landmark_list)
                    if predicted_probability >= PROBABILITY_THRESHOLD:
                        last_prediction = keypoint_classifier_labels[hand_sign_id]  # Store latest prediction

                        # Overlay prediction
                        overlay = debug_image.copy()
                        cv2.rectangle(overlay,
                                      (brect[0] - 20, brect[1] - 20),
                                      (brect[2] + 20, brect[3] + 20),
                                      (51, 204, 255), -1)
                        cv2.addWeighted(overlay, 0.4, debug_image, 0.6, 0, debug_image)

                        print(f"Predicted Character: {last_prediction} | Probability: {predicted_probability}")

                # Draw landmarks
                debug_image = draw_landmarks(debug_image, landmark_list, brect)
        else:
                last_prediction=''

        # Encode the frame to JPEG
        _, buffer = cv2.imencode('.jpg', debug_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global last_prediction
    return {"prediction": last_prediction}


if __name__ == '__main__':
    app.run(debug=True)
