import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import mediapipe as mp
from Model.hand_gesture.Hand_Gesture_Classifier import HandGestureClassifier
from Utils import *

# Global Configuration
PROBABILITY_THRESHOLD = 0.5
LAST_PREDICTION_TIME = time.time()

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Initialize Classifier
keypoint_classifier = HandGestureClassifier()

# Read Labels
with open('Model/hand_gesture/hand_gesture_label.csv', encoding='utf-8') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# Tkinter Initialization
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("700x700")

# Tkinter Video Frame
video_frame = tk.Label(root)
video_frame.pack(pady=20)

# Tkinter Text Box
text_box = tk.Text(root, height=5)
text_box.pack(padx=20, pady=10)

# Tkinter Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

def clear_text():
    text_box.delete('1.0', tk.END)

def save_to_file():
    file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text file", "*.txt")])
    if file:
        with open(file, 'w') as f:
            f.write(text_box.get('1.0', tk.END))

tk.Button(button_frame, text="Clear All", command=clear_text, bg="yellow", height=2, width=15).grid(row=0, column=0, padx=10)
tk.Button(button_frame, text="Save to File", command=save_to_file, bg="green", height=2, width=15).grid(row=0, column=1, padx=10)
tk.Button(button_frame, text="Quit", command=root.quit, bg="red", height=2, width=15).grid(row=0, column=2, padx=10)

# Prediction and Frame Processing
cap = cv2.VideoCapture(0)

def process_frame():
    global LAST_PREDICTION_TIME
    ret, image = cap.read()
    if not ret:
        return

    image = cv2.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Calculate and preprocess landmarks
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Calculate bounding rectangle for image
            brect = calc_bounding_rect(debug_image, hand_landmarks)

            current_time = time.time()
            # Make prediction every in every 2.5 second
            if current_time - LAST_PREDICTION_TIME > 2.5:

                LAST_PREDICTION_TIME = current_time

                # Hand gesture classification
                hand_sign_id, predicted_probability = keypoint_classifier(pre_processed_landmark_list)
                predicted_character = keypoint_classifier_labels[hand_sign_id]

                overlay = debug_image.copy()
                cv2.rectangle(overlay, (brect[0] - 20, brect[1]- 20), (brect[2] + 20, brect[3] +20 ), (51, 204, 255), -1)
                cv2.addWeighted(overlay, 0.4, debug_image, 0.6, 0, debug_image)

                # If predicted Probability meets probability threshold display character
                if predicted_probability >= PROBABILITY_THRESHOLD:
                    text_box.insert(tk.END, predicted_character)
                    text_box.see(tk.END)
                    print(
                        f'''
                        --------------------------------
                        Predicted Character: {predicted_character}
                        Predicted Probability: {predicted_probability}
                        --------------------------------
                        '''
                    )


            # Draw bounding box and landmarks
            debug_image = draw_landmarks(debug_image, landmark_list, brect)

    # Convert to Tkinter-compatible image and update video frame
    img = Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)

    # Repeat the process
    video_frame.after(10, process_frame)

# Start processing
process_frame()
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()


