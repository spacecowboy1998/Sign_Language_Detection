import cv2
import copy
import mediapipe as mp
from Utils import calc_landmark_list, pre_process_landmark, draw_landmarks, logging_landmarks, logging_name

# Define gestures
class_names = ['ა','კ','ი','გამარჯობა', ' ']
num_classes = len(class_names)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=1)

def main():

    cap = cv2.VideoCapture(0)

    for class_id in range(num_classes):
        print(f"Capturing photos for Gesture {class_names[class_id]}")

        image_count = 0

        while True:
            ret, image = cap.read()
            if not ret:
                return

            # Flip and prepare the image for display
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Calculate and process landmarks
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    normalized_landmarks = pre_process_landmark(landmark_list)

                    # Draw landmarks for visualization
                    debug_image = draw_landmarks(debug_image, landmark_list)

            # Display instructions
            cv2.putText(debug_image, f'Class : {class_id} Counts : {image_count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, 'Press "C" to capture or "N" to move to next class.', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 105, 50), 2, cv2.LINE_AA)
            cv2.imshow("Hand Gesture Capture", debug_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Log the normalized landmarks with the class ID
                logging_landmarks(class_id, normalized_landmarks)
                image_count += 1
                print(f"Image instance for Gesture {class_names[class_id]} is saved.")
            elif key == ord('n'):
                print(f"Finished capturing photos Gesture {class_names[class_id]}.")
                break
            elif key == 27:  # ESC to exit
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"Moving To Gesture {class_id+1}.")

    # Save class names CSV
    logging_name(class_names)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()