import os
import mediapipe as mp
from Utilities.Utils import *

# Define constants
DATA_DIR = '../data'
DATASET_SIZE = 500
CLASS_NAMES = [
    "ა", "ბ", "გ", "დ", "ე", "ვ", "ზ", "თ", "ი", "კ",
    "ლ", "მ", "ნ", "ო", "პ", "ჟ", "რ", "ს", "ტ", "უ",
    "ფ", "ქ", "ღ", "ყ", "შ", "ჩ", "ც", "ძ", "წ", "ჭ",
    "ხ", "ჯ", "ჰ", "გამარჯობა", " "
]
NUMBER_OF_CLASSES = len(CLASS_NAMES)
print(NUMBER_OF_CLASSES)

def capture_images(dataset_size, number_of_classes):

    cap = cv2.VideoCapture(0)

    for class_id in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(class_id))
        os.makedirs(class_dir, exist_ok=True)

        print(f'Preparing to collect data for class {class_id}. Press "S" to start capturing images.')

        # Wait for user input to start capturing
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Display instructions
            cv2.putText(frame, f'Class {class_id}: Get Ready!', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 255), 2, cv2.LINE_AA)  # Purple
            cv2.putText(frame, 'Press "C" to start capturing images.', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)  # White
            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(25)
            if key == 27:  # ESC key
                print('Exiting data collection...')
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                print(f'Starting image capture for sign {CLASS_NAMES[class_id]}.')
                break

        # Capture dataset_size images for the current gesture
        for i in range(dataset_size):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                return
            img_path = os.path.join(class_dir, f'{i}.jpg')
            cv2.imwrite(img_path, frame)

            # Display the progress and class
            cv2.putText(frame, f'Capturing Class {class_id}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2, cv2.LINE_AA)  # Lime Green
            cv2.putText(frame, f'Image {i+1}/{dataset_size}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2, cv2.LINE_AA)  # Gold
            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(25)
            if key == 27:  # ESC key
                break

            print(f'Captured image {i+1}/{dataset_size} for gesture {CLASS_NAMES[class_id]}.')

    cap.release()
    cv2.destroyAllWindows()
    print('Data collection complete!')

def proces_image_data(data_dir, class_names):

    # Initialize hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    # Iterate through directories where photos are saved
    for class_id, dir_ in enumerate(os.listdir(data_dir)):
        for img_path in os.listdir(os.path.join(data_dir, dir_)):
            img_full_path = os.path.join(data_dir, dir_, img_path)

            # Read and process the image
            img = cv2.imread(img_full_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # Calculate and preprocess landmarks
                    landmark_list = calc_landmark_list(img, hand_landmarks)
                    normalized_landmarks = pre_process_landmark(landmark_list)

                    # Log landmarks with class ID
                    logging_landmarks(class_id, normalized_landmarks)

    # Log classes (gestures)
    logging_name(class_names)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    capture_images(DATASET_SIZE, NUMBER_OF_CLASSES)
    proces_image_data(DATA_DIR, CLASS_NAMES)


if __name__ == '__main__':
    main()