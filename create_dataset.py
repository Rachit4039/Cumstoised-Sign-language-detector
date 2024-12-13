import os
import pickle

import mediapipe as mp
import cv2


mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1
)
mp_drawing = mp.solutions.drawing_utils


def process_image(image_path, label):
    """
    Processes a single image and extracts hand landmark coordinates.

    Args:
        image_path: Path to the image file.
        label: Label associated with the image (sign class).

    Returns:
        A list of hand landmark coordinates (normalized) or None if no hands detected.
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Normalize coordinates
                # Alternative: Offset by minimum x and y for each hand
                data_aux.append(x - min(lm.x for lm in hand_landmarks.landmark))
                data_aux.append(y - min(lm.y for lm in hand_landmarks.landmark))
            return data_aux
    else:
        print(f"No hands detected in image: {image_path}")
        return None


DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        processed_data = process_image(os.path.join(DATA_DIR, dir_, img_path), dir_)
        if processed_data:
            data.append(processed_data)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Data processing complete!")
