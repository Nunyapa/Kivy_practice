import cv2

import os
import pickle
from PIL import Image
import numpy as np

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

class_counter = 0
label_dict = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):

            path = os.path.join(root, file)

            label = os.path.basename(os.path.dirname(path)).lower()

            if not label in label_dict:
                class_counter += 1
                label_dict[label] = class_counter

            id_ = label_dict[label]

            pil_image = Image.open(path).convert("L") # grayscale

            image_array = np.array(pil_image, 'uint8')
            
            x_train.append(image_array)
            y_labels.append(id_)
            
            #faces = classifier.detectMultiScale(image_array, 1.1, 5)

            # for (x, y, w, h) in faces:
            #     x_train.append(image_array[y: y + h, x: x + w])
            #     y_labels.append(id_)
print(label_dict)
print(y_labels)
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_dict, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trained.yml')





