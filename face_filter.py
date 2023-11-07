import face_recognition
import os
import cv2
import numpy as np
import spacy
import PIL

def rec_face_ids(image_path, output_folder):
     
    image = face_recognition.load_image_file(image_path)

    # Recognize all face positions in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) > 0:
        # If the image contains a face, save it to the destination folder
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f'{filename} contains a face, save it to {output_path}')


input_folder = 'data/filtered_face'
output_folder = 'data/twitter15_face'
os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        rec_face_ids(image_path, output_folder)

print('Face screening complete')