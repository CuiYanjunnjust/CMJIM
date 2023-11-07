import face_recognition
import os
import cv2
import numpy as np
import spacy
import csv
from tqdm import tqdm
import re
import sys


nlp = spacy.load("en_core_web_sm")
TSV_FILE_DIR = "data/Evan"
IMAGE_DIR = "data/Evan"
STANDARD_IMAGE_DIR = "data/standard_Evan"
# output_folder = 'data/twitter15_face'

def reset_standard_image_dir(standard_image_dir):

    file_list = os.listdir(standard_image_dir)


    for filename in file_list:
        file_path = os.path.join(standard_image_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # 删除文件
                print(f"deleted file: {file_path}")
        except Exception as e:
            print(f"An error occurred while deleting the file: {str(e)}")

    print("All files have been deleted.")


def edit_distance(str1, str2):

    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]


    for i in range(len(str1) + 1):
        dp[i][0] = i
    for j in range(len(str2) + 1):
        dp[0][j] = j


    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost
                          )

    # Return edit distance
    return dp[len(str1)][len(str2)]


def rec_per_names(text_column_value):

    doc = nlp(text_column_value)

    # Name extraction
    people_names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            per_name = re.sub(r'[^a-zA-Z\s]', '', ent.text).strip()
            if len(per_name) > 0:
                min_distance, index = sys.maxsize, -1
                for i, known_per_name in enumerate(known_per_names):
                    distance = edit_distance(known_per_name.lower(), per_name.lower())
                    if distance < min_distance:
                        min_distance, index = distance, i
                if min_distance < 3:
                    per_name = known_per_names[index]
                else:
                    known_per_names.append(per_name)
                people_names.append(per_name)

                # print("People Names:", people_names)
    return people_names


def rec_face_ids(image_path):
    face_ids = []

    image = face_recognition.load_image_file(image_path)


    face_locations = face_recognition.face_locations(image)
    # Find all the faces and face encodings in the current frame of video
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if len(face_encodings) > 0:
        for face_location, face_encoding in zip(face_locations, face_encodings):
            if len(known_faces_encodings) == 0:
                face_id = gen_new_face(face_encoding, face_location, image)
            else:
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:

                    face_id = best_match_index
                else:
                    face_id = gen_new_face(face_encoding, face_location, image)

            face_ids.append(face_id)
    # else:
    #     print(f"No face found in {image_path}")
    return face_ids


def gen_new_face(face_encoding, face_location, image_raw):

    image = image_raw.copy()

    face_id = len(known_faces_encodings)
    known_faces_encodings.append(face_encoding)
    # Display the results
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, str(face_id), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    cv2.imwrite(f'{STANDARD_IMAGE_DIR}/{face_id}.jpg', image)
    return face_id


# Store known names for name alignment algorithm based on edit distance
known_per_names = []

known_faces_encodings = []

co_occurrence = {}

images = [image for image in os.listdir(IMAGE_DIR) if image.endswith('.jpg')]
tsv_files = [file for file in os.listdir(TSV_FILE_DIR) if file.endswith('.tsv')]
reset_standard_image_dir(STANDARD_IMAGE_DIR)


def deal_tsv_file(tsv_file):
    tsv_file_path = os.path.join(TSV_FILE_DIR, tsv_file)

    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')

        next(reader, None)

        for row in tqdm(reader):
            image_column_value = row[2]
            text_column_value = row[3]

            if image_column_value in images:
                people_names = rec_per_names(text_column_value)

                if len(people_names) > 0:
                    image_path = os.path.join(IMAGE_DIR, image_column_value)
                    face_ids = rec_face_ids(image_path)

                    if len(face_ids) > 0:
                        print(f"{image_column_value}: {text_column_value}")
                        for face_id in face_ids:
                            if face_id in co_occurrence:
                                people_name_dic = co_occurrence[face_id]
                            else:
                                people_name_dic = {}
                                co_occurrence[face_id] = people_name_dic
                            for people_name in people_names:
                                if people_name in people_name_dic:
                                    people_name_dic[people_name] += 1
                                else:
                                    people_name_dic[people_name] = 1
                                print(f"{face_id}, {people_name}, {people_name_dic[people_name]}")



for tsv_file in tsv_files:
    deal_tsv_file(tsv_file)

print(len(co_occurrence))
print(co_occurrence)
print(known_per_names)
for face_id in co_occurrence:

    people_name_dic = co_occurrence[face_id]

    max_key = max(people_name_dic, key=people_name_dic.get)

    if people_name_dic[max_key] > 1:

        print(f"{face_id}: {max_key}: {people_name_dic[max_key]}")

cv2.destroyAllWindows()