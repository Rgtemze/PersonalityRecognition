import copy
import datetime
import os.path
import time

import cv2
import zipfile
from tqdm import tqdm
import h5py
import numpy as np
from os import walk
import pandas as pd
import torch
torch.set_anomaly_enabled(True)
is_train = True

# OUR CONTRIBUTION

# annotation_paths = ['UDIVA/ghost_annotations_train', 'UDIVA/animals_annotations_train', 'UDIVA/lego_annotations_train',
#                     'UDIVA/talk_annotations_train']
# annotation_paths = ['UDIVA_val/ghost_annotations_val', 'UDIVA_val/animals_annotations_val', 'UDIVA_val/lego_annotations_val',
#                     'UDIVA_val/talk_annotations_val']
annotation_paths = ['UDIVA_test/ghost_annotations_test', 'UDIVA_test/animals_annotations_test', 'UDIVA_test/lego_annotations_test',
                    'UDIVA_test/talk_annotations_test']

def extract_zip_files(path):
    f = []
    for (dirpath, dirnames, filenames) in tqdm(list(walk(path))):
        if len(filenames) == 0 or "annotations_raw.hdf5" in filenames: continue
        if len(filenames) == 1:
            file_name = filenames[0]
        else:
            file_name = "annotations_cleaned_unmasked.zip"
        with zipfile.ZipFile(os.path.join(dirpath, file_name)) as zip:
            zip.extractall(dirpath)

        f.extend(filenames)
    print(len(f))
# for annotation_path in annotation_paths:
#     extract_zip_files(annotation_path)

ses_df = pd.read_csv("UDIVA_test/metadata_test/sessions_test.csv", dtype={'ID': str})
ppl_df = pd.read_csv("UDIVA_test/metadata_test/parts_test.csv")
# ses_df = pd.read_csv("UDIVA_val/metadata_val/sessions_val.csv", dtype={'ID': str})
# ppl_df = pd.read_csv("UDIVA_val/metadata_val/parts_val.csv")
# ses_df = pd.read_csv("UDIVA/metadata_train/sessions_train.csv", dtype={'ID': str})
# ppl_df = pd.read_csv("UDIVA/metadata_train/parts_train.csv")
number_of_people = len(ppl_df)
print(number_of_people)
if is_train:
    OCEAN_COLUMNS = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z',
                 'NEGATIVEEMOTIONALITY_Z']
    oceans = ppl_df[OCEAN_COLUMNS].to_numpy()
    oceans_median = np.median(oceans, axis=0)
    print(oceans_median)
chunk_size = 250
input_size = 24 * 3 + 68 * 3
use_file = False

reference_joints = [6]  #1
resized_dims = (chunk_size, 24)
# resized_dims = (68, 68)

X_train = np.load("X_train.npy") if use_file else np.zeros((0, resized_dims[0], resized_dims[1], 3))
y_train = np.load("y_train.npy") if use_file else np.zeros((0, 5))
info_train = np.load("info_train.npy") if use_file else np.zeros((0, 3))
X_val = np.load("X_val.npy") if use_file else np.zeros((0, resized_dims[0], resized_dims[1], 3))
y_val = np.load("y_val.npy") if use_file else np.zeros((0, 5))
info_val = np.load("info_val.npy") if use_file else np.zeros((0, 3))
X_test = np.load("X_test.npy") if use_file else np.zeros((0, resized_dims[0], resized_dims[1], 3))
y_test = np.load("y_test.npy") if use_file else np.zeros((0, 5))
info_test = np.load("info_test.npy") if use_file else np.zeros((0, 3))

person_index_randomizer = np.arange(number_of_people)
np.random.shuffle(person_index_randomizer)
print(person_index_randomizer)
if not use_file:
    index = 0
    for annotation_path in annotation_paths:
        for (dirpath, dirnames, filenames) in tqdm(list(walk(annotation_path))):
            if len(filenames) == 0: continue
            for fname in filenames:
                if "hdf5" in fname:
                    filename = fname
                    break
            path = os.path.join(dirpath, filename)
            session_id, part_id = dirpath.split("\\")[-2:]
            is_fc1 = "FC1" in part_id
            part_column = "PART.1" if is_fc1 else "PART.2"
            person_id = ses_df[ses_df['ID'] == session_id][part_column].to_numpy()[0]
            if is_train:
                ocean_scores = ppl_df[ppl_df['ID'] == person_id][OCEAN_COLUMNS].to_numpy()[0]
            person_index = ppl_df.index[ppl_df['ID'] == person_id]

            session = h5py.File(path, 'r')
            number_of_frames = len(session.keys())
            # session_data =
            session_data = []
            missing = 0
            for frame_key in session.keys():
                frame_no = int(frame_key)
                # if frame_no % 2 == 1: continue
                if not ("body" in session[frame_key].keys()):
                    missing += 1
                    continue
                # if not ("hands" in session[frame_key].keys()) \
                #         or len(session[frame_key]["hands"]['left'].keys()) == 0\
                #         or len(session[frame_key]["hands"]['right'].keys()) == 0:
                #     missing += 1
                #     continue
                if not ('landmarks' in session[frame_key]["body"].keys()):
                    missing += 1
                    continue
                # landmarks_ds = session[frame_key]["body"]['landmarks']
                # landmarks = np.array(landmarks_ds[()], dtype=float)
                # body_landmarks = np.array(session[frame_key]["body"]['landmarks'][()], dtype=float)
                # face_landmarks = np.array(session[frame_key]["face"]['landmarks'][()], dtype=float)
                landmarks_ds = session[frame_key]["body"]['landmarks']
                body_landmarks = np.array(landmarks_ds[()], dtype=float)
                # left_hand_landmarks = np.array(session[frame_key]['hands']["left"]['landmarks'][()], dtype=float)
                # right_hand_landmarks = np.array(session[frame_key]['hands']["right"]['landmarks'][()], dtype=float)
                # print(session[frame_key]["hands"]["left"].keys())
                # landmarks_body = np.array(session[frame_key]["hands"]['left']['landmarks'][()], dtype=float)
                # landmarks = np.vstack((landmarks, landmarks_body))

                final_landmarks_data = np.zeros((0, 3))
                # for i, reference_joint in enumerate([34]):
                #     landmarks_diff = landmarks_head - landmarks_head[reference_joint]
                #     # # landmarks_diff = np.delete(landmarks_diff, reference_joint, axis=0)
                #     # p = np.array(list(map(lambda v: np.linalg.norm(v[:2]), landmarks_diff)))
                #     # theta = np.array(list(map(lambda v: np.arctan2(v[1], v[0]), landmarks_diff)))
                #     # z = np.array(list(map(lambda v: v[2], landmarks_diff)))
                #     # # print(landmarks_diff[0], p[0], theta[0], z[0])
                #     # # person_frame_data_rs = person_frame_data.reshape((1, feature_size))
                #     # cylindirical_data = (np.dstack((p, theta, z))[0]) # (23, 3)
                #     # cylindirical_data[reference_joint] = np.zeros((3,))
                #
                #
                #     # landmarks_diff = np.delete(landmarks_diff, [3, 0, 1, 2, 4, 5, 7, 8, 10, 11], 0)
                #     final_landmarks_data = np.vstack((final_landmarks_data, landmarks_diff))
                # visualize(np.expand_dims(landmarks, 0), edges)

                session_data.append(body_landmarks)
                # if len(session[frame_key]["face"].keys()) == 0: continue
                # face_ds = session[frame_key]["face"]['landmarks']
                # face = np.array(face_ds[()], dtype=float).flatten()
                #
                # session_data.append(np.hstack((landmarks, face)))
            session_data = np.array(session_data)
            if(len(session_data) < chunk_size):
                print('Not enough: ', len(session_data))
                continue

            splitted = np.array(list(zip(*[iter(session_data)] * chunk_size)))
            info = np.array([session_id, person_id, splitted.shape[0]])
            # info = np.tile(info, (splitted.shape[0], 1))
            if is_train:
                y_person = np.tile(ocean_scores, (splitted.shape[0], 1))

            randomized_person_index = person_index_randomizer[person_index]
            if randomized_person_index < number_of_people * 0:
                X_train = np.vstack((X_train, splitted))
                if is_train:
                    y_train = np.vstack((y_train, y_person))
                info_train = np.vstack((info_train, info))
            elif randomized_person_index < number_of_people * 1:
                X_val = np.vstack((X_val, splitted))
                y_val = np.vstack((y_val, y_person))
                info_val = np.vstack((info_val, info))
            else:
                X_test = np.vstack((X_test, splitted))
                y_test = np.vstack((y_test, y_person))
                info_test = np.vstack((info_test, info))
            session.close()
            index += 1

    np.save("body_keypoints_full_test_250/X_train", X_train)
    np.save("body_keypoints_full_test_250/y_train", y_train)
    np.save("body_keypoints_full_test_250/info_train", info_train)
    np.save("body_keypoints_full_test_250/X_val", X_val)
    np.save("body_keypoints_full_test_250/y_val", y_val)
    np.save("body_keypoints_full_test_250/info_val", info_val)
    np.save("body_keypoints_full_test_250/X_test", X_test)
    np.save("body_keypoints_full_test_250/y_test", y_test)
    np.save("body_keypoints_full_test_250/info_test", info_test)
