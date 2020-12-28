from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import argparse
import json
import uuid
import os
import pickle
import collections
import tensorflow as tf
import imageio
from facenet.src import facenet
import sys
import math
from facenet.src import align
import numpy as np
import cv2
from sklearn.svm import SVC

# removes tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# input format


def find_duplicate(face_list):
    names_with_duplicate = {}
    for key in face_list:
        if face_list[key]["name"] in names_with_duplicate.keys():
            names_with_duplicate[face_list[key]["name"]]["count"] += 1
            names_with_duplicate[face_list[key]["name"]]["ids"].append(key)
        else:
            names_with_duplicate[face_list[key]["name"]] = {"count": 1, "ids": [key]}
    key_list = list(names_with_duplicate.keys())
    for key in key_list:
        if names_with_duplicate[key]["count"] == 1:
            del names_with_duplicate[key]
    name_list = []
    for names in names_with_duplicate.keys():
        best_id = 0
        best_prob = 0
        for id in names_with_duplicate[names]["ids"]:
            if best_prob < face_list[id]["prob"]:
                best_prob = face_list[id]["prob"]
                best_id = id
        names_with_duplicate[names]["count"] -= 1
        names_with_duplicate[names]["ids"].remove(best_id)
        name_list += names_with_duplicate[names]["ids"]

    return name_list


def replace_with_second_best(recognised_data):
    if len(recognised_data["second_best_name"]) > 0.509:
        recognised_data["name"] = recognised_data["second_best_name"]
        recognised_data["best_name"] = recognised_data["second_best_name"]
        recognised_data["second_best_name"] = recognised_data["third_best_name"]
        recognised_data["third_best_name"] = ""
        recognised_data["prob"] = recognised_data["prob_list"][recognised_data["name"]]
    else:
        recognised_data["name"] = "Unknown"
        recognised_data["best_name"] = "Unknown"
        recognised_data["prob"] = 0
    return recognised_data


def main(classifierpath, slotid, imagepath, pretrained_model):

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = classifierpath
    SLOTID = slotid
    IMAGE_PATH = imagepath
    FACENET_MODEL_PATH = pretrained_model

    IMAGE_PATH = IMAGE_PATH[1:-1].split(",")
    # return

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, "rb") as file:
        model, class_names = pickle.load(file)
    # print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            # print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "./TheMajorProject/main/src/align")

            # people_detected = set()
            person_detected = collections.Counter()

            ctr = 1
            face_list = {}
            unknown_list = {}
            unknown_count = 0
            detected_face = {}
            for IMG_ADDR in IMAGE_PATH:
                ctr = ctr + 1
                frame = imageio.imread(IMG_ADDR)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            cropped = frame[bb[i][1] : bb[i][3], bb[i][0] : bb[i][2], :]
                            # print(type(cropped))
                            scaled = cv2.resize(
                                cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
                            )
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            prob_claa = {}
                            best_prob = 0
                            second_best_prob = 0
                            third_best_prob = 0
                            best_name = ""
                            second_best_name = ""
                            third_best_name = ""
                            for iterr in range(len(predictions[0])):
                                prob_claa[str(class_names[iterr])] = predictions[0][iterr]
                                if predictions[0][iterr] > best_prob:
                                    best_prob = predictions[0][iterr]
                                    best_name = str(class_names[iterr])
                            for key in prob_claa.keys():
                                if (prob_claa[key] > second_best_prob) and (prob_claa[key] < best_prob):
                                    second_best_name = key
                                    second_best_prob = prob_claa[key]
                            for key in prob_claa.keys():
                                if (prob_claa[key] > third_best_prob) and (prob_claa[key] < second_best_prob):
                                    third_best_name = key
                                    third_best_prob = prob_claa[key]
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices
                            ]
                            best_name = class_names[best_class_indices[0]]
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            if best_class_probabilities > 0.60:
                                name = class_names[best_class_indices[0]]
                                # print(name)
                                id = uuid.uuid1().int
                                entry = {
                                    "name": name,
                                    "prob": best_class_probabilities[0],
                                    "best_name": best_name,
                                    "second_best_name": second_best_name,
                                    "third_best_name": third_best_name,
                                    "prob_list": prob_claa,
                                }
                                detected_face[id] = cropped
                                # imageio.imwrite("detected_faces/{}-{}.jpg".format(entry['name'], entry['prob']), cropped)
                                face_list[id] = entry
                                # cv2.putText(frame, name[10:], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                            else:
                                id = uuid.uuid1().int
                                detected_face[id] = cropped
                                name = "Unknown" + str(unknown_count)
                                guess = class_names[best_class_indices[0]]
                                prob = best_class_probabilities[0]
                                entry = {
                                    "name": name,
                                    "guess": guess,
                                    "prob": prob,
                                }
                                unknown_count += 1
                                unknown_list[id] = entry
                                # cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)

                            # cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                except:
                    pass

                # cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # cap.release()
            duplicate_keys = find_duplicate(face_list)
            present_list = ""
            # for key in duplicate_keys:
            #     face_list[key] = replace_with_second_best(face_list[key])
            for key in detected_face.keys():
                if key in face_list:
                    imageio.imwrite(
                        "./static/{}--{}-{}.jpg".format(SLOTID, face_list[key]["name"], face_list[key]["prob"]),
                        detected_face[key],
                    )
                    present_list = present_list + face_list[key]["name"][:9] + ","
                else:
                    imageio.imwrite(
                        "./static/{}_{}-{}.jpg".format(SLOTID, unknown_list[key]["name"], unknown_list[key]["prob"]),
                        detected_face[key],
                    )
            # print(json.dumps(face_list, indent=4))
            present_list = present_list[:-1]
            print(present_list, end="")
            cv2.destroyAllWindows()
