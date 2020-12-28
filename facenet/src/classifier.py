"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet.src.facenet as fc
import os
import sys
import math
import pickle
from sklearn.svm import SVC


def main(
    args_data_dir,
    args_model,
    args_classifier_filename,
    args_use_split_dataset,
    args_mode="CLASSIFY",
    args_batch_size=90,
    args_image_size=160,
    args_seed=666,
    args_min_nrof_images_per_class=20,
    args_nrof_train_images_per_class=10,
):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args_seed)

            if args_use_split_dataset:
                dataset_tmp = fc.get_dataset(args_data_dir)
                train_set, test_set = split_dataset(
                    dataset_tmp, args_min_nrof_images_per_class, args_nrof_train_images_per_class
                )
                if args_mode == "TRAIN":
                    dataset = train_set
                elif args_mode == "CLASSIFY":
                    dataset = test_set
            else:
                dataset = fc.get_dataset(args_data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, "There must be at least one image for each class in the dataset")

            paths, labels = fc.get_image_paths_and_labels(dataset)

            print("Number of classes: %d" % len(dataset))
            print("Number of images: %d" % len(paths))

            # Load the model
            print("Loading feature extraction model")
            fc.load_model(args_model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print("Calculating features for images")
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args_batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args_batch_size
                end_index = min((i + 1) * args_batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = fc.load_data(paths_batch, False, False, args_image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(args_classifier_filename)

            if args_mode == "TRAIN":
                # Train classifier
                print("Training classifier")
                model = SVC(kernel="linear", probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace("_", " ") for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, "wb") as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif args_mode == "CLASSIFY":
                # Classify images
                print("Testing classifier")
                with open(classifier_filename_exp, "rb") as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print("%4d  %s: %.3f" % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print("Accuracy: %.3f" % accuracy)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(fc.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(fc.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


# if __name__ == '__main__':
#     main(args_data_dir, args_classifier_filename, args_use_split_dataset, args_mode, args_batch_size, args_image_size, args_seed, args_min_nrof_images_per_class, args_nrof_train_images_per_class)
