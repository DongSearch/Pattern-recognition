#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de

#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software:
#    you can redistribute it and/or modify it under the terms of the
#    GNU General Public License as published by the Free Software Foundation,
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.
#    If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np
import numpy
import sklearn.neighbors

import numpy as np
from collections import Counter
class kNearestNeighbor(object):
    def __init__(self, k):
        """
        Initialize the class with parameter ``k`` (number of neighbors)
        """
        # X_train : training data feature vectors
        # y_tain : training data feature vectors
        # k : the number of neighbors
        self.X_train = None
        self.y_train = None
        self.num = k

    def fit(self, X, y):
        """
        Training step: Initializes this class with labeled training data

        X has shape (N,2) and contains N points in 2-d feature space
        y has shape (N,) and contains N labels. One for each feature vector
        """
        self.X_train = X
        self.y_train = y
        pass

    def predict(self, X):

        """
        Classification step

        X has shape (M,2). It contains M feature vectors to classify in 2-d feature space

        Returns:
            A vector of shape (M,) with M classification results (class labels)
        """
        prediction = np.empty(X.shape[0])
        for i, x in enumerate(X) :
            distances = np.linalg.norm(self.X_train - x, axis=1)  # Efficient Euclidean distance
            k_indices = np.argsort(distances)[:self.num]
            knn_labels = self.y_train[k_indices]
            # majority_label is for classification, we need to decide only a single label that
            # appears most frequently among k neighbors
            majority_label = Counter(knn_labels).most_common(1)[0][0]
            prediction[i] = majority_label

        return prediction

        pass  # TODO


def test_kNN():
    """
    Use this to test your code before using the GUI.

    When it's working you can try your algorithm in the GUI

    1. start Toolbox/PyClassificationToolbox.py
    2. select "Classification > kNN..." in the menu
    3. choose "kNN" (not "kNN-scikit" learn in the drop down menu)
    4. See your predication as background color
    5. Load a different data set

    When your algorithm is slow, try smaller window size!
    """
    num_classes = 4
    num_neighbors = 3
    num_training_samples = 20
    num_inference_samples = 15
    noise_magnitude = 0.2
    sample_spread = 10

    classifier = kNearestNeighbor(num_neighbors)

    cluster_centers = sample_spread * np.random.rand(num_classes, 2)

    # This is our true y
    sample_labels_training = np.random.randint(num_classes, size=(num_training_samples,))
    # These are the positions of the training samples
    sample_position_training = cluster_centers[sample_labels_training] + \
        noise_magnitude * np.random.randn(num_training_samples, 2)

    # During training we know the labels
    classifier.fit(sample_position_training, sample_labels_training)

    # This is our true y
    sample_labels_inference = np.random.randint(num_classes, size=(num_inference_samples,))
    # These are the positions of the training samples
    sample_position_inference = cluster_centers[sample_labels_inference] + \
        noise_magnitude * np.random.randn(num_inference_samples, 2)

    # During inference we test what we have learned
    prediction = classifier.predict(sample_position_inference)

    correct = [p == t for p, t in zip(prediction, sample_labels_inference)]

    accuracy = sum(correct) / num_inference_samples
    # This line requires Python 3.8
    print(f'{accuracy=}')

    sk_classifier = sklearn.neighbors.KNeighborsClassifier(num_neighbors)
    sk_classifier.fit(sample_position_training, sample_labels_training)
    sk_prediction = sk_classifier.predict(sample_position_inference)
    assert np.allclose(sk_prediction, prediction)

if __name__ == '__main__':
    test_kNN()
