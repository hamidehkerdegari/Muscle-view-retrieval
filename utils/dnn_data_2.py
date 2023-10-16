__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

##Data generator is defined here.

import numpy as np
import tensorflow as tf
import os



def flip(matrix: np.array):
    return np.flip(matrix, axis=2)


def random_crop(matrix: np.array, crop_shape: tuple):
    if len(matrix.shape) != 4:
        print(matrix.shape)
        raise Exception
    _, x, y, _ = matrix.shape

    x_left = np.random.randint(low=0, high=x-crop_shape[0])
    x_right = x_left + crop_shape[0]

    y_left = np.random.randint(low=0, high=y - crop_shape[1])
    y_right = y_left + crop_shape[1]

    matrix[:, x_left:x_right, y_left:y_right, :] = 0.0
    return matrix


def augment(matrix: np.array, probability: float = 1.0):
    for x in matrix:
        if 0.0 <= probability <= 1.0:
            if np.random.rand() < probability:
                r = np.random.rand()
                if r > 1.0/3.0:
                    x = random_crop(x, crop_shape=(10, 10))
                if r < 1.0-(1.0/3.0):
                    x = flip(x)
    return matrix

class DataGenerator(tf.keras.utils.Sequence): # Data generator for self supervised learning
    def __init__(self, dataset_paths: list, batch_size: int, verbose: bool=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.dataset_paths = dataset_paths
        self.data_paths = self.get_data_paths()
        self.data = self.load_data(self.data_paths)
        self.on_epoch_end()

    def get_data_paths(self):
        npy_paths = []
        for data_path in self.dataset_paths:  # For each patient
            for root, _, files in os.walk(data_path, topdown=False):
                for file in files:
                    if file.endswith("_train.npy") and file.replace('_train.npy', '.mp4') in files:   # if npy and mp4 existed.
                        npy_paths.append(os.path.join(root, file))
        if self.verbose:
            print('Number of npy files:', len(npy_paths))
        return npy_paths

    def load_data(self, npy_paths: list):
        data = []
        for npy in npy_paths:
            d = np.load(npy)
            if len(d.shape) == 4:
                for i in range(0, d.shape[0]-10, 10):
                    data.append(d[i:i+10])
        data = np.array(data)
        if self.verbose:
            print('Dataset size:', data.shape)
        return data

    def on_epoch_end(self):
        np.random.shuffle(self.data)

    def data_generation(self, batch_data: np.array):
        x1 = batch_data[:, 0:5]
        x2 = batch_data[:, 5:]

        #return x1, x2  #this line uncommented for baseline training
        return augment(x1, probability=0.8), augment(x2, probability=0.8)

    def __len__(self):  # Number of batches.
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):  # Get batch of index.
        batch_data = self.data[index * self.batch_size: (index + 1) * self.batch_size]
        x1, x2 = self.data_generation(batch_data)
        return x1, x2


class DataGenerator_pn(DataGenerator): # Data generator for positive and negative pairs for supervised training
    def get_data_paths(self):
        npy_paths = []
        for data_path in self.dataset_paths:  # For each patient
            for root, _, files in os.walk(data_path, topdown=False):
                for file in files:
                    if file in ['n_pairs.npy', 'p_pairs.npy']:  # if npy and mp4 existed.
                        npy_paths.append(os.path.join(root, file))
        if self.verbose:
            print('Number of npy files:', len(npy_paths))
        return npy_paths

    def load_data(self, npy_paths: list):
        data = np.zeros((0, 10, 64, 64, 1))
        label = np.zeros((0, 2))
        for npy in npy_paths:
            d = np.load(npy)
            if str(npy).endswith('n_pairs.npy'):
                l = np.array([[0., 1.]]*d.shape[0])
            if str(npy).endswith('p_pairs.npy'):
                l = np.array([[1., 0.]]*d.shape[0])

            if len(d.shape) == 5:
                data = np.concatenate((data, d), axis=0)
                label = np.concatenate((label, l), axis=0)
        if self.verbose:
            print('Dataset size:', data.shape)
        print('Label shape', label.shape)
        return [data, label]

    def on_epoch_end(self):
        pass


