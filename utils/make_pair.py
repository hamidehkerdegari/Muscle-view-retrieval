__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

# This script is used to make pairs of data (positive and negative data that were generated using positive_negative.py file) in numpy format. Then this code makes a single
#positive (p_pairs.npy) and a single negative (n_pairs.py) which is used for training of supervised baseline model, self-supervised evaluation model and their evaluation later.
# It includes all the combinations within each video and between videos (T1, T2 and T3)

import json
import os
import random

import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def make_pairs(data_paths_1: list, data_paths_2: list, max_len: int=20):
    data = []
    for i in range(len(data_paths_1)):
        data_p = np.load(data_paths_1[i])
        index_p = list(range(data_p.shape[0]))
        random.shuffle(index_p)
        for j in range(len(data_paths_2)):
            data_n = np.load(data_paths_2[j])
            index_n = list(range(data_n.shape[0]))
            random.shuffle(index_n)
            for x1 in index_p[0: max_len]:
                for x2 in index_n[0: max_len]:
                    data.append(np.append(data_p[x1][0:5], data_n[x2][0:5], axis=0))
    return np.array(data)

def read_path(dataset_path: str):
    for data_folder_name in sorted(os.listdir(dataset_path)):
        pl = []
        pr = []
        nl = []
        nr = []
        for t_folder in ['T1', 'T2', 'T3']:
            if os.path.exists(os.path.join(dataset_path, data_folder_name, t_folder)):
                npy_file_names = []
                annotation_file_names = []
                tmp = [file_name for file_name in os.listdir(os.path.join(dataset_path, data_folder_name, t_folder)) if file_name.endswith(".mp4")]
                if len(tmp):
                    name = tmp[0].split('.')[0]
                    npy_file_path = os.path.join(dataset_path, data_folder_name, t_folder, name)
                    json_file_path = os.path.join(dataset_path, data_folder_name, t_folder, name + '.json')
                    if os.path.exists(json_file_path):
                        npy_file_names.append(npy_file_path)
                        annotation_file_names.append(json_file_path)

                for i in range(len(annotation_file_names)):
                    with open(annotation_file_names[i], "r") as in_f:
                        annotation = json.load(in_f)
                    for j, ant in enumerate(annotation["annotations"]):
                        if ant["label"] == "l":
                            _pl = os.path.join(npy_file_names[i], str(j) + "_p.npy")
                            _nl = os.path.join(npy_file_names[i], str(j) + "_n.npy")
                            if os.path.exists(_pl):
                                pl.append(_pl)
                            if os.path.exists(_nl):
                                nl.append(_nl)
                        elif ant["label"] == "r":
                            _pr = os.path.join(npy_file_names[i], str(j) + "_p.npy")
                            _nr = os.path.join(npy_file_names[i], str(j) + "_n.npy")
                            if os.path.exists(_pr):
                                pr.append(_pr)
                            if os.path.exists(_nr):
                                nr.append(_nr)

        # pr, pl, nr, nl
        pr_pairs = make_pairs(pr, pr, max_len=20)[0:100]
        pl_pairs = make_pairs(pl, pl, max_len=20)[0:100]
        if any([pr_pairs.shape[0], pl_pairs.shape[0]]):
            p_data = np.concatenate([x for x in [pr_pairs, pl_pairs] if x.shape[0]], axis=0)
        else:
            p_data = None
        n_pairs_1 = make_pairs(pr, nr, max_len=20)[0:50]
        n_pairs_2 = make_pairs(pr, nl, max_len=20)[0:50]
        n_pairs_3 = make_pairs(pr, pl, max_len=20)[0:50]
        n_pairs_4 = make_pairs(nr, nl, max_len=20)[0:50]
        if any([n_pairs_1.shape[0], n_pairs_2.shape[0], n_pairs_3.shape[0], n_pairs_4.shape[0]]):
            n_data = np.concatenate([x for x in [n_pairs_1, n_pairs_2, n_pairs_3, n_pairs_4] if x.shape[0]], axis=0)
        else:
            n_data = None

        # if p_data is not None and n_data is not None:
        #     print(p_data.shape, n_data.shape)

        if p_data is not None:
            np.save(os.path.join(dataset_path, data_folder_name, "p_pairs.npy"), p_data)
        else:
            print("There is no data")
        if n_data is not None:
            np.save(os.path.join(dataset_path, data_folder_name, "n_pairs.npy"), n_data)

        # pl_pairs = make_pairs(pl)
        # nr_pairs = make_pairs(nr)
        # nl_pairs = make_pairs(nl)



    return npy_file_names, annotation_file_names


npy_file_names, annotation_file_names = read_path(dataset_path='dataset path') # path of data


