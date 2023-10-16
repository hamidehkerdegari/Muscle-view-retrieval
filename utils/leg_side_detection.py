__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

import os
import sys
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#This code generates labels for left leg and right leg scans and save them in the json file that already generated for each standard view.

class MuscleUltrasoundDatasetAnnotate:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path  # /data/Group2-MUSCLE/CNS
        self.video_file_paths, self.annotation_file_names = self._read_path(dataset_path)
        self.label = None

    def _read_path(self, dataset_path: str):
        video_file_names = []
        annotation_file_names = []
        for data_folder_name in sorted(os.listdir(dataset_path)):
            for t_folder in sorted(os.listdir(os.path.join(dataset_path, data_folder_name))):
                if t_folder in ['T1', 'T2', 'T3']:
                    for file_name in os.listdir(os.path.join(dataset_path, data_folder_name, t_folder)):
                        name = file_name.split('.')[0]
                        video_file_path = os.path.join(dataset_path, data_folder_name, t_folder, file_name)
                        json_file_path = os.path.join(dataset_path, data_folder_name, t_folder, name + '.json')
                        if video_file_path.endswith('.mp4') and os.path.exists(json_file_path):
                            video_file_names.append(video_file_path)
                            annotation_file_names.append(json_file_path)
        return video_file_names, annotation_file_names

    def _time2msec(self, time_str: str):
        splt = time_str.split(':')

        minutes = float(splt[0]) * 1000 * 60
        seconds = float(splt[1]) * 1000
        ms = float(splt[2])

        return minutes + seconds + ms

    def _get_rand_frame(self, cap, annotation):
        st = self._time2msec(annotation['start_time'])
        et = self._time2msec(annotation['end_time'])
        rt = st + (np.random.random() * (et - st))

        cap.set(cv.CAP_PROP_POS_MSEC, rt)
        success, frame = cap.read()
        if not success:
            print('[ERROR] [MuscleUltrasoundDataset.__getitem__()] Unable to extract frame at ms {} from video '.format(msec))
            return None
        return frame

    def _on_press(self, event):
        if str(event.key).lower() in ["l", "r", "n", "escape"]:
            sys.stdout.flush()
            self.label = event.key
            plt.close()
            if str(event.key).lower() != "escape":
                self._found = True

    def _show_frame(self, frame, title: str):
        self._fig, self._ax = plt.subplots()
        self._fig.canvas.mpl_connect('key_press_event', self._on_press)
        plt.imshow(frame)
        plt.title(title + "\nKeys r: Right, l: left, Esc: show another frame, n: not known")
        plt.show()

    def run_annotate(self, index: int):
        video_file_path = self.video_file_paths[index]
        cap = cv.VideoCapture(video_file_path)
        if cap.isOpened() == False:
            print('[ERROR] [MuscleUltrasoundDataset.__getitem__()] Unable to read video ' + self.video_filenames[index])
            exit(-1)

        with open(self.annotation_file_names[index], "r") as json_file:
            json_data = json.load(json_file)

        for s, annotation in enumerate(json_data['annotations']):
            self._found = False
            while not self._found:
                frame = self._get_rand_frame(cap, annotation)
                self._show_frame(frame, title=self.annotation_file_names[index].split("/")[-1] + " " + str(s))
                print(self.label)
                annotation["label"] = self.label
        with open(self.annotation_file_names[index], "w") as json_file:
            json.dump(json_data, json_file, indent=4)


def main_run(dataset_path: str):
    muscle = MuscleUltrasoundDatasetAnnotate(dataset_path)

    for i in range(len(muscle.video_file_paths)):
        print(muscle.video_file_paths[i])
        muscle.run_annotate(i)


main_run(dataset_path='dataset path')  # Path of data
