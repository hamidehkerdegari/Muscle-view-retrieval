__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

# This script is used to generate numpy data from mp4 videos. It reads all the data, and crop the videos, then normalize them. Further it extracts frame from videos: 10 continous frames with the length
# of 10 and gap of 5 (for training). For evaluation: it extracts frame from videos: 10 continous frames with the length of 10 without gap (for evaluation).

import json
import os
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def norm_dot(img1, img2):
    """
    return normalized dot product of the arrays img1, img2
    """
    # make 1D value lists
    v1 = np.ravel(img1)
    v2 = np.ravel(img2)

    # get the norms of the vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    #print('norms of NDP vectors: ', norm1, norm2)

    ndot = np.dot( v1/norm1, v2/norm2)
    return ndot

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def print_similarity_measures(img1, img2, nc0=None, nd0=None):
    nd = norm_dot(img1, img2)
    nc = ncc(img1, img2)
    # print('NCC: ', nc, ' NDP: ', nd)
    return nc

def comp_frames(frames, thr: float=0.00001):
  var = np.var(frames, axis=0)  #
  m = np.mean(var, axis=(0, 1, 2))
  if m < thr:
    return True
  else:
      return False

# def comp_frames(frames, thr: float=0.9995):
#     if print_similarity_measures(frames[0], frames[-1]) > thr:
#         return True
#     else:
#         return False

class MuscleUltrasoundDataset:


    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path  # /data/Group2-MUSCLE/CNS
        self.video_file_paths, self.annotation_file_names = self._read_path(dataset_path)

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

    def _normalize(self, frames):
        for i in range(frames.shape[0]):
            # Min Max normalization
            _min = np.amin(frames[i, :, :])
            frames[i, :, :] = frames[i, :, :] - _min
            _max = np.amax(frames[i, :, :]) + 1e-6
            frames[i, :, :] = frames[i, :, :] / _max

        frames = np.expand_dims(np.array(frames), axis=-1)
        return frames

    def _get_crop_params(self, frame):
        alpha = 0.05
        # Detect Top Down
        m = np.mean(frame[:, :, :], axis=(0, 2))

        m = m - np.min(m)
        m = m / np.max(m)
        top = 0
        down = len(m)
        for i in np.arange(int(len(m) / 2), 0, -1):
            if m[i] < alpha:
                top = i
                break
        for i in np.arange(int(len(m) / 2), len(m)):
            if m[i] < alpha:
                down = i
                break

        # Detect left right
        m = np.mean(frame[:, :, :], axis=(0, 1))
        m = m - np.min(m)
        m = m / np.max(m)
        left = 0
        right = len(m)
        for i in np.arange(int(len(m) / 2), 0, -1):
            if m[i] < alpha:
                left = i
                break
        for i in np.arange(int(len(m) / 2), len(m)):
            if m[i] < alpha:
                right = i
                break

        return [top, down, left, right]

    def _crop_frames(self, frames, params):
        [top, down, left, right] = params
        crop = frames[:, top:down, left:right]
        # plt.imshow(crop[int(len(crop) / 2)])
        # plt.show()
        crop = [cv.resize(crop[i, :, :], (64, 64), interpolation=Image.ANTIALIAS) for i in range(frames.shape[0])]
        return crop

    def _crop_frame(self, frame, params):
        [top, down, left, right] = params
        crop = frame[top:down, left:right]
        crop = cv.resize(crop, (64, 64), interpolation=Image.ANTIALIAS)
        return np.expand_dims(crop, axis=0)

    def _extract_frames(self, cap, annotation):
        # extract the frames of the clip
        row_frames = []
        cap.set(cv.CAP_PROP_POS_MSEC, self._time2msec(annotation['start_time']))

        crop_params = None
        while True:
            success, frame = cap.read()
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float64)  # grayscale the video here.

            if len(row_frames) < 100:
                row_frames.append(gray_frame)
            elif len(row_frames) == 100:
                row_frames.append(gray_frame)
                crop_params = self._get_crop_params(np.array(row_frames))
                if crop_params[0] == crop_params[1] or crop_params[2] == crop_params[3]:
                    return None
                frames = self._crop_frames(np.array(row_frames), crop_params)
            elif crop_params is not None:
                frames = np.concatenate((frames, self._crop_frame(gray_frame, crop_params)), axis=0)

            msec = cap.get(cv.CAP_PROP_POS_MSEC)
            if msec > self._time2msec(annotation['end_time']):
                break
            if not success:
                print(
                    '[ERROR] [MuscleUltrasoundDataset.__getitem__()] Unable to extract frame at ms {} from video '.format(
                        msec))
                break
        if len(frames) < 50:
            crop_params = self._get_crop_params(np.array(frames))
            if crop_params[0] == crop_params[1] or crop_params[2] == crop_params[3]:
                return None
            frames = self._crop_frames(np.array(frames), crop_params)

        frames_cropped = frames

        frames_normalized = self._normalize(frames_cropped)
        return frames_normalized

    def process(self, index: int):
        video_file_path = self.video_file_paths[index]
        features_folder_path = self.video_file_paths[index].replace(".mp4", "")
        if os.path.exists(features_folder_path):
            os.system('rm -r ' + features_folder_path)
        os.mkdir(features_folder_path)

        cap = cv.VideoCapture(video_file_path)
        if cap.isOpened() == False:
            print('[ERROR] [MuscleUltrasoundDataset.__getitem__()] Unable to read video ' + self.video_filenames[index])
            exit(-1)

        with open(self.annotation_file_names[index], "r") as json_file:
            json_data = json.load(json_file)


        for s, annotation in enumerate(json_data['annotations']):
            section_frames = self._extract_frames(cap, annotation)
            if section_frames is not None:
                data = []
                step = 5 #step of 15 is used for genertaing numpy data for training to have frames with length of 10 with gap o 5. For evaluation, step is changed to 10, then we do not have any gap between frames with length of 10.
                for i in range(0, section_frames.shape[0]-10, step):  # Sliding window with a len of 10 and step of 15
                    data.append(section_frames[i: i+10])

                p_data = []
                n_data = []
                state = 1
                for i in range(1, len(data)):
                    if state == 1:
                        if not comp_frames(data[-i]):
                            pass
                        else:
                            state = 2
                            p_data.append(data[-i])

                    if state == 2:
                        if comp_frames(data[-i]):
                            p_data.append(data[-i])
                        else:
                            state = 3

                    if state == 3:
                        n_data.append(data[-i])


                if float((len(p_data)+len(n_data)) / len(data)) < 0.5:
                    n_data = data
                    p_data = []

                np.save(features_folder_path + "/" + str(s) + "_n.npy", np.array(n_data))
                np.save(features_folder_path + "/" + str(s) + "_p.npy", np.array(p_data))
                np.save(features_folder_path + "/" + str(s) + ".npy", np.array(data))

                print("p:", len(p_data), "n:", len(n_data), "all:", len(data))



muscle = MuscleUltrasoundDataset(dataset_path='dataset path') #  path of data

for i in range(len(muscle.video_file_paths)):
    print(muscle.video_file_paths[i])
    muscle.process(i)

