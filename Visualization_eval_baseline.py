__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

#This script is used for model evaluation, It receives two videos and calculate the similarity between them.


import os
from model.supervised_baseline import build_model
from model.simclr import get_encoder
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def random_crop(matrix: np.array, crop_shape: tuple):
    if len(matrix.shape) != 2:
        raise Exception

    x, y = matrix.shape

    x_left = np.random.randint(low=0, high=x-crop_shape[0])
    x_right = x_left + crop_shape[0]

    y_left = np.random.randint(low=0, high=y - crop_shape[1])
    y_right = y_left + crop_shape[1]

    matrix[x_left:x_right, y_left:y_right] = 0.0
    return matrix

def load_data(path: str):
    svs = []
    svs_name = []
    for sv_file_name in os.listdir(path):
        sv = []
        if sv_file_name.endswith('.npy'):
            sv_x = np.load(os.path.join(path, sv_file_name))
            svs_name.append(sv_file_name.replace(".npy", ""))
            if np.any(np.isnan(sv_x)):
                print("Found None in", sv_file_name)
                raise
            for section in sv_x:
                sv.append(section[0:5])
                sv.append(section[5:])
            svs.append(np.array(sv))
    return np.array(svs), svs_name

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
    return nc

def comp_frames(frames, thr: float):
  var = np.var(frames, axis=0)  #
  m = np.mean(var, axis=(0, 1, 2))
  if m < thr:
    return True
  else:
      return False

def get_gt(data_path: str):
    frames = np.load(data_path)
    for i in range(len(frames), 0, -1):
        if comp_frames(frames[i-5: i], thr
        =0.000001):
            return np.expand_dims(frames[i-5: i], axis=0), i-5
    return None

############################################################

######Testing videos#################


gt_frames, index = get_gt("/home/localhk20/data/Group2-MUSCLE/eval-data/01NVb-003-202/T1/01NVb-003-202-1_gt/1.npy") # each time, every numpy file should be compared with the whole T2
x1_test1, sv_file_name = load_data("/home/localhk20/data/Group2-MUSCLE/eval-data/01NVb-003-202/T2/01NVb-003-202-2")

print(index)

# Compile model and start training.
baseline = build_model()
baseline.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss= keras.losses.CategoricalCrossentropy())
baseline.load_weights('checkpoints/supervised_baseline/model')

################

print('======>', x1_test1.shape, x1_test1[0].shape)


def plot(sv_file_name):
    for sc, s in enumerate(max_frames): # sc >>> s counter
        fig, axs = plt.subplots(2, 5, figsize=(10, 6), constrained_layout=True)
        fig.suptitle('Positive pair')
        for i in range(5):
            axs[0, i].imshow(gt_frames[0][i][:][:])

        for i in range(5):
            axs[1, i].imshow(s[i][:][:])
        plt.savefig("sv-{0}-positive-{1:03d}.png".format(sv_file_name, sc))
        plt.show()

    for sc, s in enumerate(min_frames):
        fig, axs = plt.subplots(2, 5, figsize=(10, 6), constrained_layout=True)
        fig.suptitle('Negative pair')
        for i in range(5):
            axs[0, i].imshow(gt_frames[0][i][:][:])

        for i in range(5):
            axs[1, i].imshow(s[i][:][:])
        plt.savefig("sv-{0}-negative-{1:03d}.png".format(sv_file_name, sc))
        plt.show()



def plot_unet(sv_file_name):
    for sc, s in enumerate(max_frames): # sc >>> s counter
        im1 = gt_frames[0][0][:][:]
        im2 = s[0][:][:]

        fig, ax1 = plt.subplots()
        fig, ax2 = plt.subplots()
        ax1.axis('off')
        ax2.axis('off')

        ax1.imshow(im1, cmap='gray')
        ax2.imshow(im2, cmap='gray')

        ax1.figure.savefig("sv-{0}-positive-gt.png".format(sv_file_name,sc))
        ax2.figure.savefig("sv-{0}-positive-prd.png".format(sv_file_name,sc))

        plt.close()





results = []
for i, sv in enumerate(x1_test1):
    result = []

    min = 10000000000000
    max = -10000000000000
    min_frames = []
    max_frames = []

    for s in sv:
        result.append(print_similarity_measures(gt_frames[0], s)) #This line uncommented for similarity measure using cross correlation
        try:
            result.append(baseline.predict((gt_frames, np.expand_dims(s, axis=0)))[0][0])
        except:
            print(gt_frames.shape, np.expand_dims(s, axis=0).shape)
        if result[-1] < min:
            min = result[-1]
            min_frames = [s]

        if result[-1] > max:
            max = result[-1]
            max_frames = [s]

    plot_unet(sv_file_name[i])
    results.append(result)


### Here positive pairs and negative pairs are visualized along with ground truth. ###
### Here probablity of being positive for all the standard views is visulaized. ###

fig, ax = plt.subplots()
for sv, result in enumerate(results):
    ax.plot(result, label="SV-"+str(sv))

ax.legend()
plt.show()
