__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

#This piece of code is used to disable GPU temporary and run the code from CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#This piece of code is used to disable GPU temporary and run the code from CPU


### This script is used for supervised baseline model training. ###


from model.supervised_baseline import Supervised_baseline, build_model, get_encoder
from tensorflow import keras
from utils.dnn_data_2 import DataGenerator_pn
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def flip(matrix: np.array):
    return np.flip(matrix, axis=3)


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

def make_data(dataset_paths):
    np.random.seed(7)
    dataset_paths = sorted(dataset_paths)
    np.random.shuffle(dataset_paths)
    train_portion = 0.8

    dg = DataGenerator_pn(dataset_paths=dataset_paths, batch_size=200, verbose=True)
    train_x1, train_x2 = dg.data_generation(dg.data[0][0: int(dg.data[0].shape[0]*train_portion)])
    train_l = dg.data[1][0: int(dg.data[0].shape[0]*train_portion)]
    validation_x1, validation_x2 = dg.data_generation(dg.data[0][int(dg.data[0].shape[0]*train_portion): ])
    validation_l = dg.data[1][int(dg.data[0].shape[0]*train_portion): ]
    print('print shapeeeeeeeeeeeee', train_x1.shape, train_l.shape)
    return train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l


paths = ['/home/localhk20/data/Group2-MUSCLE/CNS', '/home/localhk20/data/Group2-MUSCLE/TETANUS']
train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l = make_data(dataset_paths=paths)


###### This section is used for visualizing negative and positive pairs #######
# ind = list(range(validation_x1.shape[0]))
# random.shuffle(ind)
# for i in ind:
#     fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
#     axs[0].imshow(train_x1[i, 0, :, :, 0])
#     axs[1].imshow(train_x2[i, 0, :, :, 0])
#     fig.suptitle(str(validation_l[i]))
#     plt.show()
###############################################################################


train_x1_flip = flip(train_x1)
train_x2_flip = flip(train_x2)
train_x1_crop = np.array([random_crop(f, (10, 10)) for f in train_x1])
train_x2_crop = np.array([random_crop(f, (10, 10)) for f in train_x2])

train_x1_augment = np.concatenate((train_x1, train_x1_crop, train_x1_flip), axis=0)
train_x2_augment = np.concatenate((train_x2, train_x2_crop, train_x2_flip), axis=0)
train_l_augment = np.concatenate((train_l, train_l, train_l), axis=0)



# Compile model and start training.
model = build_model()
model.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.00001), metrics=keras.metrics.CategoricalAccuracy()) #Adam optimizer


# Define the Keras TensorBoard callback.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)



print("positive:", np.sum(train_l[:, 0]))
print("negative:", np.sum(train_l[:, 1]))


history = model.fit((train_x1_augment, train_x2_augment), train_l_augment,
          batch_size=40,
          epochs=50,
          validation_data=((validation_x1, validation_x2), validation_l),
          workers=8)

print("Saving model:")
model.save_weights('checkpoints/model')


#Visualize the training progress of the model.
plt.plot(history.history["loss"], 'r', label='Training loss')
plt.plot(history.history["val_loss"], 'b', label='Validation loss')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.savefig('train-validation-loss-baseline-plot.png')
plt.figure()

plt.plot(history.history["categorical_accuracy"], 'r', label='Training accuracy')
plt.plot(history.history["val_categorical_accuracy"], 'b', label='Validation accuracy')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.savefig('train-validation-acc-baseline-plot.png')







