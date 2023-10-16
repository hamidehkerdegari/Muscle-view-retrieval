__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"


###This code is used for contrastive learning model training###

from model.simclr import Simclr
from model.simclr import get_encoder
from tensorflow import keras
from utils.dnn_data_2 import DataGenerator
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def make_data(dataset_paths):
    np.random.seed(5)
    dataset_paths = sorted(dataset_paths)
    np.random.shuffle(dataset_paths)
    train_portion = 0.8

    dg = DataGenerator(dataset_paths=dataset_paths, batch_size=200, verbose=True)
    train_x1, train_x2 = dg.data_generation(dg.data[0: int(dg.data.shape[0]/train_portion)])
    validation_x1, validation_x2 = dg.data_generation(dg.data[int(dg.data.shape[0]/train_portion): ])

    return train_x1, train_x2, validation_x1, validation_x2


### Dataset path ###
paths = ['/home/localhk20/data/Group2-MUSCLE/CNS', '/home/localhk20/data/Group2-MUSCLE/TETANUS']
train_x1, train_x2, validation_x1, validation_x2 = make_data(dataset_paths=paths)

### Compile model and start training. ###
simclr_pretraining = Simclr(get_encoder())
simclr_pretraining.compile(contrastive_optimizer=keras.optimizers.Adam(0.00001)) #Adam optimizer


### Define the TensorBoard callback. ###
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = simclr_pretraining.fit((train_x1, train_x2),
                          batch_size=40,
                          validation_data=((validation_x1, validation_x2), None),
                          epochs=500,
                          workers=8, callbacks=[tensorboard_callback])
print("Saving model:")
simclr_pretraining.save_weights('checkpoints/model')


### Visualize the training progress of the model. ###
plt.plot(history.history["loss"], 'r', label='Training loss')
plt.ylabel("Similairty Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.savefig('train-loss-plot.png')

plt.plot(history.history["loss"], 'b', label='Validation loss')
plt.ylabel("Similairty Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.savefig('validation-loss-plot.png')







