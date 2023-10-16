__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

#This script is used for self-supervised model (Simclr) evaluation, It adds some dense layers on top of simclr model (Training false) and does a linear classification.


from model.simclr import Simclr
from model.simclr import get_encoder
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils.dnn_data_2 import DataGenerator_pn
from datetime import datetime
import argparse

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
    print('print shape', train_x1.shape, train_l.shape)
    return train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l


# Compile model and start training.
def make_model(model_path: str):
    simclr_pretraining = Simclr(get_encoder())
    simclr_pretraining.compile(contrastive_optimizer=keras.optimizers.Adam(0.001))
    simclr_pretraining.load_weights(model_path)

    # Extract the backbone ResNet18.
    backbone = tf.keras.Model(
        simclr_pretraining.encoder.input, simclr_pretraining.encoder.output
    )

    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = tf.keras.layers.Input((None, 64, 64, 1))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.Dense(2024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x) #prevent overfitting
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x) #prevent overfitting
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x) #prevent overfitting
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="ssl_model")

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=keras.metrics.CategoricalAccuracy()) #Adam optimizer
    model.summary()
    return model


def data_prep(paths: list):
    train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l = make_data(dataset_paths=paths)

    train_x1_flip = flip(train_x1)
    train_x2_flip = flip(train_x2)
    train_x1_crop = np.array([random_crop(f, (10, 10)) for f in train_x1])
    train_x2_crop = np.array([random_crop(f, (10, 10)) for f in train_x2])

    train_x1_augment = np.concatenate((train_x1, train_x1_crop, train_x1_flip), axis=0)
    train_x2_augment = np.concatenate((train_x2, train_x2_crop, train_x2_flip), axis=0)
    train_l_augment = np.concatenate((train_l, train_l, train_l), axis=0)


    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    print("positive:", np.sum(train_l[:, 0]))
    print("negative:", np.sum(train_l[:, 1]))


    x = np.concatenate((train_x1_augment, train_x2_augment), axis=1)
    print('x', x.shape)
    v = np.concatenate((validation_x1, validation_x2), axis=1)
    return x, train_l_augment, v, validation_l


def train(model, x_train, y_train, x_val, y_val, out_path: str):
    history = model.fit(x_train, y_train,
              batch_size=42,
              epochs=50,
              validation_data=(x_val, y_val),
              workers=8)

    print("Saving model:")
    model.save_weights(out_path)
    return history


def main():
    model = make_model(model_path='checkpoints/ss/model')
    x_train, y_train, x_val, y_val = data_prep(paths=['/home/localhk20/data/Group2-MUSCLE/eval-data'])
    history = train(model, x_train, y_train, x_val, y_val, out_path="checkpoints/ss_eval")

    #Visualize the training progress of the model.
    plt.plot(history.history["loss"], 'r', label='Training loss')
    plt.plot(history.history["val_loss"], 'b', label='Validation loss')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.savefig('train-validation-loss-ssleval-plot.png')
    plt.figure()

    plt.plot(history.history["categorical_accuracy"], 'r', label='Training accuracy')
    plt.plot(history.history["val_categorical_accuracy"], 'b', label='Validation accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.savefig('train-validation-acc-ssleval-plot.png')

#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()




