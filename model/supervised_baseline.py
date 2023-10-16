__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"


### Implementation of supervised baseline model ###

import tensorflow as tf
from model.resnet18_v2 import ResNet18
from tensorflow import keras
from tensorflow.keras import layers
from utils.dnn_data_2 import DataGenerator_pn
import numpy as np
from tensorflow.keras import regularizers


def make_data(dataset_paths):
    np.random.seed(5)
    dataset_paths = sorted(dataset_paths)
    np.random.shuffle(dataset_paths)
    train_portion = 0.8

    dg = DataGenerator_pn(dataset_paths=dataset_paths, batch_size=200, verbose=True)
    train_x1, train_x2 = dg.data_generation(dg.data[0][0: int(dg.data[0].shape[0]/train_portion)])
    train_l = dg.data[1][0: int(dg.data[0].shape[0]/train_portion)]
    validation_x1, validation_x2 = dg.data_generation(dg.data[0][int(dg.data[0].shape[0]/train_portion): ])
    validation_l = dg.data[1][int(dg.data[0].shape[0]/train_portion): ]
    return train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l


### Dataset path ###
paths = ['dataset path']
train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l = make_data(dataset_paths=paths)


def get_encoder():
    base_model = ResNet18()
    base_model.trainable = True
    inputs = tf.keras.layers.Input((None, 64, 64, 1))
    x = base_model(inputs, training=True)
    f = tf.keras.Model(inputs, x)
    return f

class Supervised_baseline(tf.keras.Model):
    def __init__(self):
        super(Supervised_baseline, self).__init__()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.encoder = get_encoder()
        self.encoder.summary()


    @property
    def metrics(self):
        return [self.classification_loss_tracker, self.classification_accuracy]


    def compile(self, classification_optimizer, **kwargs):
        super().compile(**kwargs)

        self.classification_optimizer = classification_optimizer

        # self.classification_loss will be defined as a method
        self.classification_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.classification_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.classification_accuracy = keras.metrics.SparseCategoricalAccuracy(name="classification_acc")


    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data[0]
        labels = data[1]
        # Forward pass through the encoder
        with tf.GradientTape() as tape:
            z1 = self.encoder(ds_one)
            z2 = self.encoder(ds_two)
            avg = tf.keras.layers.Average()([z1, z2])
            out = tf.keras.layers.Dense(2, activation='softmax')(avg)


            # Note that here we are enforcing the network to match the representations of two differently augmented batches of data.
            classification_loss = self.classification_loss(labels, out)
        gradients = tape.gradient(
            classification_loss,
            self.encoder.trainable_weights
        )
        self.classification_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights
            )
        )
        # Monitor loss.
        self.classification_loss_tracker.update_state(classification_loss)
        return {"loss": self.classification_loss_tracker.result(), "acc": self.classification_accuracy.result()}


    def eval(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
        avg = tf.keras.layers.Average()([z1, z2])
        out = tf.keras.layers.Dense(2, activation='softmax')(avg)


        classification_loss = self.classification_loss(out)
        return classification_loss.numpy()


    def call(self, data):
        # Unpack the data.
        ds_one, ds_two = data
        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            avg = tf.keras.layers.Average()([z1, z2])
            out = tf.keras.layers.Dense(2, activation='softmax')(avg)

        return out


def build_model():
    input_0 = tf.keras.layers.Input(shape=(None, 64, 64, 1), name='in_0')
    input_1 = tf.keras.layers.Input(shape=(None, 64, 64, 1), name='in_1')

    base_model = ResNet18()
    base_model.trainable = True
    x_0 = base_model(input_0, training=True)
    x_1 = base_model(input_1, training=True)

    x_0 = tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(l2=1e-4))(x_0)
    x_0 = layers.BatchNormalization()(x_0)
    x_0 = layers.ReLU()(x_0)
    x_0 = tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(l2=1e-4))(x_0)
    outputs_0 = layers.BatchNormalization()(x_0)

    x_1 = tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(l2=1e-4))(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.ReLU()(x_1)
    x_1 = tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(l2=1e-4))(x_1)
    outputs_1 = layers.BatchNormalization()(x_1)


    concat = tf.keras.layers.concatenate(([outputs_0, outputs_1]))
    out = keras.layers.Dense(2, activation=tf.nn.relu, name='out_likelihood')(concat)
    out = keras.layers.Softmax(name='out_softmax')(out)
    model = tf.keras.models.Model(inputs=[input_0, input_1], outputs=out)
    model.summary()
    return model


