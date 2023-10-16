__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"


###Implementation of self-supervised model###

import tensorflow as tf
from model.resnet18_v2 import ResNet18
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

temperature = 0.1
WEIGHT_DECAY = 0.0005


def get_encoder():
    base_model = ResNet18()
    base_model.trainable = True
    inputs = tf.keras.layers.Input((None, 64, 64, 1))
    x = base_model(inputs, training=True)
    ###Projection head with two dense layers###
    x = tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(l2=1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(l2=1e-4))(x)
    outputs = layers.BatchNormalization()(x)
    f = tf.keras.Model(inputs, outputs, name="encoder")
    return f



class Simclr(tf.keras.Model):
    def __init__(self, encoder):
        super(Simclr, self).__init__()
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.encoder = encoder
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(64,)),
                layers.Dense(64, activation="relu"),
                layers.Dense(64),
            ],
            name="projection_head",
        )
        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(64,)), layers.Dense(2)], name="linear_probe"
        )


    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [self.loss_tracker,
                self.contrastive_accuracy,
                self.probe_loss_tracker,
                self.probe_accuracy,
                ]


    def contrastive_loss(self, projections_1, projections_2):
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
                tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities))

        # The temperature-scaled similarities are used as logits for cross-entropy
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2


    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data[0]
        # Forward pass through the encoder
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            # Note that here we are enforcing the network to match the representations of two differently augmented batches of data.
            contrastive_loss = self.contrastive_loss(z1, z2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        # Monitor loss.
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return {"loss": self.contrastive_loss_tracker.result(), "acc": self.contrastive_accuracy.result()}


    def eval(self, data):
        # Unpack the data.
        ds_one, ds_two = data
        z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
        #p1, p2 = self.projection_head(z1), self.projection_head(z2)
        contrastive_loss = self.contrastive_loss(z1, z2)
        return contrastive_loss.numpy()


    def call(self, data):
        # Unpack the data.
        ds_one, ds_two = data
        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            #p1, p2 = self.projection_head(z1), self.projection_head(z2)

        return z1, z2

