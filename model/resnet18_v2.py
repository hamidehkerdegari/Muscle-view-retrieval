__author__ = "Hamideh Kerdegari"
__copyright__ = "Copyright 2023"
__credits__ = ["Hamideh Kerdegari"]
__license__ = "Hamideh Kerdegari"
__maintainer__ = "Hamideh Kerdegari"
__email__ = "hamideh.kerdegari@gmail.com"
__status__ = "R&D"

###Implementation of pretrained encoder architecture###

from keras.layers import Dense, Conv3D, Conv2D,  MaxPool3D, MaxPool2D, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from keras.models import Model
import keras
import tensorflow as tf



class ResNet18(Model):
    def __init__(self, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv3d_1 = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=tf.nn.relu, data_format="channels_last")
        self.init_bn_1 = BatchNormalization()
        self.conv3d_2 = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=tf.nn.relu, data_format="channels_last")
        self.init_bn_2 = BatchNormalization()
        self.pool3d_1 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding="same")

        self.conv3d_3 = Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=tf.nn.relu, data_format="channels_last")
        self.init_bn_3 = BatchNormalization()
        self.conv3d_4 = Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=tf.nn.relu, data_format="channels_last")
        self.init_bn_4 = BatchNormalization()
        self.pool3d_2 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding="same")
        self.avg_pool2d = keras.layers.TimeDistributed(GlobalAveragePooling2D())  # generates one vector (size of 512) for each frame
        self.avg_pool1d = GlobalAveragePooling1D()    # generates one vector (size of 512)
        self.fc = Dense(256, activation="relu")

    def call(self, inputs):
        out = self.conv3d_1(inputs)
        out = self.init_bn_1(out)
        out = self.conv3d_2(out)
        out = self.init_bn_2(out)
        out = self.pool3d_1(out)
        out = self.conv3d_3(out)
        out = self.init_bn_3(out)
        out = self.conv3d_4(out)
        out = self.init_bn_4(out)
        out = self.pool3d_2(out)
        out = self.avg_pool2d(out)
        out = self.avg_pool1d(out)
        out = self.fc(out)
        return out


##Checking encoder architecture###
# model = ResNet18()
# model.build(input_shape= (20, 5, 128,128,1))
# print(model.summary())
#
