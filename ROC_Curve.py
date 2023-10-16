from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, classification_report
from model.supervised_baseline import build_model
from model.simclr import get_encoder
from tensorflow import keras
import numpy as np
from utils.dnn_data_2 import DataGenerator_pn
import tensorflow as tf
from model.simclr import Simclr



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


def load_model():
    simclr_pretraining = Simclr(get_encoder())
    simclr_pretraining.compile(contrastive_optimizer=keras.optimizers.Adam(0.001))
    simclr_pretraining.load_weights('checkpoints/ss/model')

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
    model.load_weights('checkpoints/ss_eval/ss_eval')
    model.summary()
    return model

def make_data(dataset_paths):
    np.random.seed(7)
    dataset_paths = sorted(dataset_paths)
    np.random.shuffle(dataset_paths)
    train_portion = 1.0

    dg = DataGenerator_pn(dataset_paths=dataset_paths, batch_size=200, verbose=True)
    train_x1, train_x2 = dg.data_generation(dg.data[0][0: int(dg.data[0].shape[0]*train_portion)])
    train_l = dg.data[1][0: int(dg.data[0].shape[0]*train_portion)]
    validation_x1, validation_x2 = dg.data_generation(dg.data[0][int(dg.data[0].shape[0]*train_portion): ])
    validation_l = dg.data[1][int(dg.data[0].shape[0]*train_portion): ]
    print('print shapeeeeeeeeeeeee', train_x1.shape, train_l.shape)
    return train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l


paths = ['/home/localhk20/data/Group2-MUSCLE/eval-data']#, '/home/localhk20/data/Group2-MUSCLE/TETANUS']
train_x1, train_x2, train_l, validation_x1, validation_x2, validation_l = make_data(dataset_paths=paths)

###Supervised-baseline model###
baseline = build_model()
baseline.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss= keras.losses.CategoricalCrossentropy())
baseline.load_weights('checkpoints/supervised_baseline/model')
###############################

###Self-supervised evaluation###
ssl_eval = load_model()
x = np.concatenate((train_x1, train_x2), axis=0)
p1= ssl_eval.predict(np.expand_dims(x, axis=0))[:, 0]
################################

l1 = train_l[:, 0]
#p1 = baseline.predict((train_x1, train_x2))[:, 0]
l2 = train_l[:, 0]
p2 = []
for i in range(len(train_x1)):
    p2.append(print_similarity_measures(train_x1[i, 0, :, :, :], train_x2[i, 0, :, :, :]))


fpr2, tpr2, threshold2 = roc_curve(l1 , p1, pos_label=1)

print('tpr2',tpr2)
print('fpr2',fpr2)


# roc_auc1 = roc_auc_score(l1, p1)
roc_auc2 = roc_auc_score(l1, p1)
print('roccurve', roc_auc2)
l2 = np.array(l1)
p2 = np.array(p1)
print('l2',l1.shape)
print('p2', p1.shape)
p1 = (p1 >= 0.5).astype(int)


precision =precision_score(l1, p1)
recall =recall_score(l1, p1)
f1 =f1_score(l1, p1)
acc =accuracy_score(l1, p1)
print('acc', acc, 'f1', f1, 'recall', recall, 'precision', precision, 'AUC', roc_auc2 )#, 'reca', reca, 'f1', f1, 'acc', acc)
exit()
