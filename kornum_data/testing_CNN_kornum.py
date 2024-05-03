import os
import sys

sys.path.append(os.path.join(sys.path[0], ".."))

from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from metrics import *
from tools import *
# from hmm import *
from kornum_data_loading import SequenceDataset
from globals import HPC_STORAGE_PATH

save_path = os.path.join(HPC_STORAGE_PATH, 'results_spindle_latent_space')
model_name = 'kornum_model'

data_path = os.path.join(HPC_STORAGE_PATH, 'preprocessed_spindle_data/kornum')
csv_path = os.path.join(data_path, '..', 'kornum_labels_all.csv')

BATCH_SIZE = 300
ARTIFACT_DETECTION = False  # This will produce only artifact/not artifact labels
JUST_NOT_ART_EPOCHS = True  # This will filter out the artifact epochs and keep only the non-artifacts. Can only be true if ARTIFACT_DETECTION=False.
LOSS_TYPE = 'weighted_ce'  # 'weighted_ce' or 'normal_ce'

# -------------------------------------------------------------------------------------------------------------------------

if ARTIFACT_DETECTION == False:
    JUST_ARTIFACT_LABELS = False
    last_activation = 'softmax'
    if JUST_NOT_ART_EPOCHS == False:
        NCLASSES_MODEL = 4
        raise Exception(
            'Testing for JUST_NOT_ART_EPOCHS==False not implemented. compute_and_save_metrics_cnn1() needs to be adapted.')
    else:
        NCLASSES_MODEL = 3

    metrics_list = [tf.keras.metrics.CategoricalAccuracy(),
                    MulticlassF1Score(n_classes=NCLASSES_MODEL),
                    MulticlassBalancedAccuracy(n_classes=NCLASSES_MODEL)]

    if LOSS_TYPE == 'weighted_ce':
        loss = MulticlassWeightedCrossEntropy_2(n_classes=NCLASSES_MODEL)
    elif LOSS_TYPE == 'normal_ce':
        loss = tf.keras.losses.CategoricalCrossentropy()

else:
    if JUST_NOT_ART_EPOCHS == True: raise Exception('If ARTIFACT_DETECTION=True, JUST_NOT_ART_EPOCHS must be False')
    JUST_ARTIFACT_LABELS = True
    last_activation = 'sigmoid'
    NCLASSES_MODEL = 1

    metrics_list = [tf.keras.metrics.BinaryAccuracy(),
                    BinaryBalancedAccuracy(),
                    BinaryF1Score()]

    if LOSS_TYPE == 'weighted_ce':
        loss = BinaryWeightedCrossEntropy()
    elif LOSS_TYPE == 'normal_ce':
        raise Exception("Not implemented")

print("Devices available: ", tf.config.list_physical_devices())

# -------------------------------------------------------------------------------------------------------------------------

test_sequence = SequenceDataset(data_folder=data_path,
                                csv_path=csv_path,
                                set='test',
                                batch_size=BATCH_SIZE,
                                just_not_art_epochs=JUST_NOT_ART_EPOCHS,
                                just_artifact_labels=JUST_ARTIFACT_LABELS)

# test_sequence_arts = SequenceDataset(data_folder=data_path,
#                                  csv_path=csv_path,
#                                  set='test',
#                                  batch_size=BATCH_SIZE*4,
#                                  just_not_art_epochs=True,
#                                  just_artifact_labels=True)

# -------------------------------------------------------------------------------------------------------------------------

spindle_model = tf.keras.Sequential([
    Input((160, 48, 3)),
    MaxPool2D(pool_size=(2, 3), strides=(2, 3)),
    Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=NCLASSES_MODEL, activation=last_activation, kernel_initializer='glorot_uniform')
])

spindle_model.load_weights(os.path.join(save_path, model_name, 'best_model.h5'))

spindle_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5 * 1e-5,
                                                                beta_1=0.9,
                                                                beta_2=0.999),
                      loss=loss,
                      metrics=metrics_list)

for i in range(len(test_sequence)):
    x_batch, y_batch = test_sequence.__getitem__(i)
    # _, y_batch_arts = test_sequence_arts.__getitem__(i)    

    if i == 0:
        cnn_probs = spindle_model(x_batch).numpy()
        y_true = y_batch
        # y_true_arts = y_batch_arts
    else:
        cnn_probs = np.concatenate([cnn_probs, spindle_model(x_batch).numpy()], axis=0)
        y_true = np.concatenate([y_true, y_batch], axis=0)
        # y_true_arts = np.concatenate([y_true_arts, y_batch_arts], axis=0)

if ARTIFACT_DETECTION == False:
    if JUST_NOT_ART_EPOCHS == True:
        y_true = np.argmax(y_true, axis=1)
        y_cnn = np.argmax(cnn_probs, axis=1)
    else:
        raise Exception('Not implemented')

if ARTIFACT_DETECTION == False:
    compute_and_save_metrics_cnn1(y_true, y_cnn, os.path.join(save_path, model_name), model_name)
else:
    compute_and_save_metrics_cnn2(y_true, cnn_probs, 0.5, os.path.join(save_path, model_name), model_name)
