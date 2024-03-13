import sys
sys.path.append("..")
import tensorflow as tf
import sklearn.metrics
# import random
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from kornum_data_loading import SequenceDataset
from metrics import *
# from tools import *
import pickle
import os

# plt.ion()

save_path = os.path.join(HPC_STORAGE_PATH, 'results_spindle')
model_name = 'A_3'

data_path = os.path.join(HPC_STORAGE_PATH, 'preprocessed_spindle_data/kornum')
csv_path = os.path.dirname(data_path) + '/labels_all.csv'

BATCH_SIZE = 300
TRAINING_EPOCHS = 1
ARTIFACT_DETECTION = False # This will produce only artifact/not artifact labels
JUST_NOT_ART_EPOCHS = True # This will filter out the artifact epochs and keep only the non-artifacts. Can only be true if ARTIFACT_DETECTION=False.
LOSS_TYPE = 'weighted_ce' # 'weighted_ce' or 'normal_ce'


# -------------------------------------------------------------------------------------------------------------------------

if ARTIFACT_DETECTION==False:
    JUST_ARTIFACT_LABELS = False 
    last_activation = 'softmax'
    if JUST_NOT_ART_EPOCHS==False:
        NCLASSES_MODEL = 4
    else:
        NCLASSES_MODEL = 3

    metrics_list = [tf.keras.metrics.CategoricalAccuracy(),
                   MulticlassF1Score(n_classes=NCLASSES_MODEL),
                   MulticlassBalancedAccuracy(n_classes=NCLASSES_MODEL)]
    
    if LOSS_TYPE=='weighted_ce':
        loss=MulticlassWeightedCrossEntropy_2(n_classes=NCLASSES_MODEL)
    elif LOSS_TYPE=='normal_ce':
        loss=tf.keras.losses.CategoricalCrossentropy()

else:
    if JUST_NOT_ART_EPOCHS==True: raise Exception('If ARTIFACT_DETECTION=True, JUST_NOT_ART_EPOCHS must be False')
    JUST_ARTIFACT_LABELS = True
    last_activation = 'sigmoid' 
    NCLASSES_MODEL = 1

    metrics_list=[tf.keras.metrics.BinaryAccuracy(),
                BinaryBalancedAccuracy(),
                BinaryF1Score()]
    
    if LOSS_TYPE=='weighted_ce':
        loss=BinaryWeightedCrossEntropy()
    elif LOSS_TYPE=='normal_ce':
        raise Exception("Not implemented")

print("Devices available: ", tf.config.list_physical_devices())


# -------------------------------------------------------------------------------------------------------------------------

train_sequence = SequenceDataset(data_folder=data_path,
                                 csv_path=csv_path,
                                 set='train',
                                 batch_size=BATCH_SIZE,
                                 just_not_art_epochs=JUST_NOT_ART_EPOCHS,
                                 just_artifact_labels=JUST_ARTIFACT_LABELS)

val_sequence = SequenceDataset(data_folder=data_path,
                               csv_path=csv_path,
                               set='validation',
                               batch_size=BATCH_SIZE,
                               just_not_art_epochs=JUST_NOT_ART_EPOCHS,
                               just_artifact_labels=JUST_ARTIFACT_LABELS)

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

checkpoint_path = os.path.join(save_path, model_name)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
checkpoint_callback = MyCustomCallback(validation_dataset=val_sequence,
                                       save_checkpoint_path=checkpoint_path,
                                       evaluation_rate=int(len(train_sequence)/10),
                                       improvement_threshold=0.001,
                                       early_stopping_thr=10)

spindle_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5 * 1e-5,
                                                                beta_1=0.9,
                                                                beta_2=0.999),
                      loss=loss,
                      metrics=metrics_list)
                    #   run_eagerly=True)

history = spindle_model.fit(
    x=train_sequence,
    epochs=TRAINING_EPOCHS,
    verbose=1,
    callbacks=[checkpoint_callback])

print('End of training reached')

with open(os.path.join(save_path, model_name, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

print('History saving reached')