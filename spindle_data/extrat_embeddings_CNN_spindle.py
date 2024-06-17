import os
import sys

sys.path.append(os.path.join(sys.path[0], ".."))
import pickle

from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from metrics import *
from tools import *
# from hmm import *
from spindle_data_loading import SequenceDataset2
from globals import HPC_STORAGE_PATH

embedding_layer_name = 'dense_1'

save_path = os.path.join(HPC_STORAGE_PATH, 'results_spindle_latent_space_extract_embeddings')
model_name = 'spindle_model'
embeddings_path = os.path.join(save_path, model_name, 'embeddings')
os.makedirs(embeddings_path, exist_ok=True)

data_path = os.path.join(HPC_STORAGE_PATH, 'preprocessed_spindle_data/spindle')
csv_path = os.path.join(data_path, '..', 'spindle_labels_all.csv')

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

validation_sequence = SequenceDataset2(data_folder=data_path,
                                 csv_path=csv_path,
                                 set='validation',
                                 batch_size=BATCH_SIZE,
                                 just_not_art_epochs=True,
                                 just_artifact_labels=False)

print('-------------------- Validation set --------------------')
print(f'Sequence length: {len(validation_sequence)}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Number of samples: {len(validation_sequence) * BATCH_SIZE}')

test_sequence = SequenceDataset2(data_folder=data_path,
                                csv_path=csv_path,
                                set='test',
                                batch_size=BATCH_SIZE,
                                just_not_art_epochs=False,
                                just_artifact_labels=False)

print('-------------------- Test set --------------------')
print(f'Sequence length: {len(test_sequence)}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Number of samples: {len(test_sequence) * BATCH_SIZE}')

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

print(spindle_model.summary())
embedding_model = Model(inputs=spindle_model.input, outputs=spindle_model.get_layer(embedding_layer_name).output)
print(f'Embedding model: {embedding_model.summary()}')

num_samples = len(test_sequence) * test_sequence.batch_size

# Get the shape of one prediction to determine the dimensions
x_batch, y_batch = test_sequence.__getitem__(0)
print(f'x_batch shape: {x_batch.shape}')
print(f'y_batch shape: {y_batch.shape}')

# Preallocate arrays for predictions and true labels
y_true = np.zeros((num_samples, y_batch.shape[1]))

start_idx = 0
for i in range(len(test_sequence)):
    x_batch, y_batch = test_sequence.__getitem__(i)
    batch_size = x_batch.shape[0]

    # Insert the results into the preallocated arrays
    y_true[start_idx:start_idx + batch_size] = y_batch

    start_idx += batch_size

# save the true labels
np.save(os.path.join(embeddings_path, 'true_labels.npy'), y_true)
print(f'y_true shape: {y_true.shape}')
print('True labels saved')


# Extract embeddings from the test dataset
validation_embeddings = embedding_model.predict(validation_sequence)

# Save embeddings for later use
validation_embeddings_path = os.path.join(embeddings_path, 'validation_embeddings.pkl')
with open(validation_embeddings_path, 'wb') as f:
    pickle.dump(validation_embeddings, f)

print('Validation embeddings saved')


# Extract embeddings from the test dataset
test_embeddings = embedding_model.predict(test_sequence)

# Save embeddings for later use
test_embeddings_path = os.path.join(embeddings_path, 'test_embeddings.pkl')
with open(test_embeddings_path, 'wb') as f:
    pickle.dump(test_embeddings, f)

print('Test embeddings saved')