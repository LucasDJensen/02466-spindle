import os
import sys

sys.path.append(os.path.join(sys.path[0], ".."))
import pickle
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

from globals import HPC_STORAGE_PATH
from spindle_data_loading import SequenceDataset2

embedding_layer_name = 'dense_1'  # Change this to the layer before the classification layer

# Define paths
data_path = os.path.join(HPC_STORAGE_PATH, 'preprocessed_spindle_data/spindle')
save_path = os.path.join(HPC_STORAGE_PATH, 'results_spindle_latent_space_extract_embeddings')
csv_path = os.path.join(data_path, '..', 'spindle_labels_all.csv')
model_name = 'spindle_model'

checkpoint_path = os.path.join(save_path, model_name)

BATCH_SIZE = 300
TRAINING_EPOCHS = 5
ARTIFACT_DETECTION = False  # This will produce only artifact/not artifact labels
JUST_NOT_ART_EPOCHS = True  # This will filter out the artifact epochs and keep only the non-artifacts. Can only be true if ARTIFACT_DETECTION=False.
LOSS_TYPE = 'weighted_ce'  # 'weighted_ce' or 'normal_ce'


class MulticlassWeightedCrossEntropy_2(tf.keras.losses.Loss):
    '''
    Same implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    Weights are calculated as w_i = n_samples / (n_classes * n_elements_class_i)
    '''

    def __init__(self, name="class_weighted_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = 4

    def call(self, y_true, y_pred):

        ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)

        for i in range(self.n_classes):
            num_class_i = tf.shape(tf.where(y_true[:, i] == 1))[0]
            # tf.print('num_class_i: ', num_class_i)

            if num_class_i > 0:
                w = tf.shape(y_true)[0] / (self.n_classes * num_class_i)
                w = tf.cast(w, dtype=tf.float32)
                # tf.print('weight: ', w)

                ce = tf.tensor_scatter_nd_update(ce, tf.where(y_true[:, i] == 1), w * ce[y_true[:, i] == 1])
                # tf.print('ce updated')

        return ce


class MulticlassF1Score(tf.keras.metrics.Metric):
    '''
    Implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html, with average='macro'
    https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    https://en.wikipedia.org/wiki/Confusion_matrix

    Is the average of the f1-score of each batch, where the f1-score of a batch is the average of the f1-score across each
    class in that batch.

    I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
    Code to test it:
        f1_batch_1 = sklearn.metrics.f1_score(y_true1, y_pred1, average='macro')
        f1_batch_2 = sklearn.metrics.f1_score(y_true2, y_pred2, average='macro')
        av_f1 = (f1_batch_1 + f1_batch_2)/2
        y_true1 = np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred1 = np.array(
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_true2 = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred2 = np.array(
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_pred1)
        y_true2 = tf.convert_to_tensor(y_true2)
        y_pred2 = tf.convert_to_tensor(y_pred2)
        MBA = MulticlassF1Score(n_classes=3)
        MBA.update_state(y_true1, y_pred1)
        print(MBA.result())
        MBA.update_state(y_true2, y_pred2)
        print(MBA.result())
    '''

    def __init__(self, name='multiclass_F1_score', **kwargs):
        super(MulticlassF1Score, self).__init__(name=name, **kwargs)
        self.n_classes = 4
        self.TP = tf.keras.metrics.TruePositives()
        self.FP = tf.keras.metrics.FalsePositives()
        self.FN = tf.keras.metrics.FalseNegatives()
        self.epoch_average_F1 = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=1), self.n_classes)

        batch_F1_sum = tf.convert_to_tensor(0, dtype=tf.float32)

        for i in range(self.n_classes):
            if tf.shape(tf.where(y_true[:, i] == 1))[0] > 0:
                self.TP.reset_state()
                self.FP.reset_state()
                self.FN.reset_state()

                self.TP.update_state(y_true[:, i] == 1, y_pred[:, i] == 1)
                self.FP.update_state(y_true[:, i] == 1, y_pred[:, i] == 1)
                self.FN.update_state(y_true[:, i] == 1, y_pred[:, i] == 1)

                class_f1 = 2 * self.TP.result() / (2 * self.TP.result() + self.FP.result() + self.FN.result())

                batch_F1_sum = batch_F1_sum + class_f1

        batch_average_F1 = batch_F1_sum / tf.math.count_nonzero(tf.reduce_sum(y_true, axis=0), dtype=tf.float32)

        self.epoch_average_F1.update_state(batch_average_F1)

    def result(self):
        return self.epoch_average_F1.result()

    def reset_state(self):
        self.epoch_average_F1.reset_state()


class MulticlassBalancedAccuracy(tf.keras.metrics.Metric):
    '''
    Gives the average of the recall over each batch, where the recall of each batch is the average of the recall of each
    class in that batch.

    Same implementation as in https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_classification.py#L1933

    I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
    Code to test it:
    y_true1 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    y_pred1 = np.array([0, 0, 1, 0, 1, 0, 1, 2, 0])
    y_true2 = np.array([0, 2, 0, 0, 0, 0, 1, 2, 2])
    y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1, 2, 0])
    av_bacc = (sklearn.metrics.balanced_accuracy_score(y_true1, y_pred1) + sklearn.metrics.balanced_accuracy_score(y_true2, y_pred2))/2
    print(av_bacc)

    y_true1 = pd.get_dummies(y_true1, dtype=int).to_numpy()
    y_pred1 = pd.get_dummies(y_pred1, dtype=int).to_numpy()
    y_true2 = pd.get_dummies(y_true2, dtype=int).to_numpy()
    y_pred2 = pd.get_dummies(y_pred2, dtype=int).to_numpy()

    y_true1 = tf.convert_to_tensor(y_true1)
    y_pred1 = tf.convert_to_tensor(y_pred1)
    y_true2 = tf.convert_to_tensor(y_true2)
    y_pred2 = tf.convert_to_tensor(y_pred2)

    MBA = MulticlassBalancedAccuracy(n_classes=3)
    MBA.update_state(y_true1,y_pred1)
    MBA.update_state(y_true2,y_pred2)
    print(MBA.result())
    '''

    def __init__(self, name='multiclass_balanced_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = 4
        self.TP = tf.keras.metrics.TruePositives()
        self.FN = tf.keras.metrics.FalseNegatives()
        self.epoch_average_recall = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):

        batch_recalls_sum = tf.convert_to_tensor(0, dtype=tf.float32)

        for i in range(self.n_classes):
            if tf.shape(tf.where(y_true[:, i] == 1))[0] > 0:
                self.TP.reset_state()
                self.FN.reset_state()

                self.TP.update_state(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])
                self.FN.update_state(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])

                class_recall = self.TP.result() / (self.TP.result() + self.FN.result())

                batch_recalls_sum = batch_recalls_sum + class_recall

                # tf.print('class_recall: ', class_recall)

        batch_average_recall = batch_recalls_sum / tf.math.count_nonzero(tf.reduce_sum(y_true, axis=0),
                                                                         dtype=tf.float32)

        self.epoch_average_recall.update_state(batch_average_recall)

    def result(self):
        return self.epoch_average_recall.result()

    def reset_state(self):
        self.epoch_average_recall.reset_state()


with custom_object_scope(
        {'MulticlassWeightedCrossEntropy_2': MulticlassWeightedCrossEntropy_2, 'MulticlassF1Score': MulticlassF1Score,
         'MulticlassBalancedAccuracy': MulticlassBalancedAccuracy}):
    # Load the trained model
    model = load_model(os.path.join(checkpoint_path, 'best_model_info.h5'))

# Print the model summary to identify the layer name before the classification layer
print(model.summary())

# Create the embedding model (adjust 'dense_2' to your actual layer name)
embedding_model = Model(inputs=model.input, outputs=model.get_layer(embedding_layer_name).output)
if ARTIFACT_DETECTION == False:
    JUST_ARTIFACT_LABELS = False
else:
    JUST_ARTIFACT_LABELS = True

# Define your test dataset sequence similar to the training sequence
validate_sequence = SequenceDataset2(data_folder=data_path,
                                     csv_path=csv_path,
                                     set='validate',
                                     batch_size=BATCH_SIZE,
                                     just_not_art_epochs=True,
                                     just_artifact_labels=False)


print(f'Length of validate_sequence: {len(validate_sequence)}')
print(f'Batch size: {validate_sequence.batch_size}')
print(f'Number of samples in Validation set: {len(validate_sequence) * validate_sequence.batch_size}')

# # Extract embeddings from the test dataset
# validate_embeddings = embedding_model.predict(validate_sequence)
#
# # Save embeddings for later use
# validate_embeddings_path = os.path.join(checkpoint_path, 'validate_embeddings.pkl')
# with open(validate_embeddings_path, 'wb') as f:
#     pickle.dump(validate_embeddings, f)

# Define your test dataset sequence similar to the training sequence
test_sequence = SequenceDataset2(data_folder=data_path,
                                 csv_path=csv_path,
                                 set='test',
                                 batch_size=BATCH_SIZE,
                                 just_not_art_epochs=False,
                                 just_artifact_labels=False)

print(f'Length of test_sequence: {len(test_sequence)}')
print(f'Batch size: {test_sequence.batch_size}')
print(f'Number of samples in test set: {len(test_sequence) * test_sequence.batch_size}')
# for batch in test_sequence:
#
#     data, label = batch # something
#     #TODO store data and label in a list
#     break

# Extract embeddings from the test dataset
test_embeddings = embedding_model.predict(test_sequence)

# Save embeddings for later use
test_embeddings_path = os.path.join(checkpoint_path, 'test_embeddings.pkl')
with open(test_embeddings_path, 'wb') as f:
    pickle.dump(test_embeddings, f)

print('Embeddings extracted and saved successfully.')
