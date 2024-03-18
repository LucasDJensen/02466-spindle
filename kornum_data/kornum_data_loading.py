import os
import sys
import pandas as pd
import numpy as np
import os.path
import mne
import pathlib
import tensorflow as tf
from final_preprocessing import preprocess_EEG, preprocess_EMG
from spindle_data.spindle_data_loading import windowing, load_raw_recording
from globals import HPC_STORAGE_PATH


def get_tsv_filepaths(folder_path):
    tsv_files = []
    for subfolder in [s for s in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, s))]:
        l = [os.path.join(folder_path, subfolder, 'tsv', f) for f in
             os.listdir(os.path.join(folder_path, subfolder, 'tsv')) if not f.startswith('.')]
        tsv_files = tsv_files + l

    return tsv_files


def get_epoch_counts(tsv_files):
    df = pd.DataFrame()

    for filepath in tsv_files:
        tsv_data = pd.read_csv(filepath, skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]

        value_counts = tsv_data.value_counts()

        file_df = pd.DataFrame(value_counts.values.reshape((1, len(value_counts))),
                               columns=[str(i) for i in value_counts.index])
        file_df.insert(0, 'n_epochs', tsv_data.shape[0])
        file_df.insert(0, 'File', os.path.join(*filepath.split(os.sep)[-3:]))

        edf_path = os.path.join(pathlib.Path.home().drive, os.sep, *filepath.split(os.sep)[:-2], 'EDF',
                                filepath.split(os.sep)[-1][:-3] + 'edf')
        sfreq = mne.io.read_raw_edf(edf_path).info['sfreq']
        file_df.insert(1, 'sampling_freq (Hz.)', np.round(sfreq, 2))

        df = pd.concat([df, file_df], axis=0, ignore_index=True)

        df.fillna(0, inplace=True)

    return df


def load_labels(labels_path,
                just_artifact_labels,
                just_stage_labels):
    df = pd.read_csv(labels_path, skiprows=9, engine='python', sep='\t', index_col=False).iloc[:, 4]
    df.loc[(df != 1) & (df != 2) & (df != 3)] = 4
    df = df.iloc[2:-2].copy()  # Drop 2 first and 2 last epochs

    labels = pd.get_dummies(df, dtype=int)
    if labels.shape[
        1] == 4:  # Assing name and reorder columns. If shape[1]<4 is because there isn't any artifact, so the artifact column is added 'artificially'
        labels.columns = ['WAKE', 'NREM', 'REM', 'Art']  # Assign name to columns
        labels = labels[['NREM', 'REM', 'WAKE', 'Art']]  # Reorder columns to have same order as in spindle data
    elif labels.shape[1] < 4:
        labels.columns = ['WAKE', 'NREM', 'REM']  # Assign name to columns
        labels = labels[['NREM', 'REM', 'WAKE']]  # Reorder columns to have same order as in spindle data
        labels.insert(3, "Art", 0)

    if just_artifact_labels == True:
        labels = labels['Art']

    elif just_stage_labels == True:
        labels = labels[['NREM', 'REM', 'WAKE']]

    return labels


def load_recording(signal_path,
                   labels_path,
                   resample_rate,
                   just_artifact_labels,
                   just_stage_labels,
                   validation_split):
    raw_data = load_raw_recording(signal_path, resample_rate)
    eeg_1 = preprocess_EEG(raw_data[0, :])
    eeg_2 = preprocess_EEG(raw_data[1, :])
    emg = preprocess_EMG(raw_data[2, :])
    x = np.stack((eeg_1, eeg_2, emg))
    x = windowing(x, window_size=32 * 5, window_stride=32)
    x = np.transpose(x, (0, 3, 2, 1))

    y = load_labels(labels_path,
                    just_artifact_labels=just_artifact_labels,
                    just_stage_labels=just_stage_labels)
    y = y.to_numpy()

    if x.shape[0] != y.shape[0]:
        print(signal_path)
        raise Exception("x and y don't have same number of epochs")

    if validation_split != 0:
        rdm_indexes = np.arange(x.shape[0])
        np.random.shuffle(rdm_indexes)
        train_indexes = rdm_indexes[:int(len(rdm_indexes) * (1 - validation_split))]
        val_indexes = rdm_indexes[int(len(rdm_indexes) * (1 - validation_split)):]

        x_train = x[train_indexes]
        labels_train = y[train_indexes]
        # save_to_numpy(x_train, labels_train,
        #               r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d5\training')
        x_val = x[val_indexes]
        labels_val = y[val_indexes]
        # save_to_numpy(x_val, labels_val,
        #               r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d1\validation')

        return x_train, x_val, labels_train, labels_val

    else:
        # save_to_numpy(x, y, r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d1\testing\Cohort A')
        return x, y


def load_to_dataset(signal_path,
                    labels_path,
                    resample_rate,
                    just_artifact_labels,
                    just_stage_labels,
                    validation_split):  # stft_size, stft_stride, fs, epoch_length,

    # Is load_recording_to_dataset_4

    if validation_split != 0:
        x_train, x_val, labels_train, labels_val = load_recording(signal_path,
                                                                  labels_path,
                                                                  resample_rate,
                                                                  just_artifact_labels,
                                                                  just_stage_labels,
                                                                  validation_split)

        input_dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
        labels_dataset_train = tf.data.Dataset.from_tensor_slices(labels_train)
        input_dataset_val = tf.data.Dataset.from_tensor_slices(x_val)
        labels_dataset_val = tf.data.Dataset.from_tensor_slices(labels_val)

        train_dataset = tf.data.Dataset.zip((input_dataset_train, labels_dataset_train))
        val_dataset = tf.data.Dataset.zip((input_dataset_val, labels_dataset_val))

        return train_dataset, val_dataset
    else:
        x, y = load_recording(signal_path,
                              labels_path,
                              resample_rate,
                              just_artifact_labels,
                              just_stage_labels,
                              validation_split)

        input_dataset = tf.data.Dataset.from_tensor_slices(x)
        labels_dataset = tf.data.Dataset.from_tensor_slices(y)

        dataset = tf.data.Dataset.zip((input_dataset, labels_dataset))

        return dataset


def filter_epochs(path, set, just_not_art_epochs, just_artifact_labels):
    file_list = pd.read_csv(path)

    if set == 'train':
        file_list = file_list[file_list['train'] == 1]
        file_list = file_list.sample(frac=1).reset_index(drop=True)
    elif set == 'validation':
        file_list = file_list[file_list['validation'] == 1]
    elif set == 'test':
        file_list = file_list[file_list['test'] == 1]

    if just_artifact_labels:
        file_list = file_list[['File', 'Art']]
    elif just_not_art_epochs:
        file_list = file_list[file_list['Art'] != 1]
        file_list = file_list[['File', 'NREM', 'REM', 'WAKE']]
    else:
        file_list = file_list[['File', 'NREM', 'REM', 'WAKE', 'Art']]

    return file_list


def load_and_concatenate(batch, data_folder):
    x_list = []
    y_list = []

    for i in range(len(batch)):
        x_list.append(np.load(os.path.join(data_folder, batch['File'].iloc[i])))

    x = np.stack(x_list)
    y = batch.drop(columns='File').to_numpy()

    return x, y


class SequenceDataset(tf.keras.utils.Sequence):

    def __init__(self, data_folder, csv_path, set, batch_size, just_not_art_epochs, just_artifact_labels):

        self.data_folder = data_folder
        self.file_list = filter_epochs(csv_path, set, just_not_art_epochs, just_artifact_labels)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_list.index) / self.batch_size))

    def __getitem__(self, idx):
        if idx < self.__len__()-1:
            batch = self.file_list.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        elif idx == self.__len__()-1:
            batch = self.file_list.iloc[idx * self.batch_size :]

        x, y = load_and_concatenate(batch, self.data_folder)

        return x, y
