import os
import sys
import mne
import pandas as pd
import numpy as np
import tensorflow as tf
sys.path.append(os.path.join(sys.path[0], ".."))
from final_preprocessing import preprocess_EEG, preprocess_EMG
import string
import os
import scipy
import random


def load_raw_recording(file_path, resample_rate=None):
    """
    :param file_path: path to the .edf recording
    :param resample_rate: new sampling rate in Hertz
    :return: numpy array with the signal
    """

    data = mne.io.read_raw_edf(file_path)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    if resample_rate:
        new_num_samples = raw_data.shape[1]/info['sfreq']*resample_rate
        if new_num_samples.is_integer() is False:
            raise Exception("New number of samples is not integer")

        raw_data = scipy.signal.resample(x=raw_data, num=int(new_num_samples), axis=1)

    return raw_data


def windowing(signal, window_size=32*5, window_stride=32, fs=128):
    n_windows = np.floor((signal.shape[2] - window_size) / window_stride).astype(int) + 1
    # if ((n_windows - 1 + 1)*window_stride + window_size) < signal.shape[2]:
    #     n_windows += 1

    windowed_signal = np.zeros((n_windows, 3, 48, window_size))

    # signal = signal[window_size//2 : -window_size//2],

    for i in range(n_windows):
        windowed_signal[i, :, :, :] = signal[:, :, (i*window_stride) : (i*window_stride) + window_size]

    return windowed_signal


def load_labels(labels_path,
                  scorer,
                  just_artifact_labels,
                  keep_artifacts,
                  artifact_to_stages,
                  cohort):

    # Is load_labels_2

    df = pd.read_csv(labels_path, header=None)

    # column names: {1, 2, 3, n, r, w}
    # 1=wake artifact, 2=NREM artifact, 3=REM artifact

    df = df.iloc[2:-2].copy() # Drop 2 first and 2 last epochs

    labels_1 = pd.get_dummies(df[1], dtype=int)
    labels_2 = pd.get_dummies(df[2], dtype=int)

    if 'a' in labels_1:
        labels_1.drop('a', axis=1, inplace=True)

    if scorer==0:
        labels = labels_1[(labels_1 == labels_2).all(axis=1) == True].copy()
    elif scorer==1:
        labels = labels_1
    elif scorer==2:
        labels = labels_2

    if just_artifact_labels==True:
        labels.loc[(labels['1'] == 1) | (labels['2'] == 1) | (labels['3'] == 1), 'Art'] = 1
        labels.loc[(labels['1'] == 0) & (labels['2'] == 0) & (labels['3'] == 0), 'Art'] = 0

        labels = labels['Art']
    elif just_artifact_labels==False:
        if keep_artifacts==True:
            if artifact_to_stages==True:
                if cohort=='d':
                    if '1' in labels.columns:
                        labels.loc[labels["1"] == 1, 'w'] = 1
                        labels.drop('1', axis=1, inplace=True)
                else: 
                    labels.loc[labels["1"] == 1, 'w'] = 1
                    labels.loc[labels["2"] == 1, 'n'] = 1
                    labels.loc[labels["3"] == 1, 'r'] = 1

                    labels = labels.iloc[:, -3:]
            elif artifact_to_stages==False:
                if cohort=='d':
                    if '1' in labels.columns:
                        labels.loc[labels['1'] == 1, 'Art'] = 1
                        labels.loc[labels['1'] == 0, 'Art'] = 0
                        labels.drop('1', axis=1, inplace=True)
                else: 
                    labels.loc[(labels['1'] == 1) | (labels['2'] == 1) | (labels['3'] == 1), 'Art'] = 1
                    labels.loc[(labels['1'] == 0) & (labels['2'] == 0) & (labels['3'] == 0), 'Art'] = 0

                    labels.drop(['1', '2', '3'], axis=1, inplace=True)
        elif keep_artifacts==False:
            labels.drop(labels.loc[(labels['1'] == 1) | (labels['2'] == 1) | (labels['3'] == 1)].index, inplace=True)
            labels = labels.iloc[:, -3:]

    labels.rename(columns={"w": "WAKE", "n": "NREM", "r": "REM"})

    return labels


def balanced_artifacts(labels, not_art_ratio):
    art_epochs = np.where(labels == 1)[0]
    n_art_epochs = np.where(labels == 0)[0]

    # rdm_idx = np.random.randint(0, len(n_art_epochs), size=(len(art_epochs),))
    rdm_idx = np.random.randint(0, len(n_art_epochs), size=(int(len(art_epochs) * not_art_ratio / (1-not_art_ratio)),))
    rdm_n_art = n_art_epochs[rdm_idx]

    balanced_subset_idx = np.concatenate((art_epochs, rdm_n_art))

    return balanced_subset_idx


def save_to_numpy(data, labels, path):

    for idx, r  in enumerate(data):
        characters = string.ascii_lowercase + string.digits
        filename = ''.join(random.choice(characters) for i in range(16))
        # filename = filename + '_label_' + str(np.argmax(labels[idx]))
        filename = filename + '_label_' + str(int(labels[idx]))
        np.save(os.path.join(path, filename), r)


def load_recording(signal_paths,
                   labels_paths,
                   scorer,
                   just_artifact_labels,
                   artifact_to_stages,
                   keep_artifacts,
                   balance_artifacts,
                   validation_split,
                   cohort):  # stft_size, stft_stride, fs, epoch_length,

    # Is load_recording_to_dataset_4

    for i in range(len(signal_paths)):
        raw_data = load_raw_recording(signal_paths[i])
        eeg_1 = preprocess_EEG(raw_data[0, :])
        eeg_2 = preprocess_EEG(raw_data[1, :])
        emg = preprocess_EMG(raw_data[2, :])
        x_mouse = np.stack((eeg_1, eeg_2, emg))
        x_mouse = windowing(x_mouse, window_size=32 * 5, window_stride=32)
        x_mouse = np.transpose(x_mouse, (0, 3, 2, 1))

        if scorer == 3:
            y1 = load_labels(labels_paths[i],
                             scorer=1,
                             just_artifact_labels=just_artifact_labels,
                             artifact_to_stages=artifact_to_stages)
            x1 = x_mouse[y1.index.to_numpy() - 2]  # Select just the epochs in y
            y1 = y1.to_numpy()
            y2 = load_labels(labels_paths[i],
                             scorer=2,
                             just_artifact_labels=just_artifact_labels,
                             artifact_to_stages=artifact_to_stages)
            x2 = x_mouse[y2.index.to_numpy() - 2]  # Select just the epochs in y
            y2 = y2.to_numpy()
            x_mouse = np.concatenate((x1, x2), axis=0)
            y_mouse = np.concatenate((y1, y2), axis=0)
        else:
            y_mouse = load_labels(labels_paths[i],
                                  scorer=scorer,
                                  just_artifact_labels=just_artifact_labels,
                                  keep_artifacts=keep_artifacts,
                                  artifact_to_stages=artifact_to_stages,
                                  cohort=cohort)
            x_mouse = x_mouse[y_mouse.index.to_numpy() - 2]  # Select just the epochs in y
            y_mouse = y_mouse.to_numpy()

        if just_artifact_labels and balance_artifacts:
            balanced_subset_idx = balanced_artifacts(y_mouse)
            y_mouse = y_mouse[balanced_subset_idx]
            x_mouse = x_mouse[balanced_subset_idx]

        if i == 0:
            x = x_mouse
            y = y_mouse
        else:
            x = np.concatenate((x, x_mouse), axis=0)
            y = np.concatenate((y, y_mouse), axis=0)

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


def load_to_dataset(signal_paths,
                    labels_paths,
                    scorer,
                    just_artifact_labels,
                    artifact_to_stages,
                    balance_artifacts,
                    validation_split, 
                    cohort):  # stft_size, stft_stride, fs, epoch_length,

    # Is load_recording_to_dataset_4

    if validation_split != 0:
        x_train, x_val, labels_train, labels_val = load_recording(signal_paths,
                                                                  labels_paths,
                                                                  scorer,
                                                                  just_artifact_labels,
                                                                  artifact_to_stages,
                                                                  balance_artifacts,
                                                                  validation_split)

        input_dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
        labels_dataset_train = tf.data.Dataset.from_tensor_slices(labels_train)
        input_dataset_val = tf.data.Dataset.from_tensor_slices(x_val)
        labels_dataset_val = tf.data.Dataset.from_tensor_slices(labels_val)

        train_dataset = tf.data.Dataset.zip((input_dataset_train, labels_dataset_train))
        val_dataset = tf.data.Dataset.zip((input_dataset_val, labels_dataset_val))

        return train_dataset, val_dataset
    else:
        x, y = load_recording(signal_paths,
                              labels_paths,
                              scorer,
                              just_artifact_labels,
                              artifact_to_stages,
                              balance_artifacts,
                              validation_split,
                              cohort)

        input_dataset = tf.data.Dataset.from_tensor_slices(x)
        labels_dataset = tf.data.Dataset.from_tensor_slices(y)

        dataset = tf.data.Dataset.zip((input_dataset, labels_dataset))

        return dataset


def load_and_concatenate(file_list, binary):
    x_list = []
    y_list = []


    for idx, f in enumerate(file_list):
        # if idx == 0:
        #     x = np.load(f)
        #     y = np.array(int(f[-5]))
        # else:
        #     x = np.stack((x, np.load(f)))
        #     y = np.stack((y, np.array(int(f[-5]))))

        x_list.append(np.load(f))
        y_list.append(np.array(float(f[-5])))

    x = np.stack(x_list)
    y = np.stack(y_list)

    if binary:
        return x, y
    elif not binary:
        y_dummy = np.zeros((y.size, 3))
        y_dummy[np.arange(y.size), y.astype(np.int)] = 1
        # y = pd.get_dummies(np.stack(y_list)).to_numpy()

        return x, y_dummy


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class SequenceDataset(tf.keras.utils.Sequence):

    def __init__(self, path, batch_size, binary):

        self.file_list = listdir_fullpath(path)
        random.shuffle(self.file_list)
        self.batch_size = batch_size
        self.binary = binary

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        if idx < self.__len__()-1:
            x, y = load_and_concatenate(self.file_list[idx * self.batch_size : (idx + 1) * self.batch_size], self.binary)
        elif idx == self.__len__() - 1:
            x, y = load_and_concatenate(self.file_list[idx * self.batch_size :], self.binary)

        return x, y


def load_and_concatenate2(batch, data_folder):
    x_list = []
    y_list = []

    for i in range(len(batch)):
        x_list.append(np.load(os.path.join(data_folder, batch['File'].iloc[i])))

    x = np.stack(x_list)
    y = batch.drop(columns='File').to_numpy()

    return x, y


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


class SequenceDataset2(tf.keras.utils.Sequence):

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

        x, y = load_and_concatenate2(batch, self.data_folder)

        return x, y