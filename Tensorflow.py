import os
import pathlib
import sklearn.preprocessing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras

class Tensorflow:
    def __init__(self, data_dir: pathlib.Path, target_freq, epochs=3000, verbose=True):
        self.data_dir = data_dir
        self.epochs = epochs
        self.verbose = verbose
        self.target_freq = target_freq

        print(tf.version.VERSION)

        # Set seed for experiment reproducibility
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        root_dir_dataset = pathlib.Path(data_dir)
        commands = np.array(tf.io.gfile.listdir(str(root_dir_dataset) + '/RYAN_TRAIN/'))
        print(f'commends: {commands}')

        def decode_audio(audio_binary):
            # Decode WAV-encoded audio files to `float32` tensors, normalized
            # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
            audio, _ = tf.audio.decode_wav(contents=audio_binary)
            # Since all the data is single channel (mono), drop the `channels`
            # axis from the array.
            return tf.squeeze(audio, axis=-1)

        def get_label(file_path):
            parts = tf.strings.split(input=file_path, sep=os.path.sep)
            # Note: You'll use indexing here instead of tuple unpacking to enable this
            # to work in a TensorFlow graph
            return parts[-3]

        def get_waveform_and_label(file_path):
            label = get_label(file_path)
            audio_binary = tf.io.read_file(file_path)
            waveform = decode_audio(audio_binary)
            return waveform, label

        AUTOTUNE = tf.data.AUTOTUNE

        def preprocess_dataset(files):
            files_ds = tf.data.Dataset.from_tensor_slices(files)
            output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
            return output_ds

        def dataset_for_1d(dataset, start: int, end: int):
            x_dataset = []
            y_dataset = []
            for waveform, label in dataset.take(-1):
                waveform = tf.reshape(waveform, [len(waveform), 1]).numpy()
                label = tf.argmax(label == commands)
                x_dataset.append(waveform[start:end])
                y_dataset.append(label)
            return np.array(x_dataset), np.array(y_dataset)

        def make_datasets(__for: str):
            datasets = str(root_dir_dataset) + f'/*_{__for}'
            sum_x_data = list()
            sum_y_data = list()
            for __label in tf.io.gfile.glob(datasets + '/*'):
                stack_feature_x_data = list()
                stack_feature_y_data = list()
                for freq_name in target_freq:
                    if freq_name == '1271':
                        __x_data, __y_data = dataset_for_1d(
                            preprocess_dataset(tf.io.gfile.glob(__label + f'/*_{freq_name}_*/*')),
                            start=100, end=7400)
                    elif freq_name == '1313':
                        __x_data, __y_data = dataset_for_1d(
                            preprocess_dataset(tf.io.gfile.glob(__label + f'/*_{freq_name}_*/*')),
                            start=100, end=7400)
                    elif freq_name == '1356':
                        __x_data, __y_data = dataset_for_1d(
                            preprocess_dataset(tf.io.gfile.glob(__label + f'/*_{freq_name}_*/*')),
                            start=350, end=750)
                    elif freq_name == '1398':
                        __x_data, __y_data = dataset_for_1d(
                            preprocess_dataset(tf.io.gfile.glob(__label + f'/*_{freq_name}_*/*')),
                            start=350, end=750)
                    elif freq_name == '1440':
                        __x_data, __y_data = dataset_for_1d(
                            preprocess_dataset(tf.io.gfile.glob(__label + f'/*_{freq_name}_*/*')),
                            start=350, end=750)
                    else:
                        __x_data = 0
                        __y_data = 0
                    stack_feature_x_data.append(__x_data)
                    stack_feature_y_data.append(__y_data)
                sum_x_data.append(np.dstack(stack_feature_x_data))
                sum_y_data.append(np.array(stack_feature_y_data[0]))
            return np.concatenate(sum_x_data), np.concatenate(sum_y_data)
        x_train, y_train = make_datasets('TRAIN')
        x_test, y_test = make_datasets('TEST')
        x_val, y_val = make_datasets('VAL')

        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

        num_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

        # transform the labels from integers to one hot vectors
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

        # # save orignal y because later we will use binary
        # y_true = np.argmax(y_test, axis=1)

        def build_model(input_shape, nb_classes):
            padding = 'valid'
            input_layer = keras.layers.Input(input_shape)

            conv1 = keras.layers.Conv1D(filters=128, kernel_size=6, padding=padding, activation='relu')(input_layer)
            conv1 = keras.layers.AveragePooling1D(pool_size=2)(conv1)
            conv2 = keras.layers.Conv1D(filters=128, kernel_size=6, padding=padding, activation='relu')(conv1)
            conv2 = keras.layers.AveragePooling1D(pool_size=2)(conv2)

            flatten_layer = keras.layers.Flatten()(conv2)

            hidden_layer1 = keras.layers.Dense(10, activation='relu')(flatten_layer)
            hidden_layer2 = keras.layers.Dense(10, activation='relu')(hidden_layer1)
            output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(hidden_layer2)

            model = keras.models.Model(inputs=input_layer, outputs=output_layer)

            model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

            return model

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=verbose),
        ]

        history = build_model(x_train.shape[1:], num_classes).fit(
            x_train,
            y_train,
            batch_size=512,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, y_val),
            verbose=verbose,
        )

        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.show()

        y_pred = np.argmax(build_model(x_train.shape[1:], num_classes).predict(x_test), axis=1)
        y_true = y_test

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')

        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

