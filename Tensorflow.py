import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import feature_column

from IPython import display

class Tensorflow:
    def __init__(self, data_dir: pathlib.Path, epochs=100):
        self.data_dir = data_dir
        self.epochs = epochs

        print(tf.version.VERSION)

        # Set seed for experiment reproducibility
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        print('Commands:', commands)

        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
        print(len(filenames))
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        print('Number of total examples:', num_samples)
        print('Number of examples per label:',
              len(tf.io.gfile.listdir(str(data_dir / commands[0]))))
        print('Example file tensor:', filenames[0])

        __training_set_size = len(filenames) - round(len(filenames)/10) * 2
        __validation_set_size = round(len(filenames)/10) * 1
        __test_set_size = round(len(filenames)/10) * 1

        train_files = filenames[:__training_set_size]
        val_files = filenames[__training_set_size: __training_set_size + __validation_set_size]
        test_files = filenames[-__test_set_size:]

        print('Training set size', len(train_files))
        print('Validation set size', len(val_files))
        print('Test set size', len(test_files))

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
            # to work in a TensorFlow graph.
            return parts[-2]

        def get_waveform_and_label(file_path):
            label = get_label(file_path)
            audio_binary = tf.io.read_file(file_path)
            waveform = decode_audio(audio_binary)
            return waveform, label

        AUTOTUNE = tf.data.AUTOTUNE
        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)

        for waveform, label in waveform_ds.take(1):
            label = label.numpy().decode('utf-8')

        print('Label:', label)
        print('Waveform shape:', waveform.shape)

        def preprocess_dataset(files):
            files_ds = tf.data.Dataset.from_tensor_slices(files)
            output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
            return output_ds

        def dataset_for_1d(dataset):
            x_dataset = []
            y_dataset = []
            for waveform, label in dataset.take(-1):
                waveform = tf.reshape(waveform, [len(waveform), 1]).numpy()
                # label = label.numpy()
                label = tf.argmax(label == commands)
                x_dataset.append(waveform[300:800])
                y_dataset.append(label)
                print(label)
            return np.array(x_dataset), np.array(y_dataset)

        train_ds = waveform_ds
        val_ds = preprocess_dataset(val_files)
        test_ds = preprocess_dataset(test_files)

        print('여기!!!!')
        x_train, y_train = dataset_for_1d(train_ds)
        x_test, y_test = dataset_for_1d(test_ds)

        num_classes = len(np.unique(y_train))


        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

        model = keras.Sequential()
        model.add(keras.layers.Conv1D(filters=6, kernel_size=3, activation='relu', input_shape=(x_train.shape[1:])))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        keras.utils.plot_model(model, show_shapes=True)

        epochs = 500
        batch_size = 64

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=1,
        )

        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.show()

        y_pred = np.argmax(model.predict(x_test), axis=1)
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

