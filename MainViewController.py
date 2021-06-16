import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert
from Tensorflow import Tensorflow

from SignalModel import SignalModel, Signal, get_sync_sample_position, default_normalized_signal


PICC_FREQ = 847500# Hz


class MainViewController:
    def __init__(self):
        self.model = SignalModel()
        self.read_wav_files()
        self.show_enveloped_signal()
        Tensorflow(data_dir=Path('./data/tensorflow/envelop'))
        # Tensorflow(data_dir=Path('./data/wavFiles'))

        # self.show_frist_raise_signal3()
        # self.show_frist_raise_signal()
        # self.show_band()
        # self.nomalized_signal()

    def read_wav_files(self):
        self.model.open_wav_folders('./data/wavFiles/blue_wavefiles')
        self.model.open_wav_folders('./data/wavFiles/card_ook')
        self.model.open_wav_folders('./data/wavFiles/card_ho_wavefiles')
        # self.model.open_wav_folders('./data/wavFiles/card_won')
        self.model.open_wav_folders('./data/wavFiles/phone')

    def show_frist_raise_signal3(self):
        sync_value = []
        sync_ratio_value = []
        for data in self.model.signal_data[1:]:
            __first_signal_data_for_ref = self.model.signal_data[0]
            __ratio = default_normalized_signal(
                first_signal=__first_signal_data_for_ref,
                second_signal=data,
                sample_size= 94
            )
            __sync_x, __sync_y = get_sync_sample_position(
                first_signal=__first_signal_data_for_ref,
                second_signal=data
            )
            if len(sync_value) == 0 and len(sync_ratio_value) == 0:
                sync_value.append(__sync_x)
                sync_value.append(__sync_y)
                sync_ratio_value.append(1)
                sync_ratio_value.append(__ratio)
            else:
                sync_value.append(__sync_y)
                sync_ratio_value.append(__ratio)

        plt.figure(1)
        for i in range(len(self.model.signal_data)):
            __signal_data = self.model.signal_data[i]
            __bandpassed_data = __signal_data.butter_bandpass_filter(
                lowcut=PICC_FREQ - PICC_FREQ / 1.2,
                highcut=PICC_FREQ + PICC_FREQ / 1.2,
                order=1)

            plt.plot(__bandpassed_data[sync_value[i] - min(sync_value):] * sync_ratio_value[i])
            plt.legend()

        plt.show()

    def show_frist_raise_signal(self):
        __label_list = self.model.get_current_label_list()
        sync_value = []
        sync_ratio_value = []
        for data in self.model.signal_data[1:]:
            __first_signal_data_for_ref = self.model.signal_data[0]
            __ratio = default_normalized_signal(first_signal=__first_signal_data_for_ref, second_signal=data)
            __sync_x, __sync_y = get_sync_sample_position(first_signal=__first_signal_data_for_ref, second_signal=data)
            if len(sync_value) == 0 and len(sync_ratio_value) == 0:
                sync_value.append(__sync_x)
                sync_value.append(__sync_y)
                sync_ratio_value.append(1)
                sync_ratio_value.append(__ratio)
            else:
                sync_value.append(__sync_y)
                sync_ratio_value.append(__ratio)

        plt.figure(1)
        for i in range(len(self.model.signal_data)):
            __signal_data = self.model.signal_data[i]
            __bandpassed_data = __signal_data.butter_bandpass_filter(
                lowcut=PICC_FREQ - PICC_FREQ / 2,
                highcut=PICC_FREQ + PICC_FREQ / 2,
                order=1)

            plt.plot(__bandpassed_data[sync_value[i] - min(sync_value):] * sync_ratio_value[i],
                     color='C%d' % __label_list.index(__signal_data.label))
        plt.legend()

        plt.show()

    def show_enveloped_signal(self):
        __label_list = self.model.get_current_label_list()
        for data in self.model.get_enveloped_normalized_data(stop=94):
            self.model.write_wav_file(data, 'envelop')
            plt.plot(data.data, color='C%d' % __label_list.index(data.label))

        plt.legend()
        plt.show()




    def show_band(self):
        band_passed_first_signal = self.model.signal_data[0].butter_bandpass_filter(
            lowcut=PICC_FREQ - PICC_FREQ / 2,
            highcut=PICC_FREQ + PICC_FREQ / 2,
            order=1)

        band_passed_second_signal = self.model.signal_data[1].butter_bandpass_filter(
            lowcut=PICC_FREQ - PICC_FREQ / 2,
            highcut=PICC_FREQ + PICC_FREQ / 2,
            order=1)

        x, y = get_sync_sample_position(self.model.signal_data[0], self.model.signal_data[1])

        plt.plot(band_passed_first_signal[x:])
        plt.plot(band_passed_second_signal[y:])
        plt.show()


    def nomalized_signal(self):
        band_passed_first_signal = self.model.signal_data[0].butter_bandpass_filter(
            lowcut=PICC_FREQ - PICC_FREQ / 2,
            highcut=PICC_FREQ + PICC_FREQ / 2,
            order=1)

        band_passed_second_signal = self.model.signal_data[1].butter_bandpass_filter(
            lowcut=PICC_FREQ - PICC_FREQ / 2,
            highcut=PICC_FREQ + PICC_FREQ / 2,
            order=1)

        x, y = get_sync_sample_position(self.model.signal_data[0], self.model.signal_data[1])
        __nomalize_ratio = default_normalized_signal(self.model.signal_data[0], self.model.signal_data[1])

        plt.plot(band_passed_first_signal[x:])
        plt.plot(band_passed_second_signal[y:] * __nomalize_ratio)
        plt.show()

