import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert

from SignalModel import SignalModel, Signal, get_sync_sample_position, default_normalized_signal


PICC_FREQ = 847500# Hz


class MainViewController:
    def __init__(self):
        self.model = SignalModel()
        self.read_wav_files()
        self.show_frist_raise_signal2()
        self.show_frist_raise_signal3()
        self.show_frist_raise_signal()
        # self.show_band()
        # self.nomalized_signal()

    def read_wav_files(self):
        self.model.read_wav_file('./wavFiles/ho_id_1.wav')
        self.model.read_wav_file('./wavFiles/ho_id_2.wav')
        self.model.read_wav_file('./wavFiles/ho_id_3.wav')
        self.model.read_wav_file('./wavFiles/su_id_1.wav')
        self.model.read_wav_file('./wavFiles/su_id_2.wav')
        self.model.read_wav_file('./wavFiles/su_id_3.wav')
        self.model.read_wav_file('./wavFiles/ok_id_1.wav')
        self.model.read_wav_file('./wavFiles/ok_id_2.wav')
        self.model.read_wav_file('./wavFiles/ok_id_3.wav')

    def show_frist_raise_signal(self):
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

            plt.plot(__bandpassed_data[sync_value[i] - min(sync_value):] * sync_ratio_value[i])
            plt.legend(__signal_data.lable)

        plt.show()

    def show_frist_raise_signal3(self):
        sync_value = []
        sync_ratio_value = []
        for data in self.model.signal_data[1:]:
            __first_signal_data_for_ref = self.model.signal_data[0]
            __ratio = default_normalized_signal(first_signal=__first_signal_data_for_ref, second_signal=data, sample_size= 94)
            __sync_x, __sync_y = get_sync_sample_position(first_signal=__first_signal_data_for_ref,
                                                          second_signal=data)
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

            plt.plot(__bandpassed_data[sync_value[i] - min(sync_value):] * sync_ratio_value[i])
            plt.legend(__signal_data.lable)

        plt.show()

    def show_frist_raise_signal2(self):
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

        plt.figure(2)
        __max_hilbert_value = []
        for i in range(len(self.model.signal_data)):
            __signal_data = self.model.signal_data[i]
            __bandpassed_data = __signal_data.butter_bandpass_filter(
                lowcut=PICC_FREQ - PICC_FREQ / 1.5,
                highcut=PICC_FREQ + PICC_FREQ / 1.5,
                order=2)

            __hilbert_envelope = Signal.hilbert_envelope(Signal(data=__bandpassed_data,
                                                                lable= __signal_data.lable,
                                                                samplerate= __signal_data.samplerate))
            __max_hilbert_value.append(max(__hilbert_envelope[sync_value[i]:sync_value[i]+93]))

            __ratio = __max_hilbert_value[0] / max(__hilbert_envelope[sync_value[i]:sync_value[i]+93])

            plt.plot(__hilbert_envelope[sync_value[i] - min(sync_value):] * __ratio)
            plt.legend(__signal_data.lable)

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

