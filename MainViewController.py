import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert
from Tensorflow import Tensorflow

from SignalModel import SignalModel, Signal


class MainViewController:
    def __init__(self):

        self.split_signal(11865000, './data/ori_sig/sam/band_real/sam_band_1186', 0.48, 0.23)
        self.split_signal(12712500, './data/ori_sig/sam/band_real/sam_band_1271', 0.00, 0.23)
        self.split_signal(13560000, './data/ori_sig/sam/band_real/sam_band_1356', 0.7, 0.3)
        self.split_signal(14407500, './data/ori_sig/sam/band_real/sam_band_1440', 0.55, 0.25)
        self.split_signal(15255000, './data/ori_sig/sam/band_real/sam_band_1525', 0.53, 0.18)
        self.split_signal(16102500, './data/ori_sig/sam/band_real/sam_band_1610', 0.4, 0.15)

        self.split_signal(11865000, './data/ori_sig/white/white_band_real/white_band_1186', 0.57, 0.21)
        self.split_signal(12712500, './data/ori_sig/white/white_band_real/white_band_1271', 0.8, 0.5)
        self.split_signal(13560000, './data/ori_sig/white/white_band_real/white_band_1356', 1, 0.6)
        self.split_signal(14407500, './data/ori_sig/white/white_band_real/white_band_1440', 1, 0.5)
        self.split_signal(15255000, './data/ori_sig/white/white_band_real/white_band_1525', 0.7, 0.35)
        self.split_signal(16102500, './data/ori_sig/white/white_band_real/white_band_1610', 0.4, 0.25)

        self.show_plot(11865000)
        self.show_plot(12712500)
        self.show_plot(13560000)
        self.show_plot(13560000)
        self.show_plot(14407500)
        self.show_plot(15255000)
        self.show_plot(16102500)

    def split_signal(self, cent_freq, original_signal_path, reqa_amp, ataq_amp):
        __model = SignalModel(cent_freq)
        __model.open_wav_folders(original_signal_path)

        for signal in __model.signal_data:
            __reqa_start_posions = __model.get_reqa_atqa_start_position(signal, reqa_amp, ataq_amp)
            if len(__reqa_start_posions) > 1:
                for __reqa_start_posion in __reqa_start_posions:
                    __start = int(__reqa_start_posion + signal.samplerate*(__model.iso14443.reqa_down_to_down_time + __model.iso14443.fraim_delay_time - __model.iso14443.ataq_time/16))
                    __stop = int(__start + signal.samplerate * (__model.iso14443.ataq_time + __model.iso14443.ataq_time/8))
                    __model.write_wav_file(
                        Signal(signal.samplerate,
                               signal.data[__start:__stop],
                               signal.file_path
                               ),
                        'atqa'
                    )

    def show_plot(self, cent_freq):
        __model = SignalModel(cent_freq)

        atqa_data_dir = Path('./data/tensorflow/atqa')
        if atqa_data_dir.is_dir():
            for forder_path in atqa_data_dir.glob(f'*_{cent_freq//10000}'):
                __model.open_wav_folders(forder_path)



        __model.get_normalized_data()

        __label_list = __model.get_current_label_list()
        for signal in __model.signal_data:
            plt.plot(signal.data, color='C%d' % __label_list.index(signal.label))

        plt.show()

    def show_enveloped_signal(self):
        __label_list = self.model.get_current_label_list()
        for data in self.model.get_enveloped_normalized_data(stop=94, cut=True):
            self.model.write_wav_file(data, 'envelop')
            plt.plot(data.data, color='C%d' % __label_list.index(data.label))

        plt.show()


"""
    def show_frist_raise_signal3(self):
        sync_value = []
        sync_ratio_value = []
        for data in self.model.signal_data[1:]:
            __first_signal_data_for_ref = self.model.signal_data[0]
            __ratio = default_normalized_signal(
                first_signal=__first_signal_data_for_ref,
                second_signal=data,
                sample_size=94
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
"""
