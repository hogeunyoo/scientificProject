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
        #
        # band_pass
        # self.split_signal(cent_freq=12712500,
        #                   band_real_signal_path='./data/bandpass/sam/sam_bp_1271',
        #                   target_signal_path='./data/bandpass/sam/sam_bp_1271',
        #                   reqa_amp=0.12, atqa_amp=0.045)
        #
        # self.split_signal(cent_freq=13560000,
        #                   band_real_signal_path='./data/bandpass/sam/sam_bp_1356',
        #                   target_signal_path='./data/bandpass/sam/sam_bp_1356',
        #                   reqa_amp=0.15, atqa_amp=0.14)



        # self.split_signal(cent_freq=13560000,
        #                   band_real_signal_path='./data/bandpass/pika/pika_bp_1356',
        #                   target_signal_path='./data/bandpass/pika/pika_bp_1356',
        #                   reqa_amp=0.19, atqa_amp=0.07)
        #
        # self.split_signal(cent_freq=13560000,
        #                   band_real_signal_path='./data/bandpass/sanggu/sanggu_bp_1356',
        #                   target_signal_path='./data/bandpass/sanggu/sanggu_bp_1356',
        #                   reqa_amp=0.24, atqa_amp=0.17)


        # self.split_signal(cent_freq=12712500,
        #                   band_real_signal_path='./data/bandpass/pika/pika_bp_1271',
        #                   target_signal_path='./data/bandpass/pika/pika_bp_1271',
        #                   reqa_amp=0.12, atqa_amp=0.045)

        # self.split_signal(cent_freq=12712500,
        #                   band_real_signal_path='./data/bandpass/sanggu/sanggu_bp_1271',
        #                   target_signal_path='./data/bandpass/sanggu/sanggu_bp_1271',
        #                   reqa_amp=0.15, atqa_amp=0.14))

        # self.split_signal(cent_freq=12712500,
        #                   band_real_signal_path='./data/bandpass/line1/line1_bp_1271',
        #                   target_signal_path='./data/bandpass/line1/line1_bp_1271',
        #                   reqa_amp=0.125, atqa_amp=0.1)
        #
        #
        # self.split_signal(cent_freq=12712500,
        #                   band_real_signal_path='./data/bandpass/line2/line2_bp_1271',
        #                   target_signal_path='./data/bandpass/line2/line2_bp_1271',
        #                   reqa_amp=0.125, atqa_amp=0.1)


        # self.split_signal(cent_freq=14407500,
        #                   band_real_signal_path='./data/bandpass/line1/line1_bp_1525',
        #                   target_signal_path='./data/bandpass/line1/line1_bp_1525',
        #                   reqa_amp=0.2, atqa_amp=0.12)
        #
        #
        # self.split_signal(cent_freq=14407500,
        #                   band_real_signal_path='./data/bandpass/line2/line2_bp_1525',
        #                   target_signal_path='./data/bandpass/line2/line2_bp_1525',
        #                   reqa_amp=0.2, atqa_amp=0.12)

        # self.split_signal(cent_freq=12712500,
        #                   band_real_signal_path='./data/ori_sig/sam/band_real/sam_band_1271',
        #                   target_signal_path='./data/ori_sig/sam/original/sam_1271',
        #                   reqa_amp=0.55, atqa_amp=0.23)
        # self.split_signal(cent_freq=13560000,
        #                   band_real_signal_path='./data/ori_sig/sam/band_real/sam_band_1356',
        #                   target_signal_path='./data/ori_sig/sam/original/sam_1356',
        #                   reqa_amp=0.7, atqa_amp=0.3)
        # self.split_signal(cent_freq=14407500,
        #                   band_real_signal_path='./data/ori_sig/sam/band_real/sam_band_1440',
        #                   target_signal_path='./data/ori_sig/sam/original/sam_1440',
        #                   reqa_amp=0.55, atqa_amp=0.25)
        # self.split_signal(cent_freq=15255000,
        #                   band_real_signal_path='./data/ori_sig/sam/band_real/sam_band_1525',
        #                   target_signal_path='./data/ori_sig/sam/original/sam_1525',
        #                   reqa_amp=0.53, atqa_amp=0.18)
        # self.split_signal(cent_freq=16102500,
        #                   band_real_signal_path='./data/ori_sig/sam/band_real/sam_band_1610',
        #                   target_signal_path='./data/ori_sig/sam/original/sam_1525',
        #                   reqa_amp=0.4, atqa_amp=0.15)

        # self.show_plot(11865000)
        # self.show_plot(12712500)
        self.show_plot(13560000)
        # self.show_plot(14407500)
        self.show_plot(15255000)
        # self.show_plot(16102500)

    def split_signal(self, cent_freq, band_real_signal_path, target_signal_path, reqa_amp, atqa_amp):
        __model_for_reqa_finder = SignalModel(cent_freq)
        __model_for_reqa_finder.open_wav_folders(band_real_signal_path)

        __reqa_start_posion = []
        for signal in __model_for_reqa_finder.signal_data:
            __reqa_start_posion = __model_for_reqa_finder.get_reqa_start_position(signal, reqa_amp, atqa_amp)

        __model = SignalModel(cent_freq)
        __model.open_wav_folders(target_signal_path)
        for signal in __model.signal_data:
            if len(__reqa_start_posion) > 1:
                for start_posion in __reqa_start_posion:
                    __start = int(start_posion + signal.samplerate * (__model.iso14443.reqa_down_to_down_time + __model.iso14443.fraim_delay_time - __model.iso14443.atqa_time / 16))
                    __stop = int(__start + signal.samplerate * (__model.iso14443.atqa_time + __model.iso14443.atqa_time / 8))
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
        # plt.plot(__model.signal_data[3].hilbert_envelope(), color='C%d' % __label_list.index(__model.signal_data[1].label))
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
