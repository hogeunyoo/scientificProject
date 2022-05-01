import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path, PurePath
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert
from Tensorflow import Tensorflow

from SignalModel import SignalModel, Signal


class MainViewController:
    def __init__(self):
        # self.split_signals('line_1')
        # self.split_signals('line_2')
        # self.split_signals('line_3')
        # self.split_signals('line_4')
        # self.split_signals('line_5')
        # self.split_signals('line_6')
        # self.atqa_signal_write_for_tensor(12712500)
        # self.atqa_signal_write_for_tensor(13136250)
        # self.atqa_signal_write_for_tensor(13560000)
        # self.atqa_signal_write_for_tensor(14407500)
        # self.atqa_signal_write_for_tensor(14831250)
        # self.atqa_signal_write_for_tensor(15255000)
        # self.show_plot(12712500, PurePath('./data/tensorflow/atqa/1271'))
        # self.show_plot(13136250, PurePath('./data/tensorflow/atqa'))
        # self.show_plot(13560000, PurePath('./data/tensorflow/atqa/1356'))
        # self.show_plot(14407500, PurePath('./data/tensorflow/atqa'))
        # self.show_plot(14831250, PurePath('./data/tensorflow/atqa'))
        # self.show_plot(15255000, PurePath('./data/tensorflow/atqa'))
        Tensorflow(data_dir=Path('./data/tensorflow/atqa/1271_tmp'))

    def split_signals(self, card_name: str):
        # band_pass
        data_dir = '/home/pcl621/Developer/data/signal'
        self.split_signal(cent_freq=12712500,
                          band_real_signal_path=f'{data_dir}/{card_name}/{card_name}_1271',
                          target_signal_path=f'{data_dir}/{card_name}/{card_name}_1271')
        # self.split_signal(cent_freq=13136250,
        #                   band_real_signal_path=f'{data_dir}/{card_name}/{card_name}_1356',
        #                   target_signal_path=f'{data_dir}/{card_name}/{card_name}_1356')
        # self.split_signal(cent_freq=13560000,
        #                   band_real_signal_path=f'{data_dir}/{card_name}/{card_name}_1313',
        #                   target_signal_path=f'{data_dir}/{card_name}/{card_name}_1313')
        # self.split_signal(cent_freq=13983750,
        #                   band_real_signal_path=f'{data_dir}/{card_name}/{card_name}_1398',
        #                   target_signal_path=f'{data_dir}/{card_name}/{card_name}_1398')
        # self.split_signal(cent_freq=14407500,
        #                   band_real_signal_path=f'{data_dir}/{card_name}/{card_name}_1440',
        #                   target_signal_path=f'{data_dir}/{card_name}/{card_name}_1440')
        # self.split_signal(cent_freq=14831250,
        #                   band_real_signal_path=f'{data_dir}/{card_name}/{card_name}_1483',
        #                   target_signal_path=f'{data_dir}/{card_name}/{card_name}_1483')
        # self.split_signal(cent_freq=15255000,
        #                   band_real_signal_path=f'{data_dir}/{card_name}/{card_name}_1525',
        #                   target_signal_path=f'{data_dir}/{card_name}/{card_name}_1525')

    def split_signal(self, cent_freq, band_real_signal_path, target_signal_path):
        __model_for_reqa_finder = SignalModel(cent_freq)
        __model_for_reqa_finder.open_wav_folders(band_real_signal_path)

        __reqa_start_posion = []
        for signal in __model_for_reqa_finder.signal_data:
            __reqa_start_posion = __model_for_reqa_finder.get_reqa_start_position(signal)

        __model = SignalModel(cent_freq)
        __model.open_wav_folders(target_signal_path)
        __write_model = SignalModel(cent_freq)
        for signal in __model.signal_data:
            if len(__reqa_start_posion) > 1:
                for start_posion in __reqa_start_posion:
                    __start = int(start_posion + signal.samplerate * (__model.iso14443.reqa_down_to_down_time + __model.iso14443.fraim_delay_time - __model.iso14443.atqa_time / 4))
                    __stop = int(__start + signal.samplerate * (__model.iso14443.atqa_time + __model.iso14443.atqa_time / 2))
                    __write_model.signal_data.append(
                        Signal(signal.samplerate,
                               signal.data[__start:__stop],
                               signal.file_path))

        for write_signal in __write_model.signal_data:
            __atqa_sample_count = int(__write_model.iso14443.atqa_time * signal.samplerate)
            __base_sink, __target_sink = __write_model.get_sync_sample_position(
                __write_model.signal_data[1], write_signal, 100)
            if len(write_signal.data[
                   __target_sink - __atqa_sample_count // 6:
                   __target_sink + __atqa_sample_count + __atqa_sample_count // 6])\
                    == __atqa_sample_count + 2 * (__atqa_sample_count // 6):
                __write_model.write_wav_file(
                    Signal(write_signal.samplerate,
                           write_signal.data[__target_sink - __atqa_sample_count//6:
                                             __target_sink + __atqa_sample_count + __atqa_sample_count//6],
                           write_signal.file_path
                           ),
                    'separate/'
                )

    def atqa_signal_write_for_tensor(self, cent_freq):
        __model = SignalModel(cent_freq)

        reqa_atqa_data_dir = Path('./data/separate')
        if reqa_atqa_data_dir.is_dir():
            for forder_path in reqa_atqa_data_dir.glob(f'*_{cent_freq//10000}'):
                __model.open_wav_folders(forder_path)

        __label_list = __model.get_current_label_list()
        for signal in __model.signal_data:
            __atqa_sample_count = int(__model.iso14443.atqa_time * signal.samplerate)
            __base_sink, __target_sink = __model.get_sync_sample_position(__model.signal_data[1], signal, 900)
            if len(signal.data[__target_sink - __atqa_sample_count//19:__target_sink + __atqa_sample_count + __atqa_sample_count//19]) == __atqa_sample_count + 2*(__atqa_sample_count//19):
                __model.write_wav_file(Signal(samplerate=signal.samplerate,
                                              data=signal.data[__target_sink - __atqa_sample_count//19:
                                                               __target_sink + __atqa_sample_count + __atqa_sample_count//19],
                                              file_path=signal.file_path
                                              ),
                                       'tensorflow/atqa/')

    def show_plot(self, cent_freq, dir_path: PurePath):
        __model = SignalModel(cent_freq)

        atqa_data_dir = Path(dir_path)
        if atqa_data_dir.is_dir():
            for forder_path in atqa_data_dir.glob(f'*_{cent_freq//10000}*'):
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
