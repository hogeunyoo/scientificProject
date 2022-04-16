from pathlib import Path, PurePath
from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert
import numpy as np

# PICC_FREQ = 847500 # fc/16 Hz
# REQA_TIME = 0.000084955752212 # s 1152/fc
# REQA_DOWN_TO_DOWN_TIME = 0.000077876106194 # s 1056/fc
# ATAQ_TIME = 0.000179351032448 # s 2432/fc
#
# FRAIM_DELAY_TIME = 0.000086430678466 # REQA TO ATAQ 1172/fc


class ISO14443:
    def __init__(self, cent_freq):
        self.cent_freq = cent_freq
        self.picc_freq = cent_freq/16
        self.reqa_time = 1152/13560000  # 1152/cent_freq 녹음 환경 조건
        self.reqa_down_to_down_time = 1056/13560000  # 1056/cent_freq 녹음 환경 조건
        self.atqa_time = 2432 / cent_freq
        self.fraim_delay_time = 1172/cent_freq  # 1172
        self.reqa_to_ataq = self.reqa_down_to_down_time + self.fraim_delay_time + self.atqa_time

        self.atqa_finder = [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0]  # USE RFU, PARITY, Proprietary coding


        # np.repeat([1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0], samp_rate*64//cent_freq


class Signal:
    def __init__(self, samplerate, data, file_path):
        self.samplerate = samplerate
        self.data = data.astype(np.float64)
        self.file_path = file_path
        self.label = file_path.parent.name

    def ATQA_SAMPLE_SIZE(self):
        return round(1 / 847500 * 8 * self.samplerate * 18.5)

    def butter_bandpass_filter(self, lowcut, highcut, order=1):
        nyq = 0.5 * self.samplerate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, self.data)
        return y

    def hilbert_envelope(self):
        analytic_signal = hilbert(self.data)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope


"""
def default_normalized_signal(first_signal, second_signal, accuracy=1000, sample_size=3493):
    __first_sink, __second_sink = get_sync_sample_position(first_signal, second_signal)

    filtered_first_signal = first_signal.butter_bandpass_filter(
        lowcut=PICC_FREQ - PICC_FREQ / 2,
        highcut=PICC_FREQ + PICC_FREQ / 2,
        order=1)
    filtered_second_signal = second_signal.butter_bandpass_filter(
        lowcut=PICC_FREQ - PICC_FREQ / 2,
        highcut=PICC_FREQ + PICC_FREQ / 2,
        order=1)

    __ratio = 0
    __find_min_value = 100000000

    for amp in range(accuracy):
        if __find_min_value > (
                filtered_first_signal[__first_sink:__first_sink + sample_size] -
                filtered_second_signal[__second_sink:__second_sink + sample_size] *
                10 * amp / accuracy).std():
            __find_min_value = (
                    filtered_first_signal[__first_sink:__first_sink + sample_size] -
                    filtered_second_signal[__second_sink:__second_sink + sample_size] *
                    10 * amp / accuracy).std()
            __ratio = 10 * amp / accuracy
    return __ratio


def get_sync_sample_position(first_signal: Signal, second_signal: Signal):
    filtered_first_signal = first_signal.butter_bandpass_filter(
        lowcut=PICC_FREQ - PICC_FREQ / 2,
        highcut=PICC_FREQ + PICC_FREQ / 2,
        order=1)
    filtered_second_signal = second_signal.butter_bandpass_filter(
        lowcut=PICC_FREQ - PICC_FREQ / 2,
        highcut=PICC_FREQ + PICC_FREQ / 2,
        order=1)

    first_ATQA_sample_position = 0
    second_ATQA_sample_posion = 0

    __find_max_value_one = 0
    __find_max_value_two = 0

    __value = 100000000

    for i in range(len(filtered_first_signal) - first_signal.ATQA_SAMPLE_SIZE()):
        if __find_max_value_one < np.abs(filtered_first_signal[i:i + first_signal.ATQA_SAMPLE_SIZE() - 1]).sum():
            first_ATQA_sample_position = i
            __find_max_value_one = np.abs(filtered_first_signal[i:i + first_signal.ATQA_SAMPLE_SIZE() - 1]).sum()

    for j in range(len(filtered_second_signal) - second_signal.ATQA_SAMPLE_SIZE()):
        if __find_max_value_two < np.abs(filtered_second_signal[j:j + second_signal.ATQA_SAMPLE_SIZE() - 1]).sum():
            __find_max_value_two = np.abs(filtered_second_signal[j:j + second_signal.ATQA_SAMPLE_SIZE() - 1]).sum()

    __ratio = __find_max_value_one / __find_max_value_two

    filtered_second_signal = filtered_second_signal * __ratio

    for second_signal_sample in range(len(filtered_second_signal) - second_signal.ATQA_SAMPLE_SIZE()):
        a = (filtered_first_signal[first_ATQA_sample_position:first_ATQA_sample_position + first_signal.ATQA_SAMPLE_SIZE() - 1] -
             filtered_second_signal[second_signal_sample:second_signal_sample + first_signal.ATQA_SAMPLE_SIZE() - 1]
             ).std()
        if __value > a:
            __value = a
            second_ATQA_sample_posion = second_signal_sample

    return first_ATQA_sample_position, second_ATQA_sample_posion
"""

class SignalModel:
    def __init__(self, cent_freq):
        self.signal_data = []
        self.cent_freq = cent_freq
        self.iso14443 = ISO14443(cent_freq)

    def open_wav_folders(self, folder_path):
        data_dir = Path(folder_path)
        if data_dir.is_dir():
            for file_path in data_dir.glob('*.wav'):
                self.read_wav_file(file_path)
        else:
            print("ERROR : Please input wav files")

    def read_wav_file(self, file_path):
        __sample_rate = wavfile.read(file_path)[0]
        __data = wavfile.read(file_path)[1]

        __current_signal = Signal(__sample_rate, __data, file_path)

        if len(self.signal_data) != 0:
            if self.signal_data[0].samplerate == __current_signal.samplerate:
                self.signal_data.append(__current_signal)
            else:
                print('ERROR : 신호 파일 불량, %s' % (str(file_path)))
        else:
            self.signal_data.append(__current_signal)

    def write_wav_file(self, signal: Signal, name: str):
        dir_path = Path('./data/tensorflow/' + name + '/' + signal.label)
        dir_path.mkdir(parents=True, exist_ok=True)
        __data_num = 0
        __file_path = PurePath(signal.label + '_' + str(__data_num) + '.wav')
        while (dir_path/__file_path).exists():
            __data_num += 1
            __file_path = PurePath(signal.label + '_' + str(__data_num) + '.wav')
        wavfile.write(dir_path/__file_path, signal.samplerate, signal.data.astype(np.int16))

    def get_reqa_start_position(self, signal: Signal, reqa_amp, ataq_amp):

        __reqa_start_position = []

        # constant
        __reqa_down_to_down_sample_count = int(signal.samplerate * self.iso14443.reqa_down_to_down_time) # 소수점 버림
        __reqa_to_ataq_sample_count = int(signal.samplerate * self.iso14443.reqa_to_ataq)
        __fraim_delay_sample_count = int(signal.samplerate * self.iso14443.fraim_delay_time)
        __ataq_sample_count = int(signal.samplerate * self.iso14443.atqa_time)
        __atqa_finder = np.repeat(self.iso14443.atqa_finder, round(signal.samplerate * 64 / self.cent_freq))

        __amp_threshold = ((ataq_amp + reqa_amp) * 32767.5 - 0.5 ) /2

        # draft_position
        __draft_reqa_start = []
        __end = len(signal.data)
        __draft_position = 0

        while __draft_position < __end:
            __current_max = np.max(signal.data[__draft_position:__draft_position+__reqa_to_ataq_sample_count])
            if __current_max > __amp_threshold:
                print(__current_max, __amp_threshold)
                if int(__end * 0.01) < __draft_position - __reqa_down_to_down_sample_count < int(__end * 0.99):
                    __draft_reqa_start.append(__draft_position - __reqa_down_to_down_sample_count)
                __draft_position += __reqa_to_ataq_sample_count
            else:
                __draft_position += __reqa_to_ataq_sample_count

        print(__draft_reqa_start)
        for __position in __draft_reqa_start:
            __speed = __reqa_down_to_down_sample_count // 8  # 8~32, 8 권장 작을수록 빠름
            __i = __position

            while __i < __position + __reqa_to_ataq_sample_count:
                __current_max = np.max(signal.data[__i:__i+__reqa_down_to_down_sample_count])
                __current_min = np.min(signal.data[__i:__i+__reqa_down_to_down_sample_count])

                if __current_max - __current_min > __amp_threshold:
                    __current_abs_mean_fdt = np.mean(np.abs(signal.data[
                                                        __i + __reqa_down_to_down_sample_count + __fraim_delay_sample_count//10:
                                                        __i + __reqa_down_to_down_sample_count + __fraim_delay_sample_count - __fraim_delay_sample_count//20
                                                        ]))
                    __current_abs_mean_atqa = np.mean(np.abs(signal.data[
                                                        __i + __reqa_down_to_down_sample_count + __fraim_delay_sample_count - __ataq_sample_count//608:
                                                        __i + __reqa_down_to_down_sample_count + __fraim_delay_sample_count + __ataq_sample_count//38 + __ataq_sample_count//608
                                                        ]))
                    __found_max = __i, __current_abs_mean_atqa - __current_abs_mean_fdt

                    for __j in range(__reqa_down_to_down_sample_count):
                        __detail_position = __i + __j
                        __abs_mean_fdt = np.mean(np.abs(signal.data[
                                                    __detail_position + __reqa_down_to_down_sample_count + __fraim_delay_sample_count//10:
                                                    __detail_position + __reqa_down_to_down_sample_count + __fraim_delay_sample_count - __fraim_delay_sample_count//20
                                                    ]))
                        __abs_mean_atqa = np.mean(np.abs(signal.data[
                                                    __detail_position + __reqa_down_to_down_sample_count + __fraim_delay_sample_count - __ataq_sample_count//608:
                                                    __detail_position + __reqa_down_to_down_sample_count + __fraim_delay_sample_count + __ataq_sample_count//38 + __ataq_sample_count//608
                                                    ]))

                        __find_max = __abs_mean_atqa - __abs_mean_fdt
                        # print(f'{__current_abs_mean}, {__abs_mean}')
                        if __found_max[1] < __find_max and self.have_reqa_ataq(signal, __detail_position):
                            __found_max = __detail_position, __find_max
                    if self.have_reqa_ataq(signal, __found_max[0]):
                        __reqa_start_position.append(__found_max[0])
                    __i += __found_max[0] + __reqa_to_ataq_sample_count  # 다음 섹션 이동
                else:
                    __i += __speed
        return __reqa_start_position

    def have_reqa_ataq(self, signal:Signal, reqa_start_position: int):
        __FRAIM_DELAY_SAMPLE = int(signal.samplerate * self.iso14443.fraim_delay_time)
        __ATQA_SAMPLE = int(signal.samplerate * self.iso14443.atqa_time)
        __REQA_SAMPLE = int(signal.samplerate * self.iso14443.reqa_down_to_down_time)

        __FDT_MARGIN = int(__FRAIM_DELAY_SAMPLE * 0.05)  # 5% 여유
        __FDT_TIME_MAX = np.max(signal.data[
                                reqa_start_position + __REQA_SAMPLE + __FDT_MARGIN:
                                reqa_start_position + __REQA_SAMPLE + __FRAIM_DELAY_SAMPLE - __FDT_MARGIN
                                ])

        __ATQA_SAMPLE_FROM_REQA = reqa_start_position + int(signal.samplerate * (self.iso14443.reqa_down_to_down_time + self.iso14443.fraim_delay_time))
        __ATQA = signal.data[__ATQA_SAMPLE_FROM_REQA:__ATQA_SAMPLE_FROM_REQA + __ATQA_SAMPLE]
        return __FDT_TIME_MAX * 2 < np.max(__ATQA)

    def get_current_label_list(self):
        __current_date = self.signal_data
        __current_label_list = []
        for data in __current_date:
            if data.label not in __current_label_list:
                __current_label_list.append(data.label)
        print(__current_label_list)
        return __current_label_list

    def get_sorted_list(self, label):
        __current_data = self.signal_data
        __sorted_list = []
        for data in __current_data:
            if data.label == label:
                __sorted_list.append(data)
        return __sorted_list

    def get_normalized_data(self):
        __current_data = self.signal_data
        __normalized_data = []
        for signal in __current_data:
            # __data = (signal.data-np.min(signal.data))/np.max(signal.data-np.min(signal.data))
            __data = signal.data / np.max(signal.data)
            __normalized_data.append(Signal(signal.samplerate, __data, signal.file_path))
        self.signal_data = __normalized_data


"""
    def get_enveloped_normalized_data(self, start=0, stop=93, cut=True):
        __first_signal_data_for_ref = self.signal_data[0]
        __sync_value = []
        for data in self.signal_data[1:]:
            __sync_x, __sync_y = get_sync_sample_position(
                first_signal=__first_signal_data_for_ref,
                second_signal=data
            )

            if len(__sync_value) == 0:
                __sync_value.append(__sync_x)
                __sync_value.append(__sync_y)
            else:
                __sync_value.append(__sync_y)

        __enveloped_value = []
        for i in range(len(self.signal_data)):
            __signal_data = self.signal_data[i]
            __bandpassed_data = __signal_data.butter_bandpass_filter(
                lowcut=self.iso14443.picc_freq/2,
                highcut=self.iso14443.picc_freq*3/2,
                order=1)

            __hilbert_envelope = Signal.hilbert_envelope(Signal(data=__bandpassed_data,
                                                                file_path=__signal_data.file_path,
                                                                samplerate=__signal_data.samplerate))
            __ratio = (
                    max(Signal.hilbert_envelope(__first_signal_data_for_ref)[__sync_value[0]+start:__sync_value[0]+stop])/
                    max(__hilbert_envelope[__sync_value[i]+start:__sync_value[i]+stop])
            )

            if cut:
                __enveloped_value.append(
                    Signal(
                        data=__hilbert_envelope[__sync_value[i]+start:__sync_value[i]+stop] * __ratio,
                        file_path=__signal_data.file_path,
                        samplerate=__signal_data.samplerate)
                    )
            else:
                __enveloped_value.append(
                    Signal(
                        data=__hilbert_envelope[__sync_value[i] - min(__sync_value):] * __ratio,
                        file_path=__signal_data.file_path,
                        samplerate=__signal_data.samplerate)
                )
        return __enveloped_value

    def get_normalized_amp_data(self):
        __sync_value = []
        __sync_ratio_value = []
        for data in self.signal_data[1:]:
            __first_signal_data_for_ref = self.signal_data[0]
            __ratio = default_normalized_signal(
                first_signal=__first_signal_data_for_ref,
                second_signal=data)
            __sync_x, __sync_y = get_sync_sample_position(
                first_signal=__first_signal_data_for_ref,
                second_signal=data)
            if len(__sync_value) == 0 and len(__sync_ratio_value) == 0:
                __sync_value.append(__sync_x)
                __sync_value.append(__sync_y)
                __sync_ratio_value.append(1)
                __sync_ratio_value.append(__ratio)
            else:
                __sync_value.append(__sync_y)
                __sync_ratio_value.append(__ratio)

        __sync_data = []
        for i in range(len(self.signal_data)):
            __sync_data.append(
                Signal(
                    data=self.signal_data[__sync_value[i] - min(__sync_value):] * __sync_ratio_value[i],
                    samplerate=self.signal_data[i].samplerate,
                    file_path=self.signal_data[i].file_path
                )
            )

        return __sync_ratio_value
"""
