from pathlib import Path, PurePath
from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert
import numpy as np

PICC_FREQ = 847500 # Hz


class Signal:
    def __init__(self, samplerate, data, file_path):
        self.samplerate = samplerate
        self.data = data
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


class SignalModel:
    def __init__(self):
        self.signal_data = []

    def open_wav_folders(self, folder_path):
        data_dir = Path(folder_path)
        if data_dir.is_dir():
            for file_path in data_dir.glob('*.wav'):
                self.read_wav_file(file_path)
        else:
            print("ERROR : Please input wav files")

    def read_wav_file(self, file_path):
        __current_signal = Signal(wavfile.read(file_path)[0], wavfile.read(file_path)[1], file_path)

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
        __file_path = PurePath(signal.label + str(__data_num) + '.wav')
        while (dir_path/__file_path).exists():
            __data_num += 1
            __file_path = PurePath(signal.label + str(__data_num) + '.wav')
        print(dir_path/__file_path)
        print(signal.data)
        wavfile.write(dir_path/__file_path, signal.samplerate, signal.data.astype(np.int16))

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

    def get_enveloped_normalized_data(self, start=0, stop=93):
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
                lowcut=PICC_FREQ - PICC_FREQ / 2,
                highcut=PICC_FREQ + PICC_FREQ / 2,
                order=1)

            __hilbert_envelope = Signal.hilbert_envelope(Signal(data=__bandpassed_data,
                                                                file_path=__signal_data.file_path,
                                                                samplerate=__signal_data.samplerate))
            __ratio = (
                    max(Signal.hilbert_envelope(__first_signal_data_for_ref)[__sync_value[0]+start:__sync_value[0]+stop])/
                    max(__hilbert_envelope[__sync_value[i]+start:__sync_value[i]+stop])
            )

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
