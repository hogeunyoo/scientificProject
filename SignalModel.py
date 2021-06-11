from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert
import numpy as np

PICC_FREQ = 847500 # Hz


class Signal:
    def __init__(self, samplerate, data, lable):
        self.samplerate = samplerate
        self.data = data
        self.lable = lable

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

#
# def get_envelop_signal(first_signal: np.array, second_signal: np.array, first_sink: int, second_sink: int):
#     first_signal = get_envelop(first_signal)
#     second_signal = get_envelop(second_signal)
#     __ratio = first_signal[first_sink:first_sink + round(t_half_ATQA_one)].max() / second_signal[
#                                                                                    second_sink:second_sink + round(
#                                                                                        t_half_ATQA_two)].max()
#     second_signal = second_signal * __ratio
#
#     return first_signal[first_sink - 20:first_sink + t_half_ATQA_one + 20], second_signal[
#                                                                             second_sink - 20:second_sink + t_half_ATQA_two + 20]
#

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

    def read_wav_file(self, file):
        __current_signal = Signal(wavfile.read(file)[0], wavfile.read(file)[1], file)

        if len(self.signal_data) != 0:
            if self.signal_data[0].samplerate == __current_signal.samplerate:
                self.signal_data.append(__current_signal)
            else:
                print('ERROR : 신호 파일 불량')
        else:
            self.signal_data.append(__current_signal)
