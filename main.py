import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, butter, hilbert

FIRST_SIGNAL_SAMPLE_RATE, FIRST_ORIGINAL_SIGNAL = wavfile.read('./wavFiles/ho_id_1.wav')
SECOND_SIGNAL_SAMPLE_RATE, SECOND_ORIGINAL_SIGNAL = wavfile.read('./wavFiles/ok_id_0.wav')


def get_envelop(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def show_plot(first_signal, second_signal):
    plt.plot(first_signal, color='red')
    plt.plot(second_signal, color='blue')
    plt.axis('tight')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.grid(True)

if FIRST_SIGNAL_SAMPLE_RATE != SECOND_SIGNAL_SAMPLE_RATE:
    print("SAMPLE RATE ERROR")

t_ATQA_one = round(
    1 / 847500 * 8 * FIRST_SIGNAL_SAMPLE_RATE * 18.5)
t_0_one = round(
    1 / 847500 * 8 * FIRST_SIGNAL_SAMPLE_RATE)

t_half_ATQA_one = round(
    1 / 847500 * 4 * FIRST_SIGNAL_SAMPLE_RATE)

t_ATQA_two = round(
    1 / 847500 * 8 * SECOND_SIGNAL_SAMPLE_RATE * 18.5)
t_0_two = round(
    1 / 847500 * 8 * SECOND_SIGNAL_SAMPLE_RATE)
t_half_ATQA_two = round(
    1 / 847500 * 4 * SECOND_SIGNAL_SAMPLE_RATE)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_sync_sample_position(first_signal_data: np.array, second_signal_data: np.array):
    filtered_first_signal = butter_bandpass_filter(first_signal_data, 847500 - 847500 / 2, 847500 + 847500 / 1,
                                                   FIRST_SIGNAL_SAMPLE_RATE, 1)
    filtered_second_signal = butter_bandpass_filter(second_signal_data, 847500 - 847500 / 2, 847500 + 847500 / 1,
                                                    SECOND_SIGNAL_SAMPLE_RATE, 1)

    first_ATQA_sample_position = 0
    second_ATQA_sample_posion = 0

    __find_max_value_one = 0
    __find_max_value_two = 0

    __value = 100000000

    for i in range(len(filtered_first_signal) - t_ATQA_one):
        if __find_max_value_one < np.abs(filtered_first_signal[i:i + t_ATQA_one - 1]).sum():
            first_ATQA_sample_position = i
            __find_max_value_one = np.abs(filtered_first_signal[i:i + t_ATQA_one - 1]).sum()

    for j in range(len(filtered_second_signal) - t_ATQA_two):
        if __find_max_value_two < np.abs(filtered_second_signal[j:j + t_ATQA_two - 1]).sum():
            __find_max_value_two = np.abs(filtered_second_signal[j:j + t_ATQA_two - 1]).sum()

    __ratio = __find_max_value_one / __find_max_value_two

    filtered_second_signal = filtered_second_signal * __ratio

    for second_signal_sample in range(len(filtered_second_signal) - t_ATQA_two):
        a = (filtered_first_signal[first_ATQA_sample_position:first_ATQA_sample_position + t_ATQA_one - 1] -
             filtered_second_signal[second_signal_sample:second_signal_sample + t_ATQA_two - 1]
             ).std()
        if __value > a:
            __value = a
            second_ATQA_sample_posion = second_signal_sample
            print(a)

    plt.figure(4)
    show_plot(filtered_first_signal[first_ATQA_sample_position - 20:],
              filtered_second_signal[second_ATQA_sample_posion - 20:])
    print(first_ATQA_sample_position, second_ATQA_sample_posion)
    return first_ATQA_sample_position, second_ATQA_sample_posion


filtered_first_signal = butter_bandpass_filter(FIRST_ORIGINAL_SIGNAL, 847500 - 847500 / 2, 847500 + 847500 / 2,
                                               FIRST_SIGNAL_SAMPLE_RATE, 2)
filtered_second_signal = butter_bandpass_filter(SECOND_ORIGINAL_SIGNAL, 847500 - 847500 / 2, 847500 + 847500 / 2,
                                                SECOND_SIGNAL_SAMPLE_RATE, 2)

sync_first_sample, sync_second_sample = get_sync_sample_position(FIRST_ORIGINAL_SIGNAL, SECOND_ORIGINAL_SIGNAL)


def get_envelop_signal(first_signal: np.array, second_signal: np.array, first_sink: int, second_sink: int):
    first_signal = get_envelop(first_signal)
    second_signal = get_envelop(second_signal)
    __ratio = first_signal[first_sink:first_sink + round(t_half_ATQA_one)].max() / second_signal[
                                                                                       second_sink:second_sink + round(
                                                                                           t_half_ATQA_two)].max()
    second_signal = second_signal * __ratio

    return first_signal[first_sink - 20:first_sink + t_half_ATQA_one + 20], second_signal[
                                                                            second_sink - 20:second_sink + t_half_ATQA_two + 20]


def default_normalized_signal(original_first_signal: np.array, original_second_signal: np.array, accuracy=1000):
    __first_sink, __second_sink = get_sync_sample_position(original_first_signal, original_second_signal)
    filtered_first_signal = butter_bandpass_filter(original_first_signal, 847500 - 847500 / 2, 847500 + 847500 / 1,
                                                   FIRST_SIGNAL_SAMPLE_RATE, 1)
    filtered_second_signal = butter_bandpass_filter(original_second_signal, 847500 - 847500 / 2, 847500 + 847500 / 1,
                                                    SECOND_SIGNAL_SAMPLE_RATE, 1)

    __ratio = 0
    __find_min_value = 999999999
    for amp in range(accuracy):
        if __find_min_value > (filtered_first_signal[__first_sink:__first_sink + t_ATQA_one] - filtered_second_signal[
                                                                                               __second_sink:__second_sink + t_ATQA_two] * 10 * amp / accuracy).std():
            __find_min_value = (filtered_first_signal[__first_sink:__first_sink + t_ATQA_one] - filtered_second_signal[
                                                                                                __second_sink:__second_sink + t_ATQA_two] * 10 * amp / accuracy).std()
            __ratio = 10 * amp / accuracy

    plt.figure(3)
    show_plot(filtered_first_signal[__first_sink - 20:], filtered_second_signal[__second_sink - 20:] * __ratio)
    print('ratio' + str(__ratio))

    plt.figure(6)
    show_plot(get_envelop(filtered_first_signal[__first_sink - 20:]),
              get_envelop(filtered_second_signal[__second_sink - 20:] * __ratio))
    return filtered_first_signal[__first_sink:], filtered_second_signal[__second_sink:] * __ratio


plt.figure(1)

plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(t_0_one))
show_plot(filtered_first_signal[sync_first_sample:], filtered_second_signal[sync_second_sample:])
show_plot(FIRST_ORIGINAL_SIGNAL[sync_first_sample:], SECOND_ORIGINAL_SIGNAL[sync_second_sample:])

plt.figure(2)
ns1, ns2 = get_envelop_signal(filtered_first_signal, filtered_second_signal, sync_first_sample, sync_second_sample)
show_plot(ns1, ns2)


default_normalized_signal(FIRST_ORIGINAL_SIGNAL, SECOND_ORIGINAL_SIGNAL, 1000)

plt.show()
