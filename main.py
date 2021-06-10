import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, butter

FAST_MODE = False

FIRST_SIGNAL_SAMPLE_RATE, FIRST_ORIGINAL_SIGNAL = wavfile.read('./wavFiles/card_su.wav')
SECOND_SIGNAL_SAMPLE_RATE, SECOND_ORIGINAL_SIGNAL = wavfile.read('./wavFiles/card_ho.wav')

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
    filtered_first_signal = butter_bandpass_filter(first_signal_data, 847500 - 847500 / 2, 847500 + 847500 / 4,
                                                   FIRST_SIGNAL_SAMPLE_RATE, 1)
    filtered_second_signal = butter_bandpass_filter(second_signal_data, 847500 - 847500 / 2, 847500 + 847500 / 4,
                                                    SECOND_SIGNAL_SAMPLE_RATE, 1)

    first_ATQA_sample_position = 0
    second_ATQA_sample_posion = 0

    if FAST_MODE:
        __find_max_value_one = 0
        for i in range(len(filtered_first_signal) - t_ATQA_one):
            if __find_max_value_one < np.abs(filtered_first_signal[i:i + t_ATQA_one - 1]).sum():
                first_ATQA_sample_position = i
                __find_max_value_one = np.abs(filtered_first_signal[i:i + t_ATQA_one - 1]).sum()

        __find_max_value_two = 0
        for j in range(len(filtered_second_signal) - t_ATQA_two):
            if __find_max_value_two < np.abs(filtered_second_signal[j:j + t_ATQA_two - 1]).sum():
                second_ATQA_sample_posion = j
                __find_max_value_two = np.abs(filtered_second_signal[j:j + t_ATQA_two - 1]).sum()

    else:
        __value = 9999999999999
        __find_max_value_one = 0
        for first_signal_sample in range(len(filtered_first_signal) - t_ATQA_one):
            if __find_max_value_one < np.abs(filtered_first_signal[first_signal_sample:first_signal_sample + t_ATQA_one - 1]).sum():
                first_ATQA_sample_position = first_signal_sample
                __find_max_value_one = np.abs(filtered_first_signal[first_signal_sample:first_signal_sample + t_ATQA_one - 1]).sum()

                for second_signal_sample in range(len(filtered_second_signal) - t_ATQA_two):
                        a = (np.abs(filtered_first_signal[first_signal_sample:first_signal_sample + t_ATQA_one - 1]) -
                             np.abs(filtered_second_signal[second_signal_sample:second_signal_sample + t_ATQA_two - 1])
                             ).sum()
                        if __value > a:
                            __value = a
                            second_ATQA_sample_posion = second_signal_sample
                            print('...')

    print(first_ATQA_sample_position, second_ATQA_sample_posion)
    return first_ATQA_sample_position, second_ATQA_sample_posion

# def get944microsec(data_one: np.array, data_two: np.array):
#     find_max_value_one = 0
#     first_ATQA_sample_one = 0
#     for i in range(len(data_one) - t_ATQA_one):
#         if find_max_value_one < data_one[i:i + t_ATQA_one].sum():
#             first_ATQA_sample_one = i
#             find_max_value_one = data_one[i:i + t_ATQA_one].sum()
#
#     data_one = data_one[first_ATQA_sample_one - t_0_one:first_ATQA_sample_one + t_ATQA_one + t_0_one]
#
#     a_one = data_two[:round(t_0_one / 2)]
#     b_one = data_two[round(t_0_one / 2) + t_ATQA_one + t_0_one:]
#
#     lean_one = (np.mean(b_one) - np.mean(a_one) / t_ATQA_one)
#
#     for ii in range(len(data_one)):
#         data_one[ii] + ii * lean_one
#
#     find_max_value_two = 0
#     first_ATQA_sample_two = 0
#     for j in range(len(data_two) - t_ATQA_two):
#         if find_max_value_two < data_two[j:j + t_ATQA_two].sum():
#             first_ATQA_sample_two = j
#             find_max_value_two = data_two[j:j + t_ATQA_two].sum()
#
#     data_two = data_two[first_ATQA_sample_two - t_0_two:first_ATQA_sample_two + t_ATQA_two + t_0_two]
#
#     a_two = data_two[:round(t_0_two / 2)]
#     b_two = data_two[round(t_0_two / 2) + t_ATQA_two + t_0_two:]
#
#     lean_two = round((np.mean(b_two) - np.mean(a_two)) / t_ATQA_two)
#
#     for jj in range(len(data_two)):
#         data_two[jj] + jj * lean_two
#
#     print(len(data_one))
#     print(len(data_two))
#
#     _min_sync = np.mean(b_one) - np.mean(b_two)
#     data_two = data_two + _min_sync
#     _max_ratio = np.max(data_one) / np.max(data_two)
#     data_two = data_two * _max_ratio
#
#     find_sync_sample = 0
#     find_min_value = find_max_value_one
#     for k in range(t_0_two * 2):
#         x = data_two[k + t_0_two:k + t_ATQA_two] - data_one[t_0_one:t_ATQA_one]
#         if find_min_value > sum(np.abs(x)):
#             find_sync_sample = k
#             find_min_value = sum(np.abs(x))
#
#     data_two = data_two[find_sync_sample:]
#     if find_sync_sample == 0:
#         find_min_value = find_max_value_two
#         for k in range(t_0_one * 2):
#             x = data_one[k + t_0_one:k + t_ATQA_one] - data_two[t_0_one:t_ATQA_two]
#             if find_min_value > sum(np.abs(x)):
#                 find_sync_sample = k
#                 find_min_value = sum(np.abs(x))
#
#         data_one = data_one[find_sync_sample:]
#
#     print(find_sync_sample)
#
#     # find_sink_value: float = 99999999999999999
#     # sample_sink = 0
#     # for k in range(round(sample_one/2)):
#     #     a = data_one[t_0_one:sample_one] - data_one[k:round(sample_one / 2) + k]
#     #     if find_sink_value > abs(a).sum():
#     #         find_sink_value = abs(a).sum()
#     #         sample_sink = k
#
#     return data_one, data_two


filtered_first_signal = butter_bandpass_filter(FIRST_ORIGINAL_SIGNAL, 847500 - 847500 / 2, 847500 + 847500 / 2,
                                               FIRST_SIGNAL_SAMPLE_RATE, 2)
filtered_second_signal = butter_bandpass_filter(SECOND_ORIGINAL_SIGNAL, 847500 - 847500 / 2, 847500 + 847500 / 2,
                                                SECOND_SIGNAL_SAMPLE_RATE, 2)

sync_first_sample, sync_second_sample = get_sync_sample_position(FIRST_ORIGINAL_SIGNAL, SECOND_ORIGINAL_SIGNAL)

#TODO: 노멀라이즈 함수 손좀 봐...
def get_normalized_signal(first_signal: np.array, second_signal: np.array, first_sink: int, second_sink: int):
    x2 = 1 + (np.abs(first_signal[first_sink:first_sink + 6 - 1]) -
         np.abs(second_signal[second_sink:second_sink + 6 -1])
         ).mean() / np.abs(second_signal[second_sink:second_sink + 6 -1]).mean()
    return first_signal[first_sink - 10:], second_signal[second_sink - 10:] * x2


def show_plot(first_signal, second_signal):
    plt.plot(first_signal, color='red')
    plt.plot(second_signal, color='blue')
    plt.axis('tight')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.grid(True)


plt.figure(1)

plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(t_0_one))
show_plot(filtered_first_signal[sync_first_sample:], filtered_second_signal[sync_second_sample:])
show_plot(FIRST_ORIGINAL_SIGNAL[sync_first_sample:], SECOND_ORIGINAL_SIGNAL[sync_second_sample:])

plt.figure(2)
ns1,ns2 = get_normalized_signal(filtered_first_signal,filtered_second_signal,sync_first_sample,sync_second_sample)
show_plot(ns1, ns2)
plt.show()
