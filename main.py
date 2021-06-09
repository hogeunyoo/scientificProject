import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

first_data_sample_rate, first_data = wavfile.read('./wavFiles/card_su.wav')
second_data_sample_rate, second_data = wavfile.read('./wavFiles/card_ho.wav')

# times = np.arange(len(first_data)) / float(first_data_sample_rate / 1000)


def get944microsec(data_one: np.array, sample_rate_one: np.array, data_two: np.array, sample_rate_two: np.array):

    t_ATQA_one = round(
        1 / 847500 * 8 * sample_rate_one * 18.5)  # 1/f_s * 8 periods of the subcarrier * sample rate (9.44micro_sec)
    t_0_one = round(
        1 / 847500 * 8 * sample_rate_one)

    t_ATQA_two = round(
        1 / 847500 * 8 * sample_rate_two * 18.5)  # 1/f_s * 8 periods of the subcarrier * sample rate (9.44micro_sec)
    t_0_two = round(
        1 / 847500 * 8 * sample_rate_two)

    find_max_value_one = 0
    first_ATQA_sample_one = 0
    for i in range(len(data_one) - t_ATQA_one):
        if find_max_value_one < data_one[i:i + t_ATQA_one].sum():
            first_ATQA_sample_one = i
            find_max_value_one = data_one[i:i + t_ATQA_one].sum()

    data_one = data_one[first_ATQA_sample_one-t_0_one:first_ATQA_sample_one+t_ATQA_one+t_0_one]

    a_one = data_two[:round(t_0_one / 2)]
    b_one = data_two[round(t_0_one / 2) + t_ATQA_one + t_0_one:]

    lean_one = (np.mean(b_one) - np.mean(a_one) / t_ATQA_one)

    for ii in range(len(data_one)):
        data_one[ii] + ii * lean_one

    find_max_value_two = 0
    first_ATQA_sample_two = 0
    for j in range(len(data_two) - t_ATQA_two):
        if find_max_value_two < data_two[j:j + t_ATQA_two].sum():
            first_ATQA_sample_two = j
            find_max_value_two = data_two[j:j + t_ATQA_two].sum()

    data_two = data_two[first_ATQA_sample_two - t_0_two:first_ATQA_sample_two + t_ATQA_two + t_0_two]

    a_two = data_two[:round(t_0_two/2)]
    b_two = data_two[round(t_0_two/2) + t_ATQA_two + t_0_two:]

    lean_two = round((np.mean(b_two)-np.mean(a_two))/t_ATQA_two)

    for jj in range(len(data_two)):
        data_two[jj] + jj * lean_two

    print(len(data_one))
    print(len(data_two))

    _min_sync = np.mean(b_one) - np.mean(b_two)
    data_two = data_two + _min_sync
    _max_ratio = np.max(data_one) / np.max(data_two)
    data_two = data_two * _max_ratio

    find_sync_sample = 0
    find_min_value = find_max_value_one
    for k in range(t_0_two*2):
        x = data_two[k+t_0_two:k + t_ATQA_two] - data_one[t_0_one:t_ATQA_one]
        if find_min_value > sum(np.abs(x)):
            find_sync_sample = k
            find_min_value = sum(np.abs(x))

    data_two = data_two[find_sync_sample:]
    if find_sync_sample == 0:
        find_min_value = find_max_value_two
        for k in range(t_0_one * 2):
            x = data_one[k + t_0_one:k + t_ATQA_one] - data_two[t_0_one:t_ATQA_two]
            if find_min_value > sum(np.abs(x)):
                find_sync_sample = k
                find_min_value = sum(np.abs(x))

        data_one = data_one[find_sync_sample:]


    print(find_sync_sample)

    # find_sink_value: float = 99999999999999999
    # sample_sink = 0
    # for k in range(round(sample_one/2)):
    #     a = data_one[t_0_one:sample_one] - data_one[k:round(sample_one / 2) + k]
    #     if find_sink_value > abs(a).sum():
    #         find_sink_value = abs(a).sum()
    #         sample_sink = k


    plt.figure(figsize=(20, 5))
    plt.plot(data_one, color='red')
    plt.plot(data_two, color='blue')
    plt.xlim(0, t_ATQA_one)
    plt.ylim(0, max(data_one))
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.show()

    return first_ATQA_sample_one, t_ATQA_one


x, t = get944microsec(first_data, first_data_sample_rate, second_data, second_data_sample_rate)
