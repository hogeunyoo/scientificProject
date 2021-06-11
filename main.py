import sys
from scipy.io import wavfile

from SignalModel import SignalModel
from MainViewController import MainViewController


FIRST_SIGNAL_SAMPLE_RATE, FIRST_ORIGINAL_SIGNAL = wavfile.read('./wavFiles/ho_id_1.wav')
SECOND_SIGNAL_SAMPLE_RATE, SECOND_ORIGINAL_SIGNAL = wavfile.read('./wavFiles/ok_id_0.wav')


class MainView:
    def __init__(self):
        self.commend = ''
        self.mainViewController = MainViewController()
        self.signalModel = SignalModel()

    def firstView(self):
        print('welcom')
        a = str(sys.stdin.readline())
        print(a)
        if a == 'start\n':
            print("temp_start")
            self.mainViewController.temp_start()
        elif a == 'temp_start':
            print("pass")
        else:
            print("?")


    def show_plot(self, signal_data):
        pass


mainView = MainView()
mainView.firstView()
