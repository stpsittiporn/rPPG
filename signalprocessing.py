import cv2
import numpy as np
import time
from scipy import signal
from sklearn.decomposition import FastICA
from pylab import *
from skimage import data, io, color


class Signal_processing():
    def __init__(self):
        self.a = 1

    def extract_color(self, ROI1,ROI2,ROI3):

        avg1 = np.mean(ROI1[:, :, 1])
        avg2 = np.mean(ROI2[:, :, 1])
        avg3 = np.mean(ROI3[:, :, 1])
        avg = [avg1, avg2, avg3]
        Raw_rppg = np.mean(avg)

        return Raw_rppg

    def normalization(self, data_buffer):

        normalized_data = (data_buffer - np.mean(data_buffer)) / np.std(data_buffer)
        return normalized_data

    def signal_detrending(self, data_buffer):

        detrended_data = signal.detrend(data_buffer)
        return detrended_data

    def fft(self, data_buffer, fps):

        data = data_buffer
        L = len(data)
        resolution = (float(fps) / L) * 60
        freqs = float(fps) / L * np.arange(L / 2 + 1)
        freqs_in_minute = 60. * freqs
        fft = np.fft.rfft(data)
        psd = np.abs(fft) ** 2

        return psd, freqs_in_minute, resolution

    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, data_buffer)

        return filtered_data












