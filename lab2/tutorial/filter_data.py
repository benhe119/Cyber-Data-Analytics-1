import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.api import qqplot
import scipy.fftpack as fftpk
import scipy.signal as sgnl

def filter(list, df):

    idxl = 0;

    # loop through list of fields of type L_T1-7
    for fieldname in list:

        idxl = idxl + 1

        df_field = df[[fieldname]]
        field_matrix = df_field.values;

        Fs = 1.0/3600;                      # sampling rate [Hz]
        Ts = 1.0/Fs;                        # sampling period [sec]
        datalength = field_matrix.size      # data length [samples]
        t = np.arange(0,Ts*datalength,Ts)   # time vector - spread

        # print 'Fs = ' + str(Fs)
        # print 'Ts = ' + str(Ts)
        # print 'Datalength = ' + str(datalength)
        # print 't length = ' + str(t.size)

        # assign signal to y (time domain) and reshape (NB!) then detrend (remove mean)
        y_time = field_matrix;
        y_time = np.asarray(y_time).reshape(-1)
        # print 'y mean = ' + str(y_time.mean())
        # print 'y var = ' + str(y_time.var())
        y_time = (y_time - y_time.mean()) #/(y_time.var())

        # print 't length = ' + str(t.size)
        # print 'y length = ' + str(y.size)

        # datalength = y_time.size          # length of the signal
        idx = np.arange(datalength)         # i^th element list
        T = datalength*Ts                   # total length = datalength*period
        # print T
        # print idx


        frq = idx/T                         # two sides frequency range
        frq_with_neg = frq;                 # keep negative side for later
        frq = frq[range(datalength/2)]      # one side frequency range

        # print frq_with_neg                  # frequency [0 - Fs] upto sampling freq
        # print frq                           # frequency [0 - Fs/2] (Nyquist Freq)
        #
        # print frq.size

        Y_freq = np.fft.fft(y_time)/datalength   # fft computing and normalization
        Y_with_neg = Y_freq;
        Y_freq = Y_freq[range(datalength/2)]

        # print Y_with_neg
        # print Y_freq
        #
        #
        # print 'Y_with_neg = ' + str(Y_with_neg.size)      # 8600
        # print 'frq_with_neg = ' + str(frq_with_neg.size)  # 8600
        #
        # print min(frq_with_neg)
        # print max(frq_with_neg)

        ### PLOT FIGURE
        plt.figure(num=1)

        plt.subplot(211)
        plt.plot(t,y_time)
        plt.xlabel('Time [seconds]')
        plt.ylabel('Amplitude')

        plt.subplot(212)
        plt.plot(frq,abs(Y_freq),'r')
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|Y(freq)|')

        plt.grid()
        plt.show()

        # print Y_with_neg.size

        #Y_with_neg[Y_with_neg<0.02] = 0

        # cutoff frequency [Hz?]
        wn = 100;
        Y_with_neg[wn:-wn] = 0

        ## TODO RATHER USE A FILTER SO THAT TEST DATA CAN BE FILTERED TOO

        # print Y_with_neg.size
        Yinv = np.fft.ifft(Y_with_neg)*datalength

        # print 'Yinv length = ' + str(Yinv.size)
        # print 't length = ' + str(t.size)

        df[fieldname] = np.real((Yinv))

    return df

def filter_F_PUxx(list):


    return 0

def filter_S_PUxx(list):


    return 0

def filter_P_Jxxx(list):


    return 0
