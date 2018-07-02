import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.api import qqplot
import scipy.fftpack as fftpk
import scipy.signal as sgnl

import qin_sax as sax
import qin_segmentation

def discretizeBinary(list, dataframe):

    # idxl = 0;

    # loop through list of fields of type L_T1-7
    # for fieldname in list:

    idxl = 1; # increment each for loop

    fieldname = "L_T1"

    df_field = dataframe[[fieldname]]
    field_matrix = df_field.values;

    Fs = 1.0/3600;                      # sampling rate [Hz]
    Ts = 1.0/Fs;

    datalength = field_matrix.size      # data length [samples]
    t = np.arange(0,Ts*datalength,Ts)   # time vector - spread

    # discretize data
    Ydiscrete = np.real(np.copy(field_matrix))
    print Ydiscrete
    midval = Ydiscrete.mean()
    print midval
    print max(Ydiscrete)
    print min(Ydiscrete)

    # Ydiscrete[Ydiscrete<midval] = 1;
    #Ydiscrete[Ydiscrete>midval] = -1;

    for idx in np.arange(0,Ydiscrete.size,1):
        if Ydiscrete[idx] > midval:
            Ydiscrete[idx] = 1
        else:
            Ydiscrete[idx] = -1

    # plt.figure(idxl)
    # plt.title('Time domain L_T' + str(idxl))
    # plt.plot(t,field_matrix,label='y filtered')
    # # Y_with_neg = Y_with_neg[range(datalength/2)]
    # # plt.plot(t,y_time,label='y original')
    # plt.plot(t,Ydiscrete,label='y discretized')
    # plt.xlabel('Time [seconds]')
    # plt.ylabel('Signal')
    # plt.legend()
    # plt.show()

    dataframe[fieldname] = Ydiscrete

    return # void

def discretizeSAX(fieldname, dataframe):

    #fieldname = 'L_T2'

    df_field = dataframe[[fieldname]]
    field_matrix = df_field.values;

    Fs = 1.0/3600;                      # sampling rate [Hz]
    Ts = 1.0/Fs;

    datalength = field_matrix.size      # data length [samples]
    t = np.arange(0,Ts*datalength,Ts)   # time vector - spread



    #print field_matrix[:-1]

    field_matrix = field_matrix[:-1]

    y_time = field_matrix;
    y_time = np.asarray(y_time).reshape(-1)

    #set the wordSize to the exact same size as the dataframe, so they overlap
    saxobject = sax.SAX(wordSize = len(dataframe), alphabetSize = 10, epsilon = 1e-6)

    letterseq = saxobject.to_letter_rep(np.transpose(y_time))
    # print letterseq[0]
    #
    # print letterseq[1]

    xlist, ylist = saxobject.get_results(letterseq)

    #print xlist

    #uncomment to show plot
    saxobject.plot_results(x_axis_list=xlist,y_axis_list=ylist)

    return xlist, ylist # void

 # void
