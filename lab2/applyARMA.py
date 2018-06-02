"""
apply the ARMA prediction to the dataset

1) FFT analysis to remove signal noise
2) Discretization of filtered signals

# tutorial from https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/#comments
# download data csv from http://www.sidc.be/silso/INFO/snytotcsv.php
"""
### IMPORT PACKAGES ###
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.api import qqplot
import scipy.fftpack as fftpk
import scipy.signal as sgnl
from matplotlib import pyplot as plt
from filter_data import filter, filter_F_PUxx, filter_S_PUxx, filter_P_Jxxx
# from discretize_data import discretizeBinary, discretizeSAX

from ARMA import fitARMA
from general_functions import standardize_dataset, standardize_dataset_train_2, standardize_dataset_test

# For trainingset2
L_Tselect = ["L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T7"]
F_PUselect = ["F_PU1", "F_PU2", "F_PU6", "F_PU7", "F_PU8", "F_PU10", "F_PU11"]
S_PUselect = ["S_PU2", "S_PU6", "S_PU7", "S_PU8", "S_PU10", "S_PU11"]
V_Pselect = ["F_V2", "S_V2"]
P_Jselect = ['P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

listToDelete = ["L_T6", "S_PU1","S_PU3","S_PU4","S_PU5","S_PU9","F_PU3","F_PU4","F_PU5","F_PU9"]

def getBinaryDF(inputDF):

    binaryDF = inputDF.copy(deep=True)

    print 'input'
    print inputDF.describe()

    # Remove signals that do not contain exciting information
    for deletefield in listToDelete:
        del binaryDF[deletefield]

    # Apply binary selection to Tank Levels
    for fieldname in L_Tselect:
        binaryDF[fieldname] = fitARMA(inputDF,fieldname,p=2,q=2)

    # Apply binary selection to Pump Flow Rate
    for fieldname in F_PUselect:
        binaryDF[fieldname] = fitARMA(inputDF,fieldname,p=2,q=2)

    # Apply binary selection to Pump Switch Signals
    for fieldname in S_PUselect:
        binaryDF[fieldname] = fitARMA(inputDF,fieldname,p=2,q=2)

    # Apply binary selection to Valve
    for fieldname in V_Pselect:
        binaryDF[fieldname] = fitARMA(inputDF,fieldname,p=2,q=2)

    # Apply binary selection to PLC Signals
    for fieldname in P_Jselect:
        binaryDF[fieldname] = fitARMA(inputDF,fieldname,p=2,q=2)

    print '\n binary'
    print binaryDF.describe()

    return binaryDF

def obtainFinalPrediction(binaryDF):

    # Create an empty dataframe which will store the predictions
    dfPrediction = binaryDF["ATT_FLAG"].copy(deep=True)

    # set all values to zero for dfPrediction
    dfPrediction = (dfPrediction > 2).astype(int)

    print '\ndf prediction ...\n'

    # Check for scenario 1
    dfPrediction.loc[binaryDF["L_T1"] == 1] = 1

    # Check for scenario 2


    # Check for scenario 3


    # Check for scenario 4


    # Check for scenario 5
    dfPrediction.loc[binaryDF["S_PU11"] == 1] = 1

    # Check for scenario 6


    # Print Performance to console
    # tp = 0
	# fp = 0
	# tn = 0
	# fn = 0
	# for i in range(test_normalized.shape[0]):
	#     if(labels[i] == 1 and na[i] == 1):
	#         tp = tp + 1
	#     if(labels[i] == 0 and na[i] == 1):
	#         fp = fp + 1
	#     if(labels[i] == 0 and na[i] == 0):
	#         fn = fn + 1
	#     if(labels[i] == 1 and na[i] == 0):
	#         tn = tn + 1
	# print "TP: {} ".format(tp)
	# print "FP: {} ".format(fp)
	# print "FN: {} ".format(fn)
	# print "TN: {} ".format(tn)

    # plot the prediction vs actual values
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)

    ax1 = binaryDF["ATT_FLAG"].plot(ax=ax1,label='actual')
    ax1 = dfPrediction.plot(ax=ax1,label='prediction');
    ax1 = plt.title('Actual vs Predicted Attack using ARMA')
    plt.legend()
    plt.grid()
    plt.show()

    return # void
