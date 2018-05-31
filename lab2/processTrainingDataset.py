"""
Process data for training dataset 1 of BATADAL SCADA data

1) FFT analysis to remove signal noise
2) Discretization of filtered signals

# tutorial from https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/#comments
# download data csv from http://www.sidc.be/silso/INFO/snytotcsv.php
"""

print 'Cyber Data Analytics\nLab 2 - Anomaly Detection\n'
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
from discretize_data import discretizeBinary, discretizeSAX

from ARMA import fitARMA
from PCA import PCA, PCA_detection
from general_functions import standardize_dataset, standardize_dataset_train_2, standardize_dataset_test

### READ AND EDIT CSV FILE ###   ###############################################

#df = pd.read_csv("./data/BATADAL_training_dataset_1.csv", delimiter=',', parse_dates=True, index_col='DATETIME');
df_train_1 = pd.read_csv("./data/BATADAL_training_dataset_1.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DATETIME');
df_train_2 = pd.read_csv("./data/BATADAL_training_dataset_2.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DATETIME');
df_test = pd.read_csv("./data/BATADAL_test_dataset.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DATETIME');

#print list(df.columns.values)
#df.sort(columns='L_T1',axis=0,ascending=True,inplace=True)
# dat_L_T1.plot(figsize=(6,4))
#df.sort_values(by = 'DATETIME', ascending = True, inplace=True)
# print '\ndurbin watson statistics: ' + str(sm.stats.durbin_watson(dta))
# plot data
# dataToPlot = df[["L_T1"]]
# plt.plot(dataToPlot)
# plt.show()

df_train_1 = standardize_dataset(df_train_1)
df_train_2 = standardize_dataset_train_2(df_train_2)
df_test = standardize_dataset_test(df_test)

# Lists of field names for the datasets
L_Txx = ["L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7"]
F_PUxx = ["F_PU1", "F_PU2", "F_PU3", "F_PU4", "F_PU5", "F_PU6", "F_PU7", "F_PU8", "F_PU9", "F_PU10", "F_PU11"]
S_PUxx = ["S_PU1", "S_PU2", "S_PU3", "S_PU4", "S_PU5", "S_PU6", "S_PU7", "S_PU8", "S_PU9", "S_PU10", "S_PU11"]
P_Jxxx = ['P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
Attack = ['ATT_FLAG']

analysisMethod = 'NONE'

print 'Analysis method: ' + analysisMethod

if analysisMethod == 'ARMA':

    # filter data
    for field in [" L_T4"]: # L_Txx list
        filter([field],df_train_1)

    # only choose one field for the ARMA model
    # dta = df[["L_T1"]]

    # print durbin_watson statistic to choose ARMA model parameters (see tutorial)
    # print '\ndurbin watson statistics: ' + str(sm.stats.durbin_watson(dta))

    # df.index = pd.Index(sm.tsa.datetools.dates_from_range('2015','2018'))
    # del df["DATETIME"]

    print df_train_1.index.min()
    print df_train_1.index.max()
    # df.index = pd.Index(sm.tsa.datetools.dates_from_range('2014','2015'))
    datetimefield = df_train_1.index


    for field in [" L_T4"]:
        fitARMA(df_train_1[field],datetimefield,p=2,q=0)











elif analysisMethod == 'N-gram':
    # filter data using FFT
    filter(L_Txx,df_train_1)

    # discretize data using SAX discretization
    discretizeSAX(L_Txx,df_train_1)

    # create n-gram from discretized data











elif analysisMethod == 'PCA':
	#PCA(df) #sub selection of data
    PCA_detection(df_train_1)
else:
    # do nothing
    print 'no analysis method...'

print 'Done!'

#discretizeBinary(L_Txx,df)

# discretizes the dataset using SAX - only the first field for now






# asdf
