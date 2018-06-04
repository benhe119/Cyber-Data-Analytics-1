"""
Process data of BATADAL SCADA data

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
from N_gram import N_gram
from general_functions import standardize_dataset, standardize_dataset_train_2, standardize_dataset_test
from applyARMA import getBinaryDF, obtainFinalPrediction

### READ AND EDIT CSV FILE ###   ###############################################

#df = pd.read_csv("./data/BATADAL_training_dataset_1.csv", delimiter=',', parse_dates=True, index_col='DATETIME');
df_train_1 = pd.read_csv("./data/BATADAL_training_dataset_1.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DATETIME');
df_train_2 = pd.read_csv("./data/BATADAL_training_dataset_2.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DATETIME');
df_test = pd.read_csv("./data/BATADAL_test_dataset.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DATETIME');

# print list(df_train_2.columns.values)
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
V_Pxx = ["F_V2", "S_V2"]
P_Jxxx = ['P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
All = L_Txx + F_PUxx + S_PUxx + V_Pxx + P_Jxxx

Attack = ['ATT_FLAG']

# should the familiarization plots be created?
familiarizeData = False
if familiarizeData:

    dataSet = df_train_2
    # plot the different data to see which attack is which
    #dta = df_train_2["L_T1"]

    #for pumpno in np.arange(1,11,1):
    pumpno = 2
    fieldname1 = "F_PU" + str(pumpno)
    fieldname2 = "S_PU" + str(pumpno)

    # fieldname1 = "P_J14"
    # fieldname2 = "P_J422"

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(311)
    dataSet[fieldname1].plot(ax=ax1,figsize=(8,8))
    plt.title(fieldname1)
    plt.grid()

    ax2 = fig.add_subplot(312, sharex=ax1)
    dataSet[fieldname2].plot(ax=ax2,figsize=(8,8))
    plt.title(fieldname2)
    plt.grid()

    ax3 = fig.add_subplot(313, sharex=ax1)
    dataSet["ATT_FLAG"].plot(ax=ax3,figsize=(8,8))
    plt.title('Attack')
    #ax2 = ax1.twinx()
    plt.grid()

    plt.show()

analysisMethod = 'PCA' #ARMA, N-gram, PCA
ensembleMethod = 'Method1' # Method2 or None

print 'Analysis method: ' + analysisMethod

# intermediate
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# ax = df_test["F_PU1"].plot(ax=ax,label='3')
# ax = df_test["F_PU2"].plot(ax=ax,label='5')
# ax = df_test["F_PU4"].plot(ax=ax,label='9')
# plt.legend()
# plt.show()

if analysisMethod == 'ARMA':

    currentDataset = df_train_2

    # filter data
    # for field in []: #["L_T1"]: # L_Txx list
    #     filter([field],currentDataset)
    #
    # only choose one field for the ARMA model
    # dta = df[["L_T1"]]
    # print durbin_watson statistic to choose ARMA model parameters (see tutorial)
    # print '\ndurbin watson statistics: ' + str(sm.stats.durbin_watson(dta))
    # df.index = pd.Index(sm.tsa.datetools.dates_from_range('2015','2018'))
    # del df["DATETIME"]
    #
    # print currentDataset.index.min()
    # print currentDataset.index.max()
    # df.index = pd.Index(sm.tsa.datetools.dates_from_range('2014','2015'))
    # datetimefield = currentDataset.index
    #print currentDataset.describe()

    binaryDF = getBinaryDF(currentDataset)

    # for field in ['F_PU6']: #L_Txx: #["F_PU1"]: #["L_T1"]:
    #     fitARMA(currentDataset,field,0,p=2,q=2)
    # plt.show()

    obtainFinalPrediction(binaryDF)
elif analysisMethod == 'N-gram':

    # filter all L_Txx data using FFT
    filter(All,df_train_1)

    i = 1;

    all_anomalies = []

    # field to consider
    for datafieldname in All:
        # datafieldname = 'L_T2'

        # discretize data using SAX discretization
        timestampsTrain1, discretizedTrain1Data = discretizeSAX(datafieldname,df_train_1)
        timestampsTrain2, discretizedTrain2Data = discretizeSAX(datafieldname,df_train_2)

        # create n-gram from discretized data
        anomalyList = N_gram(discretizedTrain1Data, discretizedTrain2Data, 4, 0.0008) #traindata, testdata, n-gram size, alert threshold

        # # show difference in length of data
        # lengthAnomalyList = len(anomalyList)
        # lengthResultsList = len(timestampsTrain2)
        #
        # remove last values in timestamp until it matches the anomaly list length
        # this is because the discretization uses the data DIV n (of n-gram) and is
        # therefore different in length
        while (len(anomalyList) != len(timestampsTrain2)):
            timestampsTrain2 = timestampsTrain2[:-1] # remove last entry
            #print 'while loop entered'

        all_anomalies.append(anomalyList)

        #plt.figure(i)
        #plt.plot(timestampsTrain2,anomalyList,label='y filtered')
        #plt.title('Threshold breached for different')
        #plt.show()

        i = i+1

    combined_anomalies = [sum(x) for x in zip(*all_anomalies)]

    plt.figure(i)
    plt.plot(timestampsTrain2,combined_anomalies,label='y filtered')
    plt.title('Threshold breached for all')

    # plot the attacks
    plt.figure(i+1)
    df_train_2['ATT_FLAG'].plot(figsize=(8,4))
    plt.title('Attacks')
    plt.show()
elif analysisMethod == 'PCA':
	#PCA(df_train_1) #sub selection of data
    PCA_detection(df_train_1,df_train_2)
else:
    # do nothing
    print 'no analysis method...'

if ensembleMethod == 'Method1':
    print '\nEnsemble method 1'
elif ensembleMethod == 'Method2':
    print '\nEnsemble method 2'
else:
    # do nothing
    print '\nno ensemble method analysis...'

print '\nDone!'












#
