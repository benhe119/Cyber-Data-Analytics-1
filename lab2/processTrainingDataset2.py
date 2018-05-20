'''
Scipt to process data for training dataset 2 of BATADAL SCADA data

# tutorial from https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/#comments
# download data csv from http://www.sidc.be/silso/INFO/snytotcsv.php
'''

### IMPORT PACKAGES ###

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot

from statsmodels.graphics.api import qqplot

import scipy.fftpack as fftpk

import scipy.signal as signal

### READ AND EDIT CSV FILE #####################################################

df = pd.read_csv("./data/BATADAL_training_dataset_2.csv",delimiter=',');
                 #,names=["DATETIME", "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4"])
#, "F_PU5, "S_PU5", F_PU6, S_PU6, F_PU7, S_PU7, F_PU8, S_PU8, F_PU9, S_PU9, F_PU10, S_PU10, F_PU11, S_PU11, F_V2, S_V2, P_J280, P_J269, P_J300, P_J256, P_J289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J14, P_J422, ATT_FLAG])

print list(df.columns.values)

#df.sort(columns='L_T1',axis=0,ascending=True,inplace=True)
df_sort_creation = df.sort_values(by = 'DATETIME', ascending = True)

# print df.shape
# print df.describe()

# map attacks:
# ATT_FLAG = 0 - no attack, ATT_FLAG = 1 - under attack
df[' ATT_FLAG'] = df[' ATT_FLAG'].map({-999: 0,1: 1})

dta = df[[' L_T1',' L_T2',' ATT_FLAG']]

# dta.plot(figsize=(6,4))
#
# print '\ndurbin watson statistics: ' + str(sm.stats.durbin_watson(dta))

### Fast Fourier Transform
#,names=["DATETIME", "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4"])
#, "F_PU5, "S_PU5", F_PU6, S_PU6, F_PU7, S_PU7, F_PU8, S_PU8, F_PU9, S_PU9, F_PU10, S_PU10, F_PU11, S_PU11, F_V2, S_V2, P_J280, P_J269, P_J300, P_J256, P_J289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J14, P_J422, ATT_FLAG])

#P_J280, P_J269, P_J300, P_J256, P_J289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J14, P_J422

# Get and plot the data
dat_L_T1 = df[[' P_J280']]
# dat_L_T1.plot(figsize=(6,4))


# STEP 1 - time to frequency domain
L_T1_mat = dat_L_T1.values;
N = L_T1_mat.size # number of sample points
T = 1.0/3600; # sample time is 1/3600 Hz (hourly measurements)
x = np.linspace(0.0, N*T, N)

y = L_T1_mat
y = np.asarray(y).reshape(-1)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x) # 50 Hz and 80 Hz sinusoid

y_mean = np.mean(y)
print y_mean
y_stddev = np.std(y)
print y_stddev

y_detrend = (y-y_mean)/y_stddev

print y_detrend.size

fig, ax1 = plt.subplots()
ax1.plot(x,y,label='original')
plt.hold(True)
ax1.plot(x,y_detrend,label='detrend')
plt.legend()
plt.title('Time domain of Signal')
# plt.hold(True)
# ax1.plot(x,y_detrend)

#+ np.sin(1.0 * 2.0*np.pi*x)
y = y_detrend;

yfft = fftpk.fft(y)
xfft = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xfft, 2.0/N*np.abs(yfft[:N//2]))
plt.title('FFT of signal')
plt.show()

yfft_clean = yfft

wc = yfft_clean[np.argmax(2.0/N*np.abs(yfft[:N//2]))] / (0.5 / T)
wp = [wc * 0.9, wc / 0.9]
ws = [wc * 0.95, wc / 0.95]

b, a  = signal.iirdesign(wp, ws, 1, 40)
f = signal.filtfilt(b, a, f)

# yfft_clean[np.abs(yfft_clean)>1.0] = 0

print yfft.size
print yfft_clean.size

fig, ax = plt.subplots()
ax.plot(xfft,2.0/N*np.abs(yfft_clean[:N//2]))
plt.show()


# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=300, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

#### fit ARMA (p=2,q=0) model to L_T1
# dta = df[[' L_T1']]
# arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
# print 'arma mod(2,0) params: ' + str(arma_mod20.params)



#plt.show()
