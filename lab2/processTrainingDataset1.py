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
from discretize_filter import discretize_L_Txx, discretize_F_PUxx, discretize_S_PUxx, discretize_P_Jxxx

### READ AND EDIT CSV FILE ###   ###############################################
df = pd.read_csv("./data/BATADAL_training_dataset_1.csv",delimiter=',');
#,names=["DATETIME", "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4"])
#, "F_PU5, "S_PU5", F_PU6, S_PU6, F_PU7, S_PU7, F_PU8, S_PU8, F_PU9, S_PU9, F_PU10, S_PU10, F_PU11, S_PU11, F_V2, S_V2, P_J280, P_J269, P_J300, P_J256,
#P_J289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J14, P_J422, ATT_FLAG])
#print list(df.columns.values)
#df.sort(columns='L_T1',axis=0,ascending=True,inplace=True)
# dat_L_T1.plot(figsize=(6,4))
df.sort_values(by = 'DATETIME', ascending = True, inplace=True)

# print '\ndurbin watson statistics: ' + str(sm.stats.durbin_watson(dta))

# create list of field names
L_Txx = ["L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7"]
F_PUxx = ["F_PU1", "F_PU2", "F_PU3", "F_PU4", "F_PU5", "F_PU6", "F_PU7", "F_PU8", "F_PU9", "F_PU10", "F_PU11"]
S_PUxx = ["S_PU1", "S_PU2", "S_PU3", "S_PU4", "S_PU5", "S_PU6", "S_PU7", "S_PU8", "S_PU9", "S_PU10", "S_PU11"]
P_Jxxx = ['P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
Attack = ['ATT_FLAG']

discretize_L_Txx(L_Txx,df)

plt.show()
#df_F_PUxx = discretize_F_PUxx(F_PUxx)



    # b,a = sgnl.iirfilter(10,0.4,btype='lowpass',analog=False,ftype='butter',output='ba')
    # yfft_filt =
