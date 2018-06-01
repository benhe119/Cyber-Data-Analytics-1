"""
Test ARMA models

# tutorial from https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/#comments
# download data csv from http://www.sidc.be/silso/INFO/snytotcsv.php
"""

print 'Cyber Data Analytics\nLab 2 - Test ARMA models\n'
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

x = np.linspace(0, 200*np.pi, 10000)
y = np.sin(x)

#datamat = np.concatenate((x,y),axis=0)
datamat = np.column_stack((x,y))

# dataframe containing 'time' and 'sinwave' columns
df = pd.DataFrame(datamat,columns=['time','sinwave'])

# fitARMA(df,'sinwave','time',p=2,q=1)

print df.describe()

diffed = df.diff(periods=1,axis=0)

diffed = diffed.iloc[1:]

print diffed

model = sm.tsa.ARMA(diffed['sinwave'],order=(2,0)).fit()


# plt.plot(x,y)
# plt.show()
