'''

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

### READ AND EDIT CSV FILE #####################################################

df = pd.read_csv("./sunspots.csv",delimiter=';',names=["midyear","SUNACTIVITY","MeanStdDev","Observation","Marker"])
# print sm.datasets.sunspots.NOTE
# convert date to last day of the year i.e 1700-12-31 etc:
df.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2017'))

# print '\ndescribe'
# print df.describe()
# print '\nshape'
# print df.shape

# exctract only sunactivity for analysis
dta = df[['SUNACTIVITY']]

### ANALYZE DATA & AUTOCORRELATION GRAPHS ######################################

# raw data signal
dta.plot(figsize=(6,4));

print '\ndurbin watson statistics: ' + str(sm.stats.durbin_watson(dta))

## use matplotlib to plot autocorrelation
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=300, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

## use autocorrelation_plot from pandas
# dta['SUNACTIVITY_2'] = dta['SUNACTIVITY']
# dta['SUNACTIVITY_2'] = (dta['SUNACTIVITY_2'] - dta['SUNACTIVITY_2'].mean()) / (dta['SUNACTIVITY_2'].std())
# plt.acorr(dta['SUNACTIVITY_2'],maxlags = len(dta['SUNACTIVITY_2']) -1, linestyle = "solid", usevlines = False, marker='')
#
# autocorrelation_plot(dta['SUNACTIVITY'])

# plt.show() # GENERAL PLOT

### MODELING THE DATA ##########################################################

# MODEL 1 - ARMA (p=2,q=0) model
arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
#
print 'arma mod(2,0) params: ' + str(arma_mod20.params)

# aic, bic, hqic
print arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic

# durbin watson
print sm.stats.durbin_watson(arma_mod20.resid.values)

# plot the data represented by the model
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax = arma_mod20.resid.plot(ax=ax);

# analyze residuals
resid20 = arma_mod20.resid
stats.normaltest(resid20)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid20, line='q', ax=ax, fit=True)

# Model Autocorrelation
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid20.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid20, lags=40, ax=ax2)

r,q,p = sm.tsa.acf(resid20.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print table.set_index('lag')

# Predictions
predict_sunspots20 = arma_mod20.predict('1990', '2017', dynamic=True)
print predict_sunspots20

ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_sunspots20.plot(ax=ax, style='r--', label='Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0));

plt.show()
