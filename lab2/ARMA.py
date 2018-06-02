"""
File containing functions to train ARMA models

input: dataframe field

"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARMA

from statsmodels.graphics.api import qqplot

def fitARMA(dataframe,currentfield,datetimefield,p=2,q=0):

    data = dataframe[currentfield]

    # durbin watson stat
    print "Durbin Watson statistic: " + str(sm.stats.durbin_watson(data))

    # Plot autocorrelation and partial autocorrelation (using statsmodels)
    # lag = 100; # lag to plot
    # fig = plt.figure(figsize=(6,4))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=lag, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(data, lags=lag, ax=ax2)
    #plt.show()

    # Plot autocorrelation using pandas
    # data_2 = data
    # data_2 = (data_2 - data_2.mean()) / (data_2.std())
    # plt.acorr(data_2, maxlags = len(data_2) -1, linestyle = "solid", usevlines = False, marker='')
    # plt.title('autocorrelation plot using pandas')
    # #plt.show()
    # autocorrelation_plot(data)
    #plt.show()

    # fit arma models and print parameters
    model = sm.tsa.ARMA(data, order=(p,q)).fit()

    # model = ARMA(data, (2,0))

    # print model.start_params()

    # print 'Model Parameters: '
    # print model.params
    # print model

    # AIC
    # print '\nCriteria\n'
    # print model.aic

    # does out model obey the theory?
    print 'Durbin Watson (model residuals/errors): ' + str(sm.stats.durbin_watson(model.resid.values))

    # no dependency on lags, quite centered (don't need ARIMA, only ARMA or AR)
    # autocorrelation_plot(dta)
    # plt.show()

    # plot the data the model represents
    fig = plt.figure(figsize=(6,4))
    ax1 = fig.add_subplot(211)
    ax1 = model.resid.plot(ax=ax1,label='Residuals');
    ax1 = plt.title('ARMA model residuals')
    resmat = model.resid
    print resmat.describe()

    #ax = plt.plot(resmat.index,dataframe['ATT_FLAG'],label='Attack')
    ax2 = fig.add_subplot(212)#, sharex = ax1)#, sharex=ax1)

    ax2 = dataframe['ATT_FLAG'].plot(ax=ax2,label='Attack')
    ax2 = plt.title('Attack')

    # plt.legend()
    # plt.xlabel('Date')
    # plt.ylabel('Residual Error')
    #plt.show()


    # model residuals
    # resid = model.resid
    # stats.normaltest(resid)
    #
    # fig = plt.figure(figsize=(12,8))
    # ax = fig.add_subplot(111)
    # fig = qqplot(resid, line='q', ax=ax, fit=True)
    #
    # # model autocorrelation
    # fig = plt.figure(figsize=(12,8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
    #
    # r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    # data = np.c_[range(1,41), r[1:], q, p]
    # table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    # print table.set_index('lag')
    #
    # # predictions
    # predict_level = model.predict('2015', '2016', dynamic=True)
    # print predict_level

    # ax = data.ix['2014':].plot(figsize=(12,8))
    # ax = predict_levels.plot(ax=ax, style='r--', label='Dynamic Prediction');
    # ax.legend();
    # ax.axis((-20.0, 38.0, -4.0, 200.0));

    # Calculate Forecast errors
    def mean_forecast_err(y, yhat):
        return np.subtract(y,yhat).mean()

    def mean_absolute_err(y, yhat):
        return np.mean((np.abs(np.subtract(y,yhat).mean()) / yhat)) # or percent error = * 100

    # print "MFE = ", mean_forecast_err(data, predict_level)
    # print "MAE = ", mean_absolute_err(data, predict_level)

    # Model 2: ARMA(3,0)

    # arma_mod30 = sm.tsa.ARMA(data, (3,0)).fit()
    #
    # print arma_mod30.params
    #
    # print arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic
    #
    # sm.stats.durbin_watson(arma_mod30.resid.values)
    #
    # fig = plt.figure(figsize=(12,8))
    # ax = fig.add_subplot(111)
    # ax = arma_mod30.resid.plot(ax=ax);
    #
    # resid30 = arma_mod30.resid
    # stats.normaltest(resid30)
    #
    # fig = plt.figure(figsize=(12,8))
    # ax = fig.add_subplot(111)
    # fig = qqplot(resid30, line='q', ax=ax, fit=True)
    #
    # fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid30.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid30, lags=40, ax=ax2)
#
# r,q,p = sm.tsa.acf(resid30.values.squeeze(), qstat=True)
# data = np.c_[range(1,41), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print table.set_index('lag')
#
# predict_sunspots30 = arma_mod30.predict('1990', '2012', dynamic=True)
# print predict_sunspots30
#
# ax = dta.ix['1950':].plot(figsize=(12,8))
# ax = predict_sunspots30.plot(ax=ax, style='r--', label='Dynamic Prediction');
# ax.legend();
# ax.axis((-20.0, 38.0, -4.0, 200.0));
#
# print "MFE = ", mean_forecast_err(dta.SUNACTIVITY, predict_sunspots30)
# print "MAE = ", mean_absolute_err(dta.SUNACTIVITY, predict_sunspots30)


def fitAR():
    return #void

def fitARIMA():
    return #void
