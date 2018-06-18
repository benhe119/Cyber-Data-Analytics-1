"""
Code for the Reservoir Sampling as described in the slides

"""

### IMPORT PACKAGES ###
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.api import qqplot
# import scipy.fftpack as fftpk
# import scipy.signal as sgnl
from matplotlib import pyplot as plt
from random import randint, uniform


def  applyAlgorithmR(df_column, reservoirSize):

    # get number of columns n
    n = df_column.count()


    # sample first k items
    k = reservoirSize

    # print n and k
    # print 'n = ' + str(n)
    # print 'k = ' + str(k)
    # print ''

    # fill reservoir with first k elements of df_column
    df_reservoir = df_column.head(k)

    # print df_reservoir
    # print ''

    # replace elements with gradually decreasing probability
    for i in (xrange(n-k)):
        idx = i + k
        print idx
        j = randint(0,idx)
        if j < k:
            df_reservoir.iloc[j] = df_column.iloc[j]

    return df_reservoir

def  applyReservoirSampling(df_column, reservoirSize):

    # get number of columns n
    n = df_column.count()

    # sample first k items
    k = reservoirSize

    # print n and k
    # print 'n = ' + str(n)
    # print 'k = ' + str(k)
    # print ''

    # fill reservoir with first k elements of df_column
    df_reservoir = df_column.head(k)

    print df_reservoir

    onesmat = np.ones((k,1)).flatten(order='F')
    df_ones = pd.DataFrame(onesmat,columns=['Key'])

    #df_PQ = df_reservoir.join(df_ones) #pd.concat([df_reservoir, df_ones],ignore_index=True)


    df_reservoir = df_reservoir.to_frame().reset_index()

    #print df_reservoir
    #print df_ones

    #df_ones.append(df_reservoir) #,ignore_index=True)

    # print df_ones.describe()

    #df_PQ =  df_reservoir.append(df_ones,axis=1,ignore_index=True)
    df_PQ = pd.concat([df_reservoir, df_ones],axis=1)#,ignore_index=True)

    df_PQ.drop(["DateFlowStart"],axis=1,inplace=True)

    # print df_PQ

    #print df_PQ['Key'].max()

    df_column = df_column.to_frame().reset_index()

    #print df_column.iat[2,1]

    # print df_column.iloc[2]

    # loop through all entries of the data
    for i in xrange(n):

        # generate random number between 0 and 1
        r = uniform(0,1)

        if r < df_PQ['Key'].max():
            rIdx = df_PQ['Key'].idxmax()
            df_PQ.at[rIdx,'Key'] = r;
            df_PQ.at[rIdx,'SrcIPAddr'] = df_column.iat[i,1]



        # #print idx
        # j = randint(0,idx)
        # if j < k:
        #     df_reservoir.iloc[j] = df_column.iloc[j]

    print df_PQ
    print '\n'

    return# df_PQ
