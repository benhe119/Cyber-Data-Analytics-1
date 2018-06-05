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

# concatenates two datasets using an OR operation for detection of attacks
def mergeORResults(df1, df2):
    # copy same size dataframe
    mergedDF = df1.copy(deep=True)
    # set all columns to zero
    mergedDF[:] = 0
    # add OR operator
    mergedDF[((df1['PC3']==1) & (df2['PC3']==1))] == 1
    return mergedDF

# concatenates two datasets using an AND operation for detection of attacks
def mergeANDResults(df1, df2):
    return df1
