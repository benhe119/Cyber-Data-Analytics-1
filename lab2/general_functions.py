"""
Files for miscellaneous functions


"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARMA

from statsmodels.graphics.api import qqplot


# adds ATTFLAG field of not there, adjusts values etc.
def standardize_dataset(dataset):
    #print dataset.describe()
    return dataset

def standardize_dataset_train_2(dataset2):
    #print dataset2.describe()

    # convert att flag to 1 = attack, 0 = benign
    # dataset2["ATT_FLAG"] = np.where(dataset2["ATT_FLAG"] == -999).astype(int)

    # remove space from infront of field names
    dataset2.columns = [x.strip().replace(' ', '_') for x in dataset2.columns]

    mask = dataset2['ATT_FLAG'] == -999
    fieldname = 'ATT_FLAG'
    dataset2.loc[mask, fieldname] = 0

    return dataset2

def standardize_dataset_test(dataset_test):

    # add attack flag with all -1 (unknown)
    datalength = dataset_test.shape[0] # number of rows
    # print -1*np.ones((datalength,1),dtype=int)
    # att_flag_field = pd.DataFrame(-1*np.ones((datalength,1),dtype=int))
    #,index=dataset_test.index,columns='ATT_FLAG')#,index=dataset_test.index,dtype=int)

    # dataset_testnew = pd.concat([dataset_test,att_flag_field],axis=1,ignore_index=True)

    # add a field called ATT_FLAG with -1's (no data) - allows for easy plotting
    dataset_test.insert(len(dataset_test.columns), 'ATT_FLAG', -1*np.ones((datalength,1),dtype=int), allow_duplicates=True)

    return dataset_test
