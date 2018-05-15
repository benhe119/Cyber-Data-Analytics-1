'''
+-------------------------------------+
|    PREPROCESSING OF DATASET.CSV     |
|    15 May 2018                      |
+-------------------------------------+

Run this code to convert dataset.csv to preprocesseddata.csv

    - this converts 'checks' to individual columns of type 'integer'
    - eliminates only-zero columns,
    - removes the 'bin' rows with empty information (only 30 rows)
'''

#################################################################################
## 1 Import packages (make sure to use virtualenv workon CDA for correct packages)
#################################################################################

import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.linear_model import LogisticRegression     # Logistical Regression
from sklearn.ensemble import RandomForestClassifier     # Random Forest
from sklearn.neighbors import KNeighborsClassifier      # KNN
from sklearn.naive_bayes import GaussianNB              # Naive Bayes
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score

import seaborn as sns

import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
#from imblearn.unbalanced_dataset import UnbalancedDataset

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier


####################################################################
## 2 Read and Edit Feature Set (OBAINED FROM KNIME) from CSV File
####################################################################

df = pd.read_csv('./csv_files/dataset.csv')

print '\nADYEN OPTIMIZATION PROJECT'

print '\nReading: dataset.csv'

df_sort_creation = df.sort_values(by = 'creationdate', ascending = True) #Returns a new dataframe, leaving the original dataframe unchanged

####################################################################
## 3   Dealing with missing data and rename the columns
####################################################################

def num_missing(x):
    return sum(x.isnull())

###### change names of columns to more human readable ones
df_renamed = df.rename(index=str, columns = {'Unnamed: 0': 'rowID', 'bin': 'bankID'})#,
                   #'simple_journal': 'label', 'cardverificationcodesupplied': 'cvcsupply',
                  #'cvcresponsecode': 'cvcresponse', 'accountcode': 'merchant_id', 'MOD' : 'mod_hundreds', 'DIFF' : 'cmp_cntry_issuer', 'EURO' : 'amount_euro'})

###### select which columns we want in the final dataset
df_select = (df_renamed[['creationdate','amount','bankID','fraud_refusal','fraud','declined','checks']])

#df_clean = df_select.dropna(axis=0)
df_clean = df_select.fillna('missing')
# remove missing data
df_clean = df_clean.drop(df_clean[df_clean.bankID == 'missing'].index)

checkcols = df_clean['checks'].str.split(' ').apply(Series, 1) #.stack()
newcols = checkcols
for column in checkcols:
    newcols[column] = checkcols[column].str.extract('(\d+)').astype('int8')
#

df_merged = pd.concat([df_clean,newcols],axis=1)

df_merged.drop('checks',axis=1,inplace=True)

coltodeleteList = []

for colidx in range(0,78):
    print 'colidx: ' + str(colidx)
    strcolname = str(colidx)
    #colmax = df_merged[colidx].max()
    colmax = df_merged.iloc[:,colidx+6].max()
    print 'colmax: ' + str(colmax)
    if colmax < 0.01: # if max is zero
        #del df_merged.iloc[:,colidx+6]
        #df_merged.drop(df_merged.columns[colidx+6], axis=1)
        coltodeleteList.append(colidx+6)
        print ' column deleted'
    else:
        print ' column not deleted'
        #df_merged.drop(strcolname,axis=1,inplace=True)

print coltodeleteList
df_merged.drop(df_merged.columns[coltodeleteList],axis=1,inplace=True)

# create pre-processed data set as csv file
df_merged.to_csv(path_or_buf='./csv_files/processeddataset.csv')

print 'final shape of processed dataset'
print df_merged.shape

print '\nfinal dataset name: processeddataset.csv'
