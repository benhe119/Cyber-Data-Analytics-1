'''
+-------------------------------------+
|    Credit Card Fraud                |
|    4 May 2018                       |
+-------------------------------------+
'''

#################################################################################
## 1 Import packages (make sure to use virtualenv workon CDA for correct packages)
#################################################################################

import pandas as pd

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

df = pd.read_csv('./knimepreprocessedfeatures.csv')  # knimepreprocessedfeatures knime_40times_features

print '\nCREDIT CARD FRAUD - ASSIGNMENT 1'

print '\nReading: knimepreprocessedfeatures.csv'
print '\nSee google drive document for field descriptions'

# import feature set (from KNIME) CSV file using pandas

print "\nRaw data shape from KNIME dataset file: knimepreprocessedfeatures.csv"
print df.shape

## COMMENTED BELOW DONE IN KNIME:
# convert string to datetime
# df['bookingdate'] = pd.to_datetime(df['bookingdate'])
# df['creationdate'] = pd.to_datetime(df['creationdate'])

# convert currencies and add to 'euro' field
# currency_dict = {'MXN': 0.01*0.05, 'SEK': 0.01*0.11, 'AUD': 0.01*0.67, 'GBP': 0.01*1.28, 'NZD': 0.01*0.61}
# df['euro'] = map(lambda x,y: currency_dict[y]*x, df['amount'],df['currencycode'])

#key = lambda k:k.day
#print df.groupby(df['bookingdate'].apply(key))

df_sort_creation = df.sort_values(by = 'creationdate', ascending = True) #Returns a new dataframe, leaving the original dataframe unchanged
key = lambda k:(k.year,k.month,k.day)
#print df_sort_creation.groupby(df_sort_creation['creationdate'].apply(key)).mean()['amount']

####################################################################
## 3   Dealing with missing data and rename the columns
####################################################################

def num_missing(x):
    return sum(x.isnull())

###### change names of columns to more human readable ones
df_renamed = df.rename(index=str, columns = {'txvariantcode': 'cardtype', 'bin': 'issuer_id', 'shopperinteraction': 'shoppingtype',
                   'simple_journal': 'label', 'cardverificationcodesupplied': 'cvcsupply',
                  'cvcresponsecode': 'cvcresponse', 'accountcode': 'merchant_id', 'MOD' : 'mod_hundreds', 'DIFF' : 'cmp_cntry_issuer', 'EURO' : 'amount_euro'})

###### select which columns we want in the final dataset
df_select = (df_renamed[['label', 'cmp_cntry_issuer', 'bookingdate', 'merchant_id', 'issuer_id',
              'issuercountrycode', 'amount', 'mod_hundreds', 'amount_euro', 'currencycode', 'shoppingtype',
              'creationdate', 'cardtype', 'card_id','cvcsupply','cvcresponse','mail_id','ip_id','shoppercountrycode']])

print "\nMissing values per column:"
print "  not shown"
# print df_select.apply(num_missing, axis=0)
print "\nMissing values per row:"
print "  not shown"
# # axis = 0 -> rows (aka indexes), axis = 1 -> cols
# print df_select.apply(num_missing, axis=1).head(n=5) # why this command only on head?

df_clean = df_select.dropna(axis=0)
# df_clean = df_select.fillna('missing')
print "\nMissing values per column:"
print "  not shown"
# print df_select.apply(num_missing, axis=0)


# print '\nDataset Statistics: for manipulated/cleaned/selected fields of dataset'
# print '\nData Shape:'
# print df_clean.shape
# print '\nIndex Types:\n-------------------------------------------'
# print df_clean.dtypes
# print '\nData Statistics:\n------------------------------------------------------------------------------'
# print df_clean.describe()

############# Assign Categories ###############

# categories -> var taking on limited, (usually) fixed number of possible values

# no label -> boolean
# no cmp_cntry_issuer -> boolean
# no bookingdate -> date

merchant_id_category = pd.Categorical(df_clean['merchant_id'])
issuer_id_category = pd.Categorical(df_clean['issuer_id'])
issuercountrycode_category = pd.Categorical(df_clean['issuercountrycode'])

# no amount -> double
# no mod hundreds -> double
# no amount euro -> double
currencycode_category = pd.Categorical(df_clean['currencycode'])
shoppingtype_category = pd.Categorical(df_clean['shoppingtype'])
# no creation date -> date

cardtype_category = pd.Categorical(df_clean['cardtype'])
card_id_category = pd.Categorical(df_clean['card_id'])
cvcsupply_category = pd.Categorical(df_clean['cvcsupply'])
#no cvc response -> boolean

mail_id_category = pd.Categorical(df_clean['mail_id'])
ip_id_category = pd.Categorical(df_clean['ip_id'])
shoppercountrycode_category = pd.Categorical(df_clean['shoppercountrycode'])

#################            Create Dictionaries             ###################

merchant_id_dict = dict(set(zip(merchant_id_category, merchant_id_category.codes)))
issuer_id_dict = dict(set(zip(issuer_id_category, issuer_id_category.codes)))
issuercountrycode_dict = dict(set(zip(issuercountrycode_category, issuercountrycode_category.codes)))

currencycode_dict = dict(set(zip(currencycode_category, currencycode_category.codes)))
shoppingtype_dict = dict(set(zip(shoppingtype_category, shoppingtype_category.codes)))

cardtype_dict = dict(set(zip(cardtype_category, cardtype_category.codes)))
card_id_dict = dict(set(zip(card_id_category, card_id_category.codes)))
cvcsupply_dict = dict(set(zip(cvcsupply_category, cvcsupply_category.codes)))

mail_id_dict = dict(set(zip(mail_id_category, mail_id_category.codes)))
ip_id_dict = dict(set(zip(ip_id_category, ip_id_category.codes)))
shoppercountrycode_dict = dict(set(zip(shoppercountrycode_category, shoppercountrycode_category.codes)))

###################        Assign codes to Data Frame     ######################

df_clean['issuercountrycode'] = issuercountrycode_category.codes
df_clean['cardtype'] = cardtype_category.codes
df_clean['currencycode'] = currencycode_category.codes
df_clean['shoppercountrycode'] = shoppercountrycode_category.codes
df_clean['shoppingtype'] = shoppingtype_category.codes
df_clean['cvcsupply'] = cvcsupply_category.codes
df_clean['merchant_id'] = merchant_id_category.codes
df_clean['mail_id'] = mail_id_category.codes
df_clean['ip_id'] = ip_id_category.codes
df_clean['card_id'] = card_id_category.codes
df_clean['label_int'], df_clean['cvcresponse_int']= 0,0
df_clean['label_int'] = map(lambda x:1 if str(x) == 'Chargeback' else 0 if str(x) == 'Settled' else 'unknown', df_clean['label'])
df_clean['cvcresponse_int'] = map(lambda x:3 if x > 2 else x+0, df_clean['cvcresponse'])
#0 = Unknown, 1=Match, 2=No Match, 3=Not checked
df1 = df_clean.ix[(df_clean['label_int']==1) | (df_clean['label_int']==0)]#237036 instances
#df1.head()

# Describe df1 dataset

# print '\nDataset Statistics: df1'
# print '\nData Shape:'
# print df1.shape
# print '\nIndex Types:\n-------------------------------------------'
# print df1.dtypes
# print '\n-----------------------------------\n'

# print '\nData Statistics:\n------------------------------------------------------------------------------'
# print df1.describe()

df_input = (df1[['issuercountrycode', 'cardtype', 'issuer_id', 'currencycode',
              'shoppercountrycode', 'shoppingtype', 'cvcsupply', 'cvcresponse_int', 'merchant_id', 'amount_euro',
               'label_int']]) # 'mod_hundreds',
df_input[['issuer_id','label_int']] = df_input[['issuer_id','label_int']].astype(int)

# print '\nDataset Statistics: df_input'
# print '\nData Shape:'
# print df_input.shape
# print '\nIndex Types:\n-------------------------------------------'
# print df_input.dtypes
# print '\n-----------------------------------\n'

x = df_input[df_input.columns[0:-1]].as_matrix()
y = df_input[df_input.columns[-1]].as_matrix()

####  Covariance matrix to CSV file

# print '\nCovariance Matrix of df_input'
#
# df_normalized = df_input;
#
# var = df_normalized.var()
#
# for i in range(0,len(var)):
#     df_normalized.ix[:,i] *= 1/np.sqrt(var[i])
#
#
# # normdf = df_input.truediv(var, axis=1)
# # print var
# # print len(var)
#
# covMat = df_normalized.cov()
# # print covMat
# print '\nWriting cov matrix to csv file'
# covMat.to_csv(path_or_buf='./covmat.csv')
# print 'csv file created\n'

####################################################################
##    Training Dataset
####################################################################

x_array = np.array(x)

# print 'x.shape: '
# print x.shape
# print 'x_array.size: '
# print x_array.size
# print '------'
# print 'y.shape: '
# print y.size
# print '\n'
# print np.mean(y)

# #x_array = preprocessing.normalize(np.array(x), norm='l2')
# enc = preprocessing.OneHotEncoder()
# #enc = preprocessing.LabelEncoder()
# enc.fit(x_array[:,0:-1])
# x_encode = enc.transform(x_array[:,0:-1]).toarray()
# min_max_scaler = preprocessing.MinMaxScaler()
#
# x_array[:,0:-1] = min_max_scaler.fit_transform(x_array[:,0:-1])
# x_in = np.c_[x_encode,x_array[:,-1]]
# y_in = np.array(y)

min_max_scaler = preprocessing.MinMaxScaler()

X_new = min_max_scaler.fit_transform(x)

x_array = np.reshape(X_new,x.shape)

# print 'X new size: '
# print X_new.size
# print '\n'
# print 'x_array size: '
# print x_array.shape
# print '\n'

# need x_in (9781x10)
#      y_in (9781x1)

#a = x_array[:,-1]



# shape = a.shape
# a = np.reshape(a, (-1, 1))
#
# a = min_max_scaler.fit_transform(a)
#
# a = np.reshape(a, shape)
#
# print a.shape
#
# x_array[:,-1] = a
#
# print a.shape
#
# x_in = np.c_[x_encode,x_array[:,-1]]
# y_in = np.array(y)

x_in = x_array;
y_in = y;

# print 'x_in size: '
# print x_in.size
# print 'y_in size: '
# print y_in.size

print '\nSplitting Training and Testing data ...'
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_in, y_in, test_size = 0.1)#test_size: proportion of train/test data
print '  Done\n'

# print '\n y_train size: '
# print y_train.shape
# print 'y_test size: '
# print y_test.shape

print '\nRatios of split data (before sampling)'
print '  Train data size: ' + str(x_train.shape)
print '  Test data size: ' + str(x_test.shape)
print '  Fraud count: training data  = ' + str(np.sum(y_train)) + '; testing data = ' + str(np.sum(y_test))

print "\nSampling data set ..."

# for clfidx in range(0,4):
#     print clfidx
#     if clfidx == 0:

# sampler = RandomUnderSampler()
# sampler = RandomOverSampler()

sampler = SMOTE(kind='regular')
print '  SMOTE Oversampling'
print '  Sampling data ...'

sampled_X, sampled_Y = sampler.fit_sample(x_train, y_train)


    # elif clfidx == 1:
    #     sampler = RandomUnderSampler()
    #     print '  Undersampling'
    #     print '  Sampling data ...'
    #     sampled_X, sampled_Y = sampler.fit_sample(x_train, y_train)
    # elif clfidx == 2:
    #     sampler = RandomOverSampler()
    #     print '  Oversampling'
    #     print '  Sampling data ...'
    #     sampled_X, sampled_Y = sampler.fit_sample(x_train, y_train)
    # else:
    #     print '  No Sampling ...'
    #     sampled_X = x_train;
    #     sampled_Y = y_train;

print '  Done\n'

# print 'After sampler\n'
# print 'sampled_X size'
# print sampled_X.size
# print 'sampled_Y size'
# print sampled_Y.size
# print '\n'

print 'Sampled Data'
print '  Train data size: ' + str(sampled_X.shape)
print '  Fraud count: training data = ' + str(np.sum(sampled_Y))
print '  * Remember: sampling not done on test data'

################################################################################
###########              DEFINE CLASSIFIER                      ################
################################################################################

# http://scikit-learn.org/stable/modules/svm.html
# http://scikit-learn.org/stable/modules/naive_bayes.html



print '\nCreating classifier object "clf"'
print '  Classifier used:'

########################   NORMAL CLASSIFIERS   ################################

# clf = LogisticRegression()
# print '  Logistical Regression'

# clf = RandomForestClassifier(n_estimators=5, criterion='gini')
# print '  Random Forest Classifier'

# clf = GaussianNB()
# print '  Gaussian Naive Bayes'

########################## ENSEMBLE CLASSIFIERS ################################

clf = XGBClassifier(max_depth=10,learning_rate=0.1,n_estimators=50)
print '  XG Boost Classifier'

# clf = AdaBoostClassifier(n_estimators=50);
# print '  AdaBoost Classifier'

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# print '  Gradient Boosting Classifier'

########################   VOTING CLASSIFIERS   ################################

# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(random_state=1)
# clf3 = GaussianNB()
#
# clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

################################################################################

# List of ML algorithms not to be used on LARGE datasets

# clf = svm.SVC(kernel='poly', degree=3)
# clf = svm.SVC(probability=True)
# clf = svm.SVC(kernel='sigmoid')

# clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=5, p=2,metric='minkowski',n_jobs=1)
# print ' K Nearest Neighbours'

################################################################################
##############              CROSS VALIDATION             #######################
################################################################################

cutoff = 0.5 # for y_predict

def cutoff_predict(clf, x, cutoff):
    return (clf.predict_proba(x)[:,1]>cutoff).astype(int)

def custom_score(cutoff):
    def score_cutoff(clf, x, y):
        ypred = cutoff_predict(clf, x, cutoff)
        return f1_score(y, ypred)
        #return precision_score(y, ypred)
    return score_cutoff

print '\nRunning Cross Validation ...'

scores = []
for cutoff in np.round(np.arange(0.1, 1.0, 0.1),decimals=1):
    # clf = LogisticRegression()
    validated = model_selection.cross_val_score(clf, sampled_X , sampled_Y, cv = 10, scoring = custom_score(cutoff))
    scores.append(validated) # possible pre-allocation needed? - NO => list length = 9
    print '  ' + str(cutoff)

print 'Done'

################################################################################
##############            ANALYSIS AND RESULTS           #######################
################################################################################

plt.figure(1)
sns.boxplot(np.round(np.arange(0.1, 1.0, 0.1),decimals=1), scores)
plt.title('F scores for each cutoff setting')
plt.xlabel('each cutoff value')
plt.ylabel('custom score')
plt.grid()

print '\nFit classifier with sampled training data ...'
clf.fit(sampled_X, sampled_Y)
print 'Done\n'
y_predict = clf.predict(x_test)#output label

#print 'clf.predict_proba'
predict_proba = clf.predict_proba(x_test)
#print 'y_predict'
y_predict = (predict_proba[:,1]>cutoff).astype(int)
#print 'false_positive_rate'

false_positive_rate, true_positive_rate, thresholds1 = roc_curve(y_test, predict_proba[:,1])
#print 'roc_auc'
roc_auc = auc(false_positive_rate, true_positive_rate)


## END FOR LOOP

plt.figure(2)

plt.subplot(1, 2, 1)
#plt.hold(1)
#plt.plot(false_positive_rate_none, true_positive_rate_none, 'k', label = 'None AUC = %0.2f'% roc_auc_none)
#plt.plot(false_positive_rate_under, true_positive_rate_under, 'g', label = 'Under AUC = %0.2f'% roc_auc_under)
#plt.plot(false_positive_rate_over, true_positive_rate_over, 'r', label = 'Over AUC = %0.2f'% roc_auc_over)
plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'SMOTE AUC = %0.2f'% roc_auc)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid()
plt.legend(loc="lower right")

precision, recall, thresholds2 = precision_recall_curve(y_test, predict_proba[:,1])
#precision_under, recall_under, thresholds2_under = precision_recall_curve(y_test, predict_proba_under[:,1])
#precision_over, recall_over, thresholds2_over = precision_recall_curve(y_test, predict_proba_over[:,1])
#precision_none, recall_none, thresholds2_none = precision_recall_curve(y_test, predict_proba_none[:,1])

plt.subplot(1, 2, 2)
#plt.hold(1)
#plt.plot(recall_none, precision_none, 'k', label = 'None')
#plt.plot(recall_under, precision_under, 'g', label = 'under_sampling')
#plt.plot(recall_over, precision_over, 'r', label = 'Oversampling')
plt.plot(recall, precision, 'b', label = 'SMOTE')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.grid()
plt.legend(loc="upper right")

# SMOTE
TP, FP, FN, TN = 0, 0, 0, 0
for i in xrange(len(y_predict)):
    if y_test[i]==1 and y_predict[i]==1:
        TP += 1
    if y_test[i]==0 and y_predict[i]==1:
        FP += 1
    if y_test[i]==1 and y_predict[i]==0:
        FN += 1
    if y_test[i]==0 and y_predict[i]==0:
        TN += 1

print 'Result of classifier applied to test data: SMOTED'
print 'TP: '+ str(TP)
print 'FP: '+ str(FP)
print 'FN: '+ str(FN)
print 'TN: '+ str(TN)

# # under_sampling
# TP, FP, FN, TN = 0, 0, 0, 0
# for i in xrange(len(y_predict_under)):
#     if y_test[i]==1 and y_predict_under[i]==1:
#         TP += 1
#     if y_test[i]==0 and y_predict_under[i]==1:
#         FP += 1
#     if y_test[i]==1 and y_predict_under[i]==0:
#         FN += 1
#     if y_test[i]==0 and y_predict_under[i]==0:
#         TN += 1
#
# print 'Result of classifier applied to test data: UNDERSAMPLING'
# print 'TP: '+ str(TP)
# print 'FP: '+ str(FP)
# print 'FN: '+ str(FN)
# print 'TN: '+ str(TN)
#
# # over_sampling
# TP, FP, FN, TN = 0, 0, 0, 0
# for i in xrange(len(y_predict_over)):
#     if y_test[i]==1 and y_predict_over[i]==1:
#         TP += 1
#     if y_test[i]==0 and y_predict_over[i]==1:
#         FP += 1
#     if y_test[i]==1 and y_predict_over[i]==0:
#         FN += 1
#     if y_test[i]==0 and y_predict_over[i]==0:
#         TN += 1
#
# print 'Result of classifier applied to test data: OVERSAMPLING'
# print 'TP: '+ str(TP)
# print 'FP: '+ str(FP)
# print 'FN: '+ str(FN)
# print 'TN: '+ str(TN)
#
# # no sampling
# TP, FP, FN, TN = 0, 0, 0, 0
# for i in xrange(len(y_predict_none)):
#     if y_test[i]==1 and y_predict_none[i]==1:
#         TP += 1
#     if y_test[i]==0 and y_predict_none[i]==1:
#         FP += 1
#     if y_test[i]==1 and y_predict_none[i]==0:
#         FN += 1
#     if y_test[i]==0 and y_predict_none[i]==0:
#         TN += 1
#
# print 'Result of classifier applied to test data: NONE'
# print 'TP: '+ str(TP)
# print 'FP: '+ str(FP)
# print 'FN: '+ str(FN)
# print 'TN: '+ str(TN)

plt.show()

###############################################################################
## END
###############################################################################
