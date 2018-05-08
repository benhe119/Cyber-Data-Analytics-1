'''
+---------------------------------------------+
|    Credit Card Fraud                        |
|    4 May 2018                               |
+---------------------------------------------+
'''

'''
    1   import packages (make sure to use virtualenv workon CDA for correct packages)
'''
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# cross_validation will soon be deprecated, rather use model_selection
from sklearn import cross_validation
#from sklearn import model_selection

import numpy as np


#from imblearn.over_sampling import OverSampler

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
#from imblearn.unbalanced_dataset import UnbalancedDataset

'''
    2   Read and Edit Feature Set (OBAINED FROM KNIME) from CSV File
'''
df = pd.read_csv('/home/lx/Documents/Coursework/Q4/CyberDataAnalytics/Assignment1/Code/knime_10times_features.csv')  # knimepreprocessedfeatures

print '\nCREDIT CARD FRAUD - ASSIGNMENT 1'

print '\nReading: knimepreprocessedfeatures.csv'
print '\nSee google drive document for field descriptions'

# import feature set (from KNIME) CSV file using pandas

print "\n Raw data shape from KNIME:"
print df.shape

'''DONE IN KNIME'''
# convert string to datetime
# df['bookingdate'] = pd.to_datetime(df['bookingdate'])
# df['creationdate'] = pd.to_datetime(df['creationdate'])

# convert currencies and add to 'euro' field
# currency_dict = {'MXN': 0.01*0.05, 'SEK': 0.01*0.11, 'AUD': 0.01*0.67, 'GBP': 0.01*1.28, 'NZD': 0.01*0.61}
# df['euro'] = map(lambda x,y: currency_dict[y]*x, df['amount'],df['currencycode'])

#key = lambda k:k.day
#print df.groupby(df['bookingdate'].apply(key))

df_sort_creation = df.sort_values(by = 'creationdate', ascending = True)#Returns a new dataframe, leaving the original dataframe unchanged
key = lambda k:(k.year,k.month,k.day)
#print df_sort_creation.groupby(df_sort_creation['creationdate'].apply(key)).mean()['amount']

'''
    3   Dealing with missing data and rename the columns
'''

def num_missing(x):
    return sum(x.isnull())

''' change names of columns to more human readable ones'''
df_renamed = df.rename(index=str, columns = {'txvariantcode': 'cardtype', 'bin': 'issuer_id', 'shopperinteraction': 'shoppingtype',
                   'simple_journal': 'label', 'cardverificationcodesupplied': 'cvcsupply',
                  'cvcresponsecode': 'cvcresponse', 'accountcode': 'merchant_id', 'MOD' : 'mod_hundreds', 'DIFF' : 'cmp_cntry_issuer', 'EURO' : 'amount_euro'})

''' select which columns we want in the final dataset'''
df_select = (df_renamed[['label', 'cmp_cntry_issuer', 'bookingdate', 'merchant_id', 'issuer_id',
              'issuercountrycode', 'amount', 'mod_hundreds', 'amount_euro', 'currencycode', 'shoppingtype',
              'creationdate', 'cardtype', 'card_id','cvcsupply','cvcresponse','mail_id','ip_id','shoppercountrycode']])

# data shape, datatypes, description

# print '\nData Statistics: for manipulated/selected fields of dataset'
# print '\nData Shape:'
# print df_select.shape
# print '\nIndex Types:\n----------------------------------------------'
# print df_select.dtypes
# print '\nData Statistics:\n----------------------------------------------------------------------------------'
# print df_select.describe()
#
print "\nMissing values per column:"
print df_select.apply(num_missing, axis=0)
print "\nMissing values per row:"
# # axis = 0 -> rows (aka indexes), axis = 1 -> cols
print df_select.apply(num_missing, axis=1).head(n=5) # why this command only on head?

''' create final dataset - without N/A values '''

df_clean = df_select.dropna(axis=0)
# df_clean = df_select.fillna('missing')
print "\nMissing values per column:"
print df_select.apply(num_missing, axis=0)


# print '\nDataset Statistics: for manipulated/cleaned/selected fields of dataset'
# print '\nData Shape:'
# print df_clean.shape
# print '\nIndex Types:\n-------------------------------------------'
# print df_clean.dtypes
# print '\nData Statistics:\n------------------------------------------------------------------------------'
# print df_clean.describe()

# Assign Categories

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

# Create Dictionaries

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

# Assign codes to Data Frame

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

print '\nDataset Statistics: df1'
print '\nData Shape:'
print df1.shape
print '\nIndex Types:\n-------------------------------------------'
print df1.dtypes
print '\n-----------------------------------\n'


#part 6


df_input = (df1[['issuercountrycode', 'cardtype', 'issuer_id', 'currencycode',
              'shoppercountrycode', 'shoppingtype', 'cvcsupply', 'cvcresponse_int', 'merchant_id', 'amount_euro',
              'label_int']])
df_input[['issuer_id','label_int']] = df_input[['issuer_id','label_int']].astype(int)

print '\nDataset Statistics: df_input'
print '\nData Shape:'
print df_input.shape
print '\nIndex Types:\n-------------------------------------------'
print df_input.dtypes
print '\n-----------------------------------\n'

x = df_input[df_input.columns[0:-1]].as_matrix()
y = df_input[df_input.columns[-1]].as_matrix()

# print '\nx ():\n---------------------------------------------'
# print x
print '\n'


#    7   Train dataset

TP, FP, FN, TN = 0, 0, 0, 0
x_array = np.array(x)

print 'x.shape: '
print x.shape
print 'x_array.size: '
print x_array.size
print '------'
print 'y.shape: '
print y.size
print '\n'
print np.mean(y)

'''
#x_array = preprocessing.normalize(np.array(x), norm='l2')
enc = preprocessing.OneHotEncoder()
#enc = preprocessing.LabelEncoder()
enc.fit(x_array[:,0:-1])
x_encode = enc.transform(x_array[:,0:-1]).toarray()
min_max_scaler = preprocessing.MinMaxScaler()

x_array[:,0:-1] = min_max_scaler.fit_transform(x_array[:,0:-1])
x_in = np.c_[x_encode,x_array[:,-1]]
y_in = np.array(y)
'''

min_max_scaler = preprocessing.MinMaxScaler()

X_new = min_max_scaler.fit_transform(x)

x_array = np.reshape(X_new,x.shape)

print 'X new size: '
print X_new.size
print '\n'
print 'x_array size: '
print x_array.shape
print '\n'

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

print 'x_in size: '
print x_in.size
print 'y_in size: '
print y_in.size

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_in, y_in, test_size = 0.2)#test_size: proportion of train/test data
print "\nTraining data set ..."
#sampler = RandomUnderSampler()
#sampler = RandomOverSampler()
sampler = SMOTE(kind='regular')
print 'before sampler'
usx, usy = sampler.fit_sample(x_train, y_train)
print 'after sampler'
print '\n'
print 'usx size'
print usx.size
print 'usy size'
print usy.size

'''
part 8
'''

def cutoff_predict(clf, x, cutoff):
    return (clf.predict_proba(x)[:,1]>cutoff).astype(int)
scores = []

def custom_score(cutoff):
    def score_cutoff(clf, x, y):
        ypred = cutoff_predict(clf, x, cutoff)
        return f1_score(y, ypred)
        #return precision_score(y, ypred)
    return score_cutoff

print 'before cutoff loop'
for cutoff in np.arange(0.1, 0.9, 0.1):
    clf = LogisticRegression()
    validated = cross_validation.cross_val_score(clf, usx , usy, cv = 10, scoring = custom_score(cutoff))
    scores.append(validated) # possible pre-allocation needed?
    print cutoff



sns.boxplot(np.arange(0.1, 0.9, 0.1), scores)
plt.title('F scores for each cutoff setting')
plt.xlabel('each cutoff value')
plt.ylabel('custom score')
plt.show()


# http://scikit-learn.org/stable/modules/svm.html
# http://scikit-learn.org/stable/modules/naive_bayes.html

'''
part 9
'''
cutoff = 0.6

#clf = LogisticRegression()
#clf = svm.SVC(kernel='poly', degree=3)
print 'create clf'
clf = svm.SVC(probability=True)

#clf = RandomForestClassifier(n_estimators=50, criterion='gini')
#clf = svm.SVC(kernel='sigmoid')
print 'fit clf'
clf.fit(usx, usy)
#y_predict = clf.predict(x_test)#output label

print 'clf.predict_proba'
predict_proba = clf.predict_proba(x_test)
print 'y_predict'
y_predict = (predict_proba[:,1]>cutoff).astype(int)
print 'false_positive_rate'
false_positive_rate, true_positive_rate, thresholds1 = roc_curve(y_test, predict_proba[:,1])
print 'roc_auc'
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.subplot(1, 2, 1)
plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'AUC = %0.2f'% roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
#plt.legend('AUC = %0.2f'% roc_auc)
plt.legend(loc="lower right")
precision, recall, thresholds2 = precision_recall_curve(y_test, predict_proba[:,1])
plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.show()
for i in xrange(len(y_predict)):
    if y_test[i]==1 and y_predict[i]==1:
        TP += 1
    if y_test[i]==0 and y_predict[i]==1:
        FP += 1
    if y_test[i]==1 and y_predict[i]==0:
        FN += 1
    if y_test[i]==0 and y_predict[i]==0:
        TN += 1
print 'TP: '+ str(TP)
print 'FP: '+ str(FP)
print 'FN: '+ str(FN)
print 'TN: '+ str(TN)






''' ERNST PREV

TP, FP, FN, TN = 0, 0, 0, 0
x_array = np.array(x)
#x_array = preprocessing.normalize(np.array(x), norm='l2')
enc = preprocessing.OneHotEncoder()
#enc = preprocessing.LabelEncoder()
enc.fit(x_array[:,0:-1])

x_encode = enc.transform(x_array[:,0:-1]).toarray()

min_max_scaler = preprocessing.MinMaxScaler()

a = x_array[:,-1]

# reshape
shape = a.shape
a = np.reshape(a, (-1, 1))

a = min_max_scaler.fit_transform(a)

# reshape back
a = np.reshape(a, shape)
print a.shape

x_array[:,-1] = a
x_in = np.c_[x_encode,x_array[:,-1]]
y_in = np.array(y)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_in, y_in, test_size = 0.2)#test_size: proportion of train/test data

print "Training date set"
sampler = RandomUnderSampler()

#sampler = RandomOverSampler()
#sampler = SMOTE(kind='regular')
usx, usy = sampler.fit_sample(x_train, y_train)
'''


''' ALEX PREV
TP, FP, FN, TN = 0, 0, 0, 0
x_array = np.array(x)

print 'x_array:'
print x_array
print ''

#x_array = preprocessing.normalize(np.array(x), norm='l2')
enc = preprocessing.OneHotEncoder() # sklearn
#enc = preprocessing.LabelEncoder()

# removes last column of x_array (WHY??)
enc.fit(x_array[:,0:-1])
x_encode = enc.transform(x_array[:,0:-1]).toarray()
min_max_scaler = preprocessing.MinMaxScaler()
x_array[:,-1] = min_max_scaler.fit_transform(x_array[:,-1])
x_in = np.c_[x_encode,x_array[:,-1]]
y_in = np.array(y)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_in, y_in, test_size = 0.2)#test_size: proportion of train/test data
print "Training data set"
sampler = UnderSampler(ratio=10, verbose = True)
#sampler = OverSampler(verbose = True)
#sampler = SMOTE(kind='regular', verbose = True)
usx, usy = sampler.fit_transform(x_train, y_train)

#raw_input('Press RETURN to exit')
'''
