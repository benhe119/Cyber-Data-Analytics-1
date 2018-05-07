#inspired by https://beckernick.github.io/oversampling-modeling/

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

#import all transactions
transactions = pd.read_csv('./data_for_student_case.csv')

#set the dates to unix timestamps
transactions['bookingdate'] = pd.to_datetime(transactions['bookingdate'])
transactions['creationdate'] = pd.to_datetime(transactions['creationdate'])

#change all currencies to EUR
currency_dict = {'MXN': 0.01*0.05, 'SEK': 0.01*0.11, 'AUD': 0.01*0.67, 'GBP': 0.01*1.28, 'NZD': 0.01*0.61}
transactions['euro'] = map(lambda x,y: currency_dict[y]*x, transactions['amount'],transactions['currencycode'])

#print transactions.iloc[0]

print transactions.simple_journal.value_counts()

#which variables do we want to use
model_variables = ['txvariantcode', 'bin','shopperinteraction', 'simple_journal','cardverificationcodesupplied',
            'cvcresponsecode', 'accountcode', 'issuercountrycode', 'euro', 'currencycode',
            'shoppercountrycode', 'mail_id', 'ip_id', 'card_id']

#filter out the the above defined variables
transactions_data_relevant = transactions[model_variables]

#hot encode the categorial features as binary to use them in the sklearn random forest classifier
transactions_relevant_enconded = pd.get_dummies(transactions_data_relevant)

#create test and training set (remove the simple_journal in the test set)
'''training_features, test_features, \
training_target, test_target, = train_test_split(transactions_relevant_enconded.drop(['simple_journal'], axis=1),
                                               transactions_relevant_enconded['simple_journal'],
                                               test_size = .1,
                                               random_state=12)'''

print "done"
'''
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .1,
                                                  random_state=12)'''
'''
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(training_features, training_target)
print training_target.value_counts(), np.bincount(y_res)'''