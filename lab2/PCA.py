import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt

#read more: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

#PCA selection of subspace
def PCA(csv):

	#delete the flag column
	del csv['ATT_FLAG']

	#scale
	scaler = StandardScaler()
	training_normalized = scaler.fit_transform(csv)
	#print training_normalized
	#print training_normalized.shape

	#create PCA model
	pca = decomposition.PCA() #components can be changed
	pca.fit(training_normalized)
	pca_model = pca.transform(training_normalized)

	#print pca_model
	#output variance and variance_ratio
	print np.sort(pca.explained_variance_)[::-1]
	print np.sort(pca.explained_variance_ratio_)[::-1]

	#cummulative variance ratio
	pca.explained_variance_ratio_.cumsum()

	#plot variance
	x_axis = np.arange(1, (len(pca.explained_variance_ratio_) + 1), 1)
	#plt.xlabel('Principal Component')
	#plt.ylabel('Variance Captured')
	#plt.plot(x_axis, pca.explained_variance_ratio_)
	#plt.show()

	#plot cumulative variance
	plt.figure(1,figsize=(6,2.5))

	plt.title('Cumulative Variance of Principal Components')
	plt.xlabel('Principal Components')
	plt.ylabel('Cumulative Variance Captured')
	plt.plot(x_axis, pca.explained_variance_ratio_.cumsum())
	plt.grid()
	plt.tight_layout()
	plt.show()

	#plot component 5 (for comparison to show the normal behaviour)
	csv = csv.assign(PC5=pca_model[:,5])
	csv['PC5'].plot(figsize=(15,5))
	plt.show();

	#plot component 6 or 7 (which are irregular)
	csv = csv.assign(PC6=pca_model[:,6])
	csv['PC6'].plot(figsize=(15,5))
	plt.show();

	return

#PCA detection
def PCA_detection(csv,testset):

	#delete the flag column
	#del csv['ATT_FLAG']

	#scale
	scaler = StandardScaler()
	training_normalized = scaler.fit_transform(csv)
	#print training_normalized
	#print training_normalized.shape

	pca = decomposition.PCA()
	pca.fit(training_normalized)
	pca_model = pca.transform(training_normalized)

	#cummulative variance ratio
	pca.explained_variance_ratio_.cumsum()

	#load and process the test_dataset
	#test_dataset = pd.read_csv("./data/BATADAL_test_dataset.csv",delimiter=',',parse_dates=True, index_col='DATETIME');
	test_dataset = testset

	#print testset.describe()

	#print testset.index

	#these lines need to be remove for the test_dataset (because they don't contain that label)
	labels =  test_dataset['ATT_FLAG']
	# del test_dataset['ATT_FLAG']


	test_normalized = scaler.fit_transform(test_dataset)
	pca_test = pca.transform(test_normalized)

	#rename variable for easier reading code below
	eigenvectors = pca.components_

	# Matrix P represents principal components corresponding to normal subspace
	P = np.transpose(eigenvectors[:11])
	P_T = np.transpose(P)
	C = np.dot(P, P_T)

	# Identity Matrix with dimensions 43 X 43
	I = np.identity(44)

	# y_residual is the projection of test data on anomalous subspace
	y_residual = np.zeros((test_normalized.shape))

	# Calculate projection of test data on anomalous subspace
	for i in range(test_normalized.shape[0]):
	    # Convert row to column vector
	    y = np.transpose(test_normalized[i])
	    y_residual[i] = np.dot(I - C, y)


	#Calculate SPE for each y_residual
	spe = np.zeros((test_normalized.shape[0]))

	# na will be set to 1 if the spe is greater than the threshold
	na = np.zeros((test_normalized.shape[0]))
	predicted_attacks_array = np.zeros((test_normalized.shape[0]))
	threshold = 500
	for i in range(test_normalized.shape[0]):
	    spe[i] = np.square(np.sum(np.subtract(y_residual[i], test_normalized[i])))

	    # if spe is greater than threshold then classify as attack by setting na to 1
	    if(spe[i] > threshold):
	        na[i] = 1
			#predicted_attacks_array[i] = 1

	predicted_attacks = pd.DataFrame(na,index=test_dataset.index,columns=['Prediction'])

	#print predicted_attacks.describe()

	# set detection Threshold
	threshold = 500;

	test_dataset = test_dataset.assign(ResidualVector=spe)
	test_dataset['ResidualVector'] = test_dataset['ResidualVector']*0.0005
	test_dataset['ResidualVector'].plot(figsize=(9,3),label='Residual Error')
	predicted_attacks['Prediction'].plot(label='PCA-based Prediction')
	test_dataset['ATT_FLAG'] = test_dataset['ATT_FLAG']*1
	test_dataset['ATT_FLAG'].plot(label='Actual Attack')
	#plt.plot([test_dataset['DatatimeIndex'].iloc[0], [test_dataset['DatatimeIndex'].iloc[-1]] ], [threshold, threshold],'k--')
	#plt.plot([test_dataset.index[0], test_dataset.index[-1] ], [threshold, threshold],'k--')
	#plt.plot(['2016-07-04 00:00:00', '2016-12-25 00:00:00' ], [threshold, threshold],'k--')
	plt.grid()
	plt.legend()
	plt.show()


	# DETERMINE PERFORMANCE METRICS
	print '\nPERFORMANCE METRICS\n'

	dfActualAttack = test_dataset['ATT_FLAG']
	dfPrediction = predicted_attacks['Prediction']
    # True Positive Rate aka Recall
	PositiveTotal = dfActualAttack[dfActualAttack == 1].sum()
	totalpoints = dfActualAttack.sum()
	print 'Total datapoints: ' + str(totalpoints)
	print 'Total positive values: ' + str(PositiveTotal)

    # print dfPrediction.shape
    # print binaryDF.shape

	TPtotal = dfPrediction[((dfActualAttack == 1) & (dfPrediction == 1))].sum()
	print 'Total predicted positives: ' + str(TPtotal)
	TPR = float(TPtotal)/float(PositiveTotal)
	print 'TPR: ' + str(TPR)
	Recall = TPR
	print 'Recall: ' + str(Recall)

    # Precision
	FPtotal = dfPrediction[((dfActualAttack == 0) & (dfPrediction == 1))].sum()
	Precision = float(TPtotal)/float(TPtotal + FPtotal)
	print 'Precision: ' + str(Precision)

	# compute confusion matrix (this isn't working as it should..)

	print '\n'
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for i in range(test_normalized.shape[0]):

	    if(labels[i] == 1 and na[i] == 1):
	        tp = tp + 1
	    if(labels[i] == 0 and na[i] == 1):
	        fp = fp + 1
	    if(labels[i] == 1 and na[i] == 0):
	        fn = fn + 1
	    if(labels[i] == 0 and na[i] == 0):
	        tn = tn + 1


	print "TP: {} ".format(tp)
	print "FP: {} ".format(fp)
	print "FN: {} ".format(fn)
	print "TN: {} ".format(tn)
