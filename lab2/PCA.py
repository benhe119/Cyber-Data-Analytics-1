import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt

#read more: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

#PCA selection of subspace
def VisualizeComponentsPCA(trainDF):

	labels = trainDF['ATT_FLAG'].copy(deep=True)
	print labels.describe()
	#delete the flag column
	del trainDF['ATT_FLAG']

	#scale
	scaler = StandardScaler()
	training_normalized = scaler.fit_transform(trainDF)
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
	# plt.figure(20,figsize=(6,2.5))
	#
	# plt.title('Cumulative Variance of Principal Components')
	# plt.xlabel('Principal Components')
	# plt.ylabel('Cumulative Variance Captured')
	# plt.plot(x_axis, pca.explained_variance_ratio_.cumsum())
	# plt.grid()
	# plt.tight_layout()
	#plt.show()

	print trainDF.describe()

	### VISUALIZE DIFFERENT PCA COMPONENTS
	sizeX = 6
	sizeY = 3
	xlabel = 'time'
	ylabel = 'residual'

	# assign PCA components to trainDF
	PCAcomponentDF = pd.DataFrame(trainDF.index,index=trainDF.index)
	print PCAcomponentDF.describe()
	PCAcomponentDF = PCAcomponentDF.assign(PC3=pca_model[:,3])
	PCAcomponentDF = PCAcomponentDF.assign(PC4=pca_model[:,4])
	PCAcomponentDF = PCAcomponentDF.assign(PC5=pca_model[:,5])
	PCAcomponentDF = PCAcomponentDF.assign(PC6=pca_model[:,6])
	PCAcomponentDF = PCAcomponentDF.assign(PC7=pca_model[:,7])
	PCAcomponentDF = PCAcomponentDF.assign(PC8=pca_model[:,8])
	PCAcomponentDF = PCAcomponentDF.assign(PC9=pca_model[:,9])
	PCAcomponentDF = PCAcomponentDF.assign(PC10=pca_model[:,10])
	PCAcomponentDF = PCAcomponentDF.assign(PC11=pca_model[:,11])
	PCAcomponentDF = PCAcomponentDF.assign(PC12=pca_model[:,12])
	PCAcomponentDF = PCAcomponentDF.assign(PC13=pca_model[:,13])
	PCAcomponentDF = PCAcomponentDF.assign(PC14=pca_model[:,14])
	print 'added PCA components'
	print PCAcomponentDF.describe()
	# normalize
	# del PCAcomponentDF['DATETIME']
	# PCAcomponentsNormalized = scaler.fit_transform(PCAcomponentDF)
	# print 'normalize!'
	PCAcomponentDF = PCAcomponentDF.drop('DATETIME', axis=1).abs()

	print PCAcomponentDF.describe()

	shouldPlot = False

	if shouldPlot:
		#plot component 5 (for comparison to show the normal behaviour)
		plt.figure(3)
		PCAcomponentDF['PC3'].plot(figsize=(sizeX,sizeY))
		plt.title('PC3')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		plt.savefig('pc3_component.eps',div=300,format='eps')
		# plt.show();

		#plot component 6 or 7 (which are irregular)
		plt.figure(4)
		PCAcomponentDF['PC4'].plot(figsize=(sizeX,sizeY))
		plt.title('PC4')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		plt.savefig('pc4_component.eps',div=300,format='eps')
		# plt.show();

		plt.figure(5)

		PCAcomponentDF['PC5'].plot(figsize=(sizeX,sizeY))
		plt.title('PC5')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		plt.savefig('pc5_component.eps',div=300,format='eps')
		# plt.show();

		#plot component 6 or 7 (which are irregular)
		plt.figure(6)
		# trainDF = trainDF.assign(PC6=pca_model[:,6])
		PCAcomponentDF['PC6'].plot(figsize=(sizeX,sizeY))
		plt.title('PC6')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		# plt.show();

		#plot component 5 (for comparison to show the normal behaviour)
		plt.figure(7)
		# trainDF = trainDF.assign(PC7=pca_model[:,7])
		PCAcomponentDF['PC7'].plot(figsize=(sizeX,sizeY))
		plt.title('PC7')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		# plt.show();

		#plot component 6 or 7 (which are irregular)
		plt.figure(8)
		# trainDF = trainDF.assign(PC8=pca_model[:,8])
		PCAcomponentDF['PC8'].plot(figsize=(sizeX,sizeY))
		plt.title('PC8')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		# plt.show();

		#plot component 5 (for comparison to show the normal behaviour)
		# plt.figure(9)
		# # trainDF = trainDF.assign(PC9=pca_model[:,9])
		# PCAcomponentDF['PC9'].plot(figsize=(sizeX,sizeY))
		# plt.title('PC9')
		# plt.xlabel(xlabel)
		# plt.ylabel(ylabel)
		# plt.grid()
		# plt.show();

		#plot component 6 or 7 (which are irregular)
		# plt.figure(10)
		# # trainDF = trainDF.assign(PC10=pca_model[:,10])
		# PCAcomponentDF['PC10'].plot(figsize=(sizeX,sizeY))
		# plt.title('PC10')
		# plt.xlabel(xlabel)
		# plt.ylabel(ylabel)
		# plt.grid()
		# plt.show();

		#plot component 5 (for comparison to show the normal behaviour)
		plt.figure(11)
		# trainDF = trainDF.assign(PC11=pca_model[:,11])
		PCAcomponentDF['PC11'].plot(figsize=(sizeX,sizeY))
		plt.title('PC11')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		# plt.show();

		#plot component 6 or 7 (which are irregular)
		# plt.figure(12)
		# # trainDF = trainDF.assign(PC12=pca_model[:,12])
		# PCAcomponentDF['PC12'].plot(figsize=(sizeX,sizeY))
		# plt.title('PC12')
		# plt.xlabel(xlabel)
		# plt.ylabel(ylabel)
		# plt.grid()
		#plt.show();

		#plot component 6 or 7 (which are irregular)
		# plt.figure(13)
		# # trainDF = trainDF.assign(PC13=pca_model[:,13])
		# PCAcomponentDF['PC13'].plot(figsize=(sizeX,sizeY))
		# plt.title('PC13')
		# plt.xlabel(xlabel)
		# plt.ylabel(ylabel)
		# plt.grid()
		#plt.show()

	binaryDF = PCAcomponentDF.copy(deep=True)

	binaryDF['PC3'] = applyThreshold(binaryDF['PC3'],5.5)
	binaryDF['PC4'] = applyThreshold(binaryDF['PC4'],5)
	binaryDF['PC5'] = applyThreshold(binaryDF['PC5'],6)
	binaryDF['PC6'] = applyThreshold(binaryDF['PC6'],5)
	binaryDF['PC7'] = applyThreshold(binaryDF['PC7'],6.37)
	#binaryDF['PC8'] = applyThreshold(binaryDF['PC8'],4)
	binaryDF['PC11'] = applyThreshold(binaryDF['PC11'],3)

	# print 'descirbe DF'
	# print binaryDF.describe()
	#
	# print binaryDF['PC3']

	# plt.figure(12)
	# PCAcomponentDF['PC3'].plot(figsize=(sizeX,sizeY))
	shouldPlotBinary = False
	if shouldPlotBinary:
		plt.figure(1)
		binaryDF['PC3'].plot(figsize=(sizeX,sizeY))
		plt.figure(2)
		binaryDF['PC4'].plot(figsize=(sizeX,sizeY))
		plt.figure(3)
		binaryDF['PC5'].plot(figsize=(sizeX,sizeY))
		plt.figure(4)
		binaryDF['PC6'].plot(figsize=(sizeX,sizeY))
		plt.figure(5)
		binaryDF['PC7'].plot(figsize=(sizeX,sizeY))
		#plt.figure(6)
		#binaryDF['PC8'].plot(figsize=(sizeX,sizeY))
		plt.figure(7)
		binaryDF['PC11'].plot(figsize=(sizeX,sizeY))

	dfPrediction = binaryDF.copy(deep=True)
	dfPrediction = dfPrediction['PC3']*0.0
	print dfPrediction.describe()
	dfPrediction[((binaryDF["PC3"] == 1) | (binaryDF["PC4"] == 1) | (binaryDF["PC5"] == 1) | (binaryDF["PC6"] == 1) | (binaryDF["PC7"] == 1) | (binaryDF["PC11"] == 1))] = 1#0.1

	# DETERMINE PERFORMANCE METRICS
	print '\nPERFORMANCE METRICS\n'

	dfActualAttack = labels
	# dfPrediction

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

	if shouldPlotBinary:
		plt.figure(14)
		dfPrediction.plot(figsize=(sizeX,sizeY),label='Detected Attacks')
		labels.plot(figsize=(sizeX,sizeY),label='Actual Attack')
		plt.title('PCA Detection of Attacks')
		plt.legend()
		plt.grid()
		plt.ylabel('Attack Bool')
		plt.tight_layout()
		plt.savefig('pca_results.eps',div=300,format='eps')
	#plt.show()



	return dfPrediction

def applyThreshold(datafield, threshold):

	resultfield = datafield.copy(deep=True)

	# set all values to 0
	resultfield[:] = 0;

	resultfield[(datafield>threshold)] = 1

	return resultfield

#PCA detection
def PCA_detection(trainingDF,testDF):

	### TRAINING DATASET

	# normalize all signals (remove mean and divide by variance)
	scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
	training_normalized = scaler.fit_transform(trainingDF)

	# create dataframe of normalized dataset
	training_normalized_DF = pd.DataFrame(training_normalized, index=trainingDF.index, columns = trainingDF.columns)

	#print training_normalized_DF.describe()
	#print training_normalized.shape

	# transform the training data into its PCA components
	pca = decomposition.PCA()
	pca.fit(training_normalized)
	pca_training_model = pca.transform(training_normalized)

	# cumulative variance ratio
	pca.explained_variance_ratio_.cumsum()

	#these lines need to be remove for the test_dataset (because they don't contain that label)
	labels =  testDF['ATT_FLAG']
	# del test_dataset['ATT_FLAG']

	### TESTING DATASET

	# normalize all signals (remove mean and divide by variance)
	test_normalized = scaler.fit_transform(testDF)
	pca_test_model = pca.transform(test_normalized)

	# rename variable for easier reading code below
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

	# na will be set to 1 if the SPE is greater than the threshold
	na = np.zeros((test_normalized.shape[0]))

	# evaluate which residuals exceed a given threshold
	threshold = 500
	for i in range(test_normalized.shape[0]):
	    spe[i] = np.square(np.sum(np.subtract(y_residual[i], test_normalized[i])))

	    # if spe is greater than threshold then classify as attack by setting na to 1
	    if(spe[i] > threshold):
	        na[i] = 1

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
