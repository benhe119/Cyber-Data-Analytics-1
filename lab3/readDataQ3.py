print 'Cyber Data Analytics\nLab 3 - Streaming Data - Question 3 & 4\n'
### IMPORT PACKAGES ###
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import operator
import time
# import datetime as dt
from datetime import datetime as dt
from discretize_data import discretizeBinary, discretizeSAX
from N_gram import N_gram
import math

# ==============================================================================
#                    3 Botnet flow data discretization
# ==============================================================================

columns = ['DateFlowStart','Duration','Protocol', 'SrcIPAddr_Port','Dir','DstIPAddr_Port','Flags','Tos','Packets','Bytes','Flows','Labels']
dtypes = ['datetime','float','str','str','str','str','str','int','int','int','int','str']
dtypes = {'DateFlowStart': 'str','Duration': 'float','Protocol': 'str', 'SrcIPAddr_Port': 'str','Dir': 'str','DstIPAddr_Port': 'str','Flags': 'str','Tos': 'str','Packets': 'str','Bytes': 'str','Flows': 'str', 'Lables': 'str'}
parse_dates = ['DateFlowStart']
# load data
print ' reading csv...'

#reading the dataset in chuncks
chunksize = 20000
# reader = pd.read_csv("./data/capture20110818.pcap.netflow.labeled.csv",delimiter=',',names=columns,dtype=dtypes,parse_dates=parse_dates,header=0, chunksize=chunksize)
reader = pd.read_csv("./data/test.netflow",delimiter=',',names=columns,dtype=dtypes,parse_dates=parse_dates,header=0, chunksize=chunksize)
count = 1

#set coeficients
C1 = 1.0;
C2 = 1.0;

#keep track of hosts that have labled botnet traffic
infectedHosts = []							#used to output all IP addresses of infected hosts
chosenInfectedHost = '147.32.96.69'			#infected host IP we used to train the N-gram. NOTE: for question 3 set this value to '', because all data will then get to the same variable for plotting
infectedHostData = None						#variable that stores the SAX descretized data
allHostsExcludingInfectedHostData = None		#variable that stores the data off all other hosts

#set start date of first entry form the netflow
d = pd.to_datetime({'year':[2011], 'month':[8], 'day':[18], 'hour':[10], 'minute':[19],'second':[13]})


#loop through all chuncks
for chunk in reader:

	# split IP address and port number
	chunk['SrcIP'], chunk['SrcPort'] = chunk['SrcIPAddr_Port'].str.split(':',1).str
	chunk['DstIP'], chunk['DstPort'] = chunk['DstIPAddr_Port'].str.split(':',1).str

	#parsing dates
	#print chunk['DateFlowStart']
	chunk['DateFlowStart'] = pd.to_datetime(chunk['DateFlowStart']) - d[0]

	# get time in seconds
	chunk['timeLong'] = chunk['DateFlowStart'].dt.total_seconds()

	# create temporary entry
	chunk['lastAttackIP'] = chunk['timeLong'].shift(-1) - chunk['timeLong']
	chunk['lastAttackIP'].clip(-10,10)

	#determine cost function based on properties
	chunk['costVal'] = C2*chunk['lastAttackIP'].astype(float)
	chunk['costVal'] = chunk['costVal'].clip(-0.0,10.0)

	#decided in what variable we want to store the entries (training hosts, or other hosts)
	for index, row in chunk.iterrows():
		#check for IP's with botnet we hadn't stored yet
		if row['SrcIP'] not in infectedHosts and "Botnet" in row['Labels']:
			infectedHosts.append(row['SrcIP'])

	# remove columns we don't need anymore to save memory (we keep the costVal we just computed, as well as the Labels to check later for TP/FP and the DateFlowStart as a primary key)
	columnsToDrop = ['Flows','SrcPort','DstPort','SrcIPAddr_Port','DstIPAddr_Port', 'Duration','Protocol','Dir','Flags','Tos','Packets','Bytes','timeLong','lastAttackIP']
	chunk.drop(columns=columnsToDrop,inplace=True)

	infectedSet = chunk[chunk["SrcIP"].str.contains(chosenInfectedHost)]
	restSet 	= chunk[~chunk["SrcIP"].str.contains(chosenInfectedHost)]

	#if infectedHostData is not None:
	infectedHostData 					= pd.concat([infectedHostData,infectedSet])
	allHostsExcludingInfectedHostData 	= pd.concat([allHostsExcludingInfectedHostData,restSet])

	print ' chunk read ' + str(((count * chunksize)/51800)) +"%" + " (" + str(count * chunksize) + ")"
	count = count + 1

#log to console
print 'Done reading data'
print ''
print '================ INFECTED HOSTS IPs ================'
print infectedHosts
print ''

infectedHostData['costVal'] = 	infectedHostData['costVal'].fillna(0)
allHostsExcludingInfectedHostData['costVal'] = allHostsExcludingInfectedHostData['costVal'].fillna(0)

print infectedHostData['costVal'].sort_values(ascending=False)
print allHostsExcludingInfectedHostData['costVal'].sort_values(ascending=False)


#SAX both datasets
print '================ SAX ================'
xTrain, yTrain = discretizeSAX('costVal',infectedHostData)
print 'discretization training set done!'
xTest, yTest = discretizeSAX('costVal',allHostsExcludingInfectedHostData)
print 'discretization test set done!'
print ''


# Question 3 is done here. We can plot the xText and yText as a discretization example

#N gram for all other hosts
print '================ N-gram ================'
listWithAssumedBotnetTraffic = N_gram(yTrain, yTest, 2, 0.1)
print 'N-gram analysis completed, start count TP/FP'

# # show difference in length of data
# lengthAnomalyList = len(anomalyList)
# lengthResultsList = len(timestampsTrain2)
#
# remove last values in timestamp until it matches the anomaly list length
# this is because the discretization uses the data DIV n (of n-gram) and is
# therefore different in length
while (len(listWithAssumedBotnetTraffic) != len(yTest)):
	yTest = yTest[:-1] # remove last entry

# print "Length yTest:" + str(len(yTest))
# print "Length listWithAssumedBotnetTraffic:" + str(len(listWithAssumedBotnetTraffic))
# print "Length allHostsExcludingInfectedHostData:" + str(len(allHostsExcludingInfectedHostData))

count 	= 0
tp 		= 0
fp 		= 0
fn 		= 0
tn 		= 0



for flag in listWithAssumedBotnetTraffic:
	#check for true positives
	if flag == 1 and "Botnet" in allHostsExcludingInfectedHostData.iloc[count]['Labels']:
		#we found a true positive
		print "TP => " + str(allHostsExcludingInfectedHostData.iloc[count]['DateFlowStart']) + " from " + allHostsExcludingInfectedHostData.iloc[count]['SrcIP'] + " to " + allHostsExcludingInfectedHostData.iloc[count]['DstIP']
		tp = tp + 1
	elif flag == 1 and "Botnet" not in allHostsExcludingInfectedHostData.iloc[count]['Labels']:
		#we found a false positive
		#print "FP => " + str(allHostsExcludingInfectedHostData.iloc[count]['DateFlowStart']) + " from " + allHostsExcludingInfectedHostData.iloc[count]['SrcIP'] + " to " + allHostsExcludingInfectedHostData.iloc[count]['DstIP']
		fp = fp + 1
	elif flag == 0 and "Botnet" not in allHostsExcludingInfectedHostData.iloc[count]['Labels']:
		#we found a false negative
		#print "FN => " + ytest[count]['DateFlowStart'] + " from " + ytest[count]['SrcIP'] + " to " + ytest[count]['DstIP']
		fn = fn + 1
	else:
		tn = tn + 1

	count = count + 1


#print listWithAssumedBotnetTraffic
print 'N-gram done'
print 'TP count:' + str(tp)
print 'FP count:' + str(fp)
print 'FN count:' + str(fn)
print 'TN count:' + str(tn)
