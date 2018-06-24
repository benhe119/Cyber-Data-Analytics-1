"""
CDA Lab 3

Process data of CTU13 data

download data from https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46/

file: 'capture20110815-2.pcap.netflow.labeled'
parsed using: parse_data.py

"""

print 'Cyber Data Analytics\nLab 3 - Streaming Data\n'
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

from reservoirSampling import applyAlgorithmR, applyReservoirSampling
from CountMinSketch import CountMinSketch
import operator
import time
# import datetime as dt
from datetime import datetime as dt
from discretize_data import discretizeBinary, discretizeSAX


# Read dataset 42 from CTU-13 dataset
# df_ctu13_42 = pd.read_csv("./data/capture20110815-2.pcap.netflow.labeled.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DateFlowStart')
#
# # ==============================================================================
# #                       Import and Pre-process the data
# # ==============================================================================
#
# # we see that the host's IP address starts with 147.32.***.***
# HostIP = '147.32.'
#
# # Only consider rows where (1) Destination IP = hostIP AND (2) Source IP != hostIP
# df_ctu13_42 = df_ctu13_42[((df_ctu13_42["DstIPAddr_Port"].str.contains(HostIP)) & (~df_ctu13_42["SrcIPAddr_Port"].str.contains(HostIP)))]

# ==============================================================================
#                        1   Min-wise hashing
# ==============================================================================

# # split source IP:port into IP address and port separately
# df_ctu13_42[['SrcIPAddr','SrcPort']] = df_ctu13_42['SrcIPAddr_Port'].str.split(':', expand=True)
#
# # df_reservoir = applyAlgorithmR(df_ctu13_42["SrcIPAddr"],10)
#
# for resSize in [30]:#[5, 10, 15, 20, 30]:
#     print '-------------------------------------------\nAnalyzing Reservoir ...'
#     print ''
#     print 'reservoir size:' + str(resSize)
#     print ''
#     applyReservoirSampling(df_ctu13_42["SrcIPAddr"],resSize)

# ==============================================================================
#                       2  Apply Min-Count hashing
# ==============================================================================

# w = 1000    #number of column
# d = 10      #number of row, hash count
# data = df_ctu13_42["SrcIPAddr"].as_matrix()
# cm = CountMinSketch(w, d)
# query = df_ctu13_42["SrcIPAddr"]
# ips = {}
# for item in data:
#     cm.add(str(item))
# for item in query:
#     frequency_est = cm.estimate(str(item))
#     ips[str(item)] = frequency_est
#
# print "Printing 30 most used IP's Min-Count hashing"
# ips_sorted = sorted(ips.items(), key=operator.itemgetter(1), reverse=True)
# for ip in ips_sorted[:30]:
# 	print ip[0] + "  =>  " + str(ip[1])

# ==============================================================================
#                    3 Botnet flow data discretization
# ==============================================================================

columns = ['DateFlowStart','Duration','Protocol', 'SrcIPAddr_Port','Dir','DstIPAddr_Port','Flags','Tos','Packets','Bytes','Flows','Labels']
dtypes = ['datetime','float','str','str','str','str','str','int','int','int','int','str']
dtypes = {'DateFlowStart': 'str','Duration': 'float','Protocol': 'str', 'SrcIPAddr_Port': 'str','Dir': 'str','DstIPAddr_Port': 'str','Flags': 'str','Tos': 'str','Packets': 'str','Bytes': 'str','Flows': 'str', 'Lables': 'str'}
parse_dates = ['DateFlowStart']
# load data
print ' reading csv...'
df_ctu13_10 = pd.read_csv("./data/capture20110815_short.csv",delimiter=',',names=columns,dtype=dtypes,parse_dates=parse_dates,header=0)#, dayfirst=True)#, index_col='StartTime')
print ' done!\n'

# split port and IP address
print ' adding and dropping data fields...'
# split IP address and port number
df_ctu13_10['SrcIP'], df_ctu13_10['SrcPort'] = df_ctu13_10['SrcIPAddr_Port'].str.split(':',1).str
df_ctu13_10['DstIP'], df_ctu13_10['DstPort'] = df_ctu13_10['DstIPAddr_Port'].str.split(':',1).str
# remove flow column
columnsToDrop = ['Flows','SrcPort','DstPort','SrcIPAddr_Port','DstIPAddr_Port']
df_ctu13_10.drop(columns=columnsToDrop,inplace=True)
print ' done!\n'

print ' parsing dates...'
df_ctu13_10['DateFlowStart'] = pd.to_datetime(df_ctu13_10['DateFlowStart']) - pd.to_datetime(df_ctu13_10['DateFlowStart'][0])
# df_ctu13_10['DateFlowStart'] = df_ctu13_10['DateFlowStart'] - df_ctu13_10['DateFlowStart'][0]
print ' done!\n'

# print df_ctu13_10.head(10)

# print df_ctu13_10['DateFlowStart'].dtype

# df_ctu13_10['DateFlowStartShift'] = df_ctu13_10['DateFlowStart'].shift(-1)
# print df_ctu13_10['DateFlowStartShift'].dtype

# print df_ctu13_10['DateFlowStart'][0]

# df_ctu13_10['DateFlowStart'] = df_ctu13_10['DateFlowStart'].shift(-1) - df_ctu13_10['DateFlowStart']

# print df_ctu13_10['DateFlowStart'].dtype

# get time in seconds
print ' converting datetime to seconds...'
df_ctu13_10['timeLong'] = df_ctu13_10['DateFlowStart'].dt.total_seconds()
print ' done!\n'

print df_ctu13_10.describe()
print df_ctu13_10.head(30)

print '-------------------------------------'



#df_ctu13_10['StartTime'] = df_ctu13_10['StartTime'].astype(int)

print df_ctu13_10.columns.values

print df_ctu13_10.dtypes



# create temporary entry
df_ctu13_10['lastAttackIP'] = df_ctu13_10['timeLong'].shift(-1) - df_ctu13_10['timeLong']



# print df_ctu13_10.head(5)

C1 = 1.0;
C2 = 1.0;

# print 'asdfasdfasdfasdf\n\n\n\n\n\n\n5'

print df_ctu13_10['lastAttackIP'].describe()

print df_ctu13_10['TotPkts'].describe()

df_ctu13_10['lastAttackIP'].clip(-10,10)

print df_ctu13_10['lastAttackIP'].describe()


# create a cost value from these two features C1*(df_ctu13_10['TotPkts'].astype(float)-415)

df_ctu13_10['costVal'] = C2*df_ctu13_10['lastAttackIP'].astype(float)

df_ctu13_10['costVal'] = df_ctu13_10['costVal'].clip(-0.0,10.0)

print df_ctu13_10['costVal'].describe()

# discretize the extracted features

# apply SAX
xlist, ylist = discretizeSAX('costVal',df_ctu13_10)

#### LOOK HERE!

# print xlist



#




#
# # print df_ctu13_10.head(15)
#
# # print df_ctu13_10['StartTime'].dtype
#
#
#
# df_ctu13_10['Day'], df_ctu13_10['Time'] = df_ctu13_10['DayandTime'].str.split(' ',1).str
# df_ctu13_10['Hour'], df_ctu13_10['MinutesExt'] = df_ctu13_10['Time'].str.split(':',1).str
# df_ctu13_10['Minutes'], df_ctu13_10['Seconds'] = df_ctu13_10['MinutesExt'].str.split(':',1).str
#
# df_ctu13_10['daysInMonth'] = df_ctu13_10['Month']
# # df_ctu13_10['daysInMonth'] =
#
# df_ctu13_10['timeLong'] = (df_ctu13_10['Month'].astype(float)-1)*(24.0*3600) + df_ctu13_10['Seconds'].astype(float) + 60.0*(df_ctu13_10['Minutes'].astype(float)) + 3600.0*(df_ctu13_10['Hour'].astype(float)) + (24*3600.0)*(df_ctu13_10['Day'].astype(float))
# df_ctu13_10['deltaTfeature'] = df_ctu13_10['SrcAddr'] + ';' + df_ctu13_10['DstAddr'] + df_ctu13_10['timeLong'].astype(str)
#
# # print df_ctu13_10.head(15)
#
# # array = pd.to_numeric(df_ctu13_10['DayandTime'])
# # print array
#
# # df_ctu13_10['StartTime'] = df_ctu13_10['StartTime'].apply(int) #.astype(str).astype(int)
# # print "successs"
#
# # create list of IP address combinations
# ipComboList = [];
#
# # feature extraction
# # for index, row in df_ctu13_10.iterrows():
#
#     #ipSrcDst = row["SrcAddr"] + ":" + row["DstAddr"] + ";" + pd.to_numeric(index)
#     #print ipSrcDst
#
#     # 10.2.3.4:1.4.2.4

































###
