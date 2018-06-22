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


# Read data from CTU113 dataset

# dataset 42
df_ctu13_42 = pd.read_csv("./data/capture20110815-2.pcap.netflow.labeled.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='DateFlowStart')

# Describe Data
# print df_ctu13_42.describe()
#print df_ctu13_42.head()
# print df_ctu13_42.columns.values
# print df_ctu13_42.columns

# ==============================================================================
#                       Import and Pre-process the data
# ==============================================================================

# we see that the host's IP address starts with 147.32.***.***
HostIP = '147.32.'

# only consider rows where the source (SRC) is the host IP
# df_ctu13_42 = df_ctu13_42[(df_ctu13_42["SrcIPAddr_Port"].str.contains(HostIP))]

# Only consider rows where (1) Destination IP = hostIP AND (2) Source IP != hostIP
df_ctu13_42 = df_ctu13_42[((df_ctu13_42["DstIPAddr_Port"].str.contains(HostIP)) & (~df_ctu13_42["SrcIPAddr_Port"].str.contains(HostIP)))]

# ==============================================================================
#                           Min-wise hashing
# ==============================================================================
# objective: estimate the distribution over the other ip addresses

# split source IP:port into IP address and port separately
df_ctu13_42[['SrcIPAddr','SrcPort']] = df_ctu13_42['SrcIPAddr_Port'].str.split(':', expand=True)

# df_reservoir = applyAlgorithmR(df_ctu13_42["SrcIPAddr"],10)

for resSize in [30]:#[5, 10, 15, 20, 30]:
    print '-------------------------------------------\nAnalyzing Reservoir ...'
    print ''
    print 'reservoir size:' + str(resSize)
    print ''
    applyReservoirSampling(df_ctu13_42["SrcIPAddr"],resSize)



# print df_reservoir.count
# print df_reservoir.describe()

# ==============================================================================
#                         Apply Min-Count hashing
# ==============================================================================
# w = 1000#number of column
# d = 10#number of row, hash count
# data = df_ctu13_42["SrcIPAddr"].as_matrix()
# cm = CountMinSketch(w, d)
# query = df_ctu13_42["SrcIPAddr"]
# for item in data:
#     cm.add(str(item))
# for item in query:
#     frequency_est = cm.estimate(str(item))
#     print str(item)+': '+str(frequency_est)
