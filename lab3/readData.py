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

# Read data from CTU113 dataset

# dataset 42
df_ctu13_42 = pd.read_csv("./data/capture20110815-2.pcap.netflow.labeled.csv", delimiter=',', parse_dates=True, dayfirst=True)#, index_col='Date flow start')

# Describe Data
# print df_ctu13_42.describe()
# print df_ctu13_42.head()
# print df_ctu13_42.columns.values
# print df_ctu13_42.columns

# ==============================================================================
#                   Apply minhashing a.k.a min-wise hashing
# ==============================================================================

# we see that the host's IP address is 147.32.***.***
HostIP = '147.32.'

# only consider rows where the source (SRC) is the host IP
df_ctu13_42 = df_ctu13_42[(df_ctu13_42["SrcIPAddr_Port"].str.contains(HostIP))]

print df_ctu13_42.describe()

# only consider fields where hostIP is associated with



# ==============================================================================
#                         Apply Min-Count hashing
# ==============================================================================
