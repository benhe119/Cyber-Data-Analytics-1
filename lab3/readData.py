"""
CDA Lab 3

Process data of CTU13 data

# download data from ...
"""

print 'Cyber Data Analytics\nLab 3 - Streaming Data\n'
### IMPORT PACKAGES ###
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.api import qqplot
import scipy.fftpack as fftpk
import scipy.signal as sgnl
from matplotlib import pyplot as plt

from pcapfile import savefile
from pcapfile.protocols.linklayer import ethernet
from pcapfile.protocols.network import ip

import sys
sys.path.insert(0, 'other/pcap2frame-master')
import pcap2frame

import pyshark

cap = pyshark.FileCapture('data/capture20110815-2.pcap.netflow.labeled')




# testcap = open('data/capture20110815-2.pcap.netflow.labeled', 'rb')
#
# print testcap

# capfile = savefile.load_savefile(testcap, verbose=True)
#
# eth_frame = ethernet.Ethernet(capfile.packets[0].raw())
# print eth_frame
#
# ip_packet = ip.IP(binascii.unhexlify(eth_frame.payload))
# print ip_packet

# 'capture20110815-2.pcap.netflow.labeled'
