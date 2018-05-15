# import packages

# download csv from http://www.sidc.be/silso/INFO/snytotcsv.php

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.graphics.api import qqplot

# input data

df = pd.read_csv("./sunspots.csv",delimiter=';',names=["midyear","sunactivity","MeanStdDev","Observation","Marker"])

print sm.datasets.sunspots.NOTE

# convert date to last day of the year i.e 1700-12-31 etc
df.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2017'))

print '\ndescribe'
print df.describe()
print '\nshape'
print df.shape

df.plot(figsize=(6,4));

print '\ndurbin watson statistics:'
print sm.stats.durbin_watson(df)

plt.show()
