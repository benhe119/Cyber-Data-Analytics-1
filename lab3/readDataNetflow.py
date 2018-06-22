import pandas as pd
items = pd.read_csv("./data/2013-08-20_capture-win10.netflow.csv", delimiter=',', parse_dates=True, dayfirst=True, index_col='StartTime')

print items['SrcAddr']