import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import numpy as np
from scipy.signal import find_peaks

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# 读取数据
df_raw = pd.read_csv('dataset/AQ.csv', delimiter=';', decimal=',')

# 假设需要处理的列是所有除了第一列的列
cols_data = df_raw.columns[2:]
print("Available columns in the dataset:", df_raw.columns[2:])
df_data = df_raw[cols_data]

# 确保数据是数值型
train_data = df_data[:12*30*24]

# 或者替换字符串值
train_data = train_data.replace('cv', np.nan)  # 例如把'cv'替换为NaN

# 填充缺失值
train_data = train_data.fillna(0)  # 或者用其它值进行填充

# 标准化
scaler = StandardScaler()
scaler.fit(train_data.values)  # 现在应该能正常工作了
# Calculate autocorrelation coefficients
acf_values = acf(train_data.iloc[:, 0], nlags=48)

plt.plot(train_data)
plt.title("AQ Dataset")
plt.show()

# Plot the bar chart
lags = np.arange(len(acf_values))
plt.figure(figsize=(3, 2.5))
plt.bar(lags, acf_values)  # use bar plot

plt.xlim([0, 48])
plt.ylim([0, 1])

plt.xlabel("Lags", fontsize=12)
plt.ylabel("Autocorrelation", fontsize=12)

plt.grid(True)
plt.show()

# Find peaks in autocorrelation function
peaks, _ = find_peaks(acf_values)

# Identify the lag corresponding to the highest peak
main_period = peaks[0]

print("Main period detected at lag:", main_period)