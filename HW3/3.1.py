#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
import seaborn as sns; sns.set() # graphing data

import warnings
warnings.filterwarnings("ignore")


# In[12]:


CSI_df = pd.read_csv('d:\homework\FIN3080\HW3\TRD_Index.csv')


# In[13]:
CSI_df = CSI_df[CSI_df["Indexcd"] == 300]


# 将日期列转换为日期时间类型，并设置为索引
CSI_df['Trddt'] = pd.to_datetime(CSI_df['Trddt'])
CSI_df.set_index('Trddt', inplace=True)

# 将每日收盘指数数据按月重新采样，并取每月最后一天的收盘指数
monthly_closing_prices = CSI_df['Clsindex'].resample('M').last()

# 创建新的DataFrame存储每月的收盘指数
monthly_df = pd.DataFrame({'Clsindex_M': monthly_closing_prices})

# 计算每月回报率
monthly_df['Return'] = monthly_df['Clsindex_M'].pct_change()


# In[15]:


from scipy.stats import skew, kurtosis

mean_return = monthly_df['Return'].mean()
std_return = monthly_df['Return'].std()
skewness = skew(monthly_df['Return'].dropna())  # 需要去除空值
kurt = kurtosis(monthly_df['Return'].dropna())  # 需要去除空值

# 打印摘要统计信息
print("Mean Return:", mean_return)
print("Standard Deviation of Return:", std_return)
print("Skewness of Return:", skewness)
print("Kurtosis of Return:", kurt)


# In[19]:


import matplotlib.pyplot as plt

# 假设你已经有了monthly_df DataFrame，包含了每月的回报率数据

# 创建直方图
plt.hist(monthly_df['Return'].dropna(), bins=100, range=(-1, 1), edgecolor='black')  # 需要去除空值，设置30个柱子，黑色边界线
plt.title('Histogram of Monthly Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

