#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
import seaborn as sns; sns.set() # graphing data

import warnings
warnings.filterwarnings("ignore")


# In[2]:


price_df = pd.read_csv('d:\homework\FIN3080\HW2\TRD_Mnth.csv')
pershare_df= pd.read_csv('d:\homework\FIN3080\HW2\FI_T9.csv')
#转换日期格式
price_df['Trdmnt'] = pd.to_datetime(price_df['Trdmnt'])
pershare_df['Accper'] = pd.to_datetime(pershare_df['Accper'])
# 提取年月信息
price_df['YM'] = price_df['Trdmnt'].dt.to_period('M')
pershare_df['YM'] = pershare_df['Accper'].dt.to_period('M')
# 确保两个表有相同的股票代码列
common_stock_codes = pd.merge(price_df[['Stkcd']].drop_duplicates(), pershare_df[['Stkcd']].drop_duplicates(), how='inner', on='Stkcd')

# 根据相同的股票代码和年月进行合并
merged_df = pd.merge(price_df[price_df['Stkcd'].isin(common_stock_codes['Stkcd'])], pershare_df[pershare_df['Stkcd'].isin(common_stock_codes['Stkcd'])], on=['Stkcd', 'YM'], how='left')

merged_df['F091001A'] = merged_df['F091001A'].fillna(method='bfill')
merged_df['PB ratio'] = merged_df['Mclsprc'] / (merged_df['F091001A'])
# 计算百分位数
percentile_5 = merged_df['PB ratio'].quantile(0.05)
percentile_95 = merged_df['PB ratio'].quantile(0.95)

# 排除在百分位5以下和95以上的PB值
filtered_PB_df = merged_df[(merged_df['PB ratio'] > percentile_5) & (merged_df['PB ratio'] < percentile_95)]


# In[3]:


# 定义函数来删除 Typrep 为 'B' 的行
def filter_rows(group):
    if 'A' in group['Typrep'].values and 'B' in group['Typrep'].values:
        return group[group['Typrep'] != 'B']
    else:
        return group

# 按 Stkcd 分组，并应用函数来删除 Typrep 为 'B' 的行.reset_index(drop=True)
filtered_PB_df = filtered_PB_df.groupby('Stkcd').apply(filter_rows)


# In[4]:





# In[6]:


YM_df = filtered_PB_df.groupby('YM')


# In[ ]:





# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

deciles = filtered_PB_df.groupby('YM')['PB ratio'].apply(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))





# In[9]:





# In[22]:


filtered_PB_df['PB ratio'] = filtered_PB_df['PB ratio'].astype(np.float32)


# In[24]:


# 将十分位数加入 DataFrame
filtered_PB_df['Decile'] = filtered_PB_df.groupby('YM')['PB ratio'].transform(lambda x: pd.qcut(x, q=10, labels=False))



# In[25]:





# In[26]:


# 计算每个月每个组的平均回报率
monthly_returns = filtered_PB_df.groupby(['YM', 'Decile'])['Mretnd'].mean().reset_index()

# 计算每个组2010年1月到2023年12月的月平均回报的平均值
average_returns = monthly_returns.groupby('Decile')['Mretnd'].mean()


# In[27]:





# In[28]:


plt.figure(figsize=(10, 6))
average_returns.plot(kind='bar', color='skyblue')
plt.title('Average Monthly Returns for Ten Portfolios (2010-2023)')
plt.xlabel('Portfolio Decile')
plt.ylabel('Average Monthly Return')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 打印结果
print(average_returns)

