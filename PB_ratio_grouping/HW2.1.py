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


# In[3]:


pershare_df= pd.read_csv('d:\homework\FIN3080\HW2\FI_T9.csv')


# In[4]:


#转换日期格式
price_df['Trdmnt'] = pd.to_datetime(price_df['Trdmnt'])
pershare_df['Accper'] = pd.to_datetime(pershare_df['Accper'])


# In[5]:


# 提取年月信息
price_df['YM'] = price_df['Trdmnt'].dt.to_period('M')
pershare_df['YM'] = pershare_df['Accper'].dt.to_period('M')


# In[6]:


# 确保两个表有相同的股票代码列
common_stock_codes = pd.merge(price_df[['Stkcd']].drop_duplicates(), pershare_df[['Stkcd']].drop_duplicates(), how='inner', on='Stkcd')

# 根据相同的股票代码和年月进行合并
merged_df = pd.merge(price_df[price_df['Stkcd'].isin(common_stock_codes['Stkcd'])], pershare_df[pershare_df['Stkcd'].isin(common_stock_codes['Stkcd'])], on=['Stkcd', 'YM'], how='left')


# In[7]:


# 设置最大显示行数，None表示显示所有行
pd.set_option('display.max_rows',100)


# In[8]:


merged_df['F091001A'] = merged_df['F091001A'].fillna(method='bfill')


# In[9]:


# 计算PB ratio
merged_df['PB ratio'] = merged_df['Mclsprc'] / (merged_df['F091001A'])


# In[21]:


# 计算百分位数
percentile_5 = merged_df['PB ratio'].quantile(0.05)
percentile_95 = merged_df['PB ratio'].quantile(0.95)

# 排除在百分位5以下和95以上的PB值
filtered_PB_df = merged_df[(merged_df['PB ratio'] > percentile_5) & (merged_df['PB ratio'] < percentile_95)]


# In[10]:


ROE_df= pd.read_csv('d:\homework\FIN3080\HW2\FI_T5.csv')


# In[11]:


SV_df= pd.read_csv('d:\homework\FIN3080\HW2\STK_MKT_STKBTAL.csv')


# In[13]:


#转换日期格式
ROE_df['Accper'] = pd.to_datetime(ROE_df['Accper'])
SV_df['TradingDate'] = pd.to_datetime(SV_df['TradingDate'])
# 提取年月信息
ROE_df['YM'] = ROE_df['Accper'].dt.to_period('M')
SV_df['YM'] = SV_df['TradingDate'].dt.to_period('M')


# In[14]:





# In[19]:


ROE_df_2010 = ROE_df[ROE_df['YM']== '2010-12' ]


# In[20]:





# In[22]:


filtered_PB_df_2010 = filtered_PB_df[filtered_PB_df['YM']== '2010-12' ]
filtered_PB_df_2010.head(10)


# In[23]:


# 定义函数来删除 Typrep 为 'B' 的行
def filter_rows(group):
    if 'A' in group['Typrep'].values and 'B' in group['Typrep'].values:
        return group[group['Typrep'] != 'B']
    else:
        return group

# 按 Stkcd 分组，并应用函数来删除 Typrep 为 'B' 的行
ROE_df_2010 = ROE_df_2010.groupby('Stkcd').apply(filter_rows).reset_index(drop=True)
filtered_PB_df_2010 = filtered_PB_df_2010.groupby('Stkcd').apply(filter_rows).reset_index(drop=True)


# In[24]:





# In[28]:


SV_df.rename(columns={'Symbol': 'Stkcd'}, inplace=True)


# In[25]:


# 确保两个表有相同的股票代码列
common_stock_codes = pd.merge(filtered_PB_df_2010[['Stkcd']].drop_duplicates(), SV_df[['Stkcd']].drop_duplicates(), how='inner', on='Stkcd')

# 根据相同的股票代码和年月进行合并
merged_df_1 = pd.merge(filtered_PB_df_2010[filtered_PB_df_2010['Stkcd'].isin(common_stock_codes['Stkcd'])],SV_df[SV_df['Stkcd'].isin(common_stock_codes['Stkcd'])], on=['Stkcd', 'YM'], how='left')


# In[26]:





# In[29]:


# 确保两个表有相同的股票代码列
common_stock_codes = pd.merge(merged_df_1[['Stkcd']].drop_duplicates(), SV_df[['Stkcd']].drop_duplicates(), how='inner', on='Stkcd')

# 根据相同的股票代码和年月进行合并
merged_df_2 = pd.merge(merged_df_1[merged_df_1['Stkcd'].isin(common_stock_codes['Stkcd'])], SV_df[SV_df['Stkcd'].isin(common_stock_codes['Stkcd'])], on=['Stkcd', 'YM'], how='left')


# In[30]:





# In[31]:


merged_df_2.rename(columns={'F050504C': 'ROE'}, inplace=True)


# In[33]:


merged_df_2.dropna(subset=['ROE','Volatility','PB ratio'],inplace=True)


# In[34]:


import pandas as pd
from sklearn.linear_model import LinearRegression

# 使用Skicit-learn的LinearRegression模型进行回归
X = merged_df_2[['ROE', 'Volatility']]
y = merged_df_2['PB ratio']

# 初始化线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 打印回归系数和截距
print('β:', model.coef_)
print('α:', model.intercept_)


# In[ ]:




