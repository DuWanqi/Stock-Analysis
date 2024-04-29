#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
import seaborn as sns; sns.set() # graphing data

import warnings
warnings.filterwarnings("ignore")


# In[16]:


import openpyxl


# In[84]:


week_df = pd.read_csv('d:\homework\FIN3080\HW3\TRD_Week.csv')


# In[85]:


def filter_rows(group):
    return group[(group['Markettype'] == 4) | (group['Markettype'] == 1) | (group['Markettype'] == 64)]

week_df_1 = week_df.groupby('Trdwnt').apply(filter_rows).reset_index(drop=True)


# In[11]:


# 按 'Trdwnt' 列分组，计算 'Wretnd' 列的平均值
average_wretnd = week_df.groupby('Trdwnt')['Wretnd'].mean()


# In[19]:


average_wretnd_df = average_wretnd.reset_index()


# In[120]:


rf_df = pd.read_excel('d:\homework\FIN3080\HW3\weekly_risk_free_rate.xlsx')


# In[122]:


condition = ~(rf_df['trading_date_yw'].isin(["2017-01-01", "2018-01-01",  "2020-01-01",  "2022-01-01"]))

filtered_rf_df = rf_df[condition].reset_index(drop=True)


# In[124]:


merged_df = pd.concat([average_wretnd_df, filtered_rf_df], axis=1)


# In[182]:


# 确保两个表有相同的股票代码列
common_w = pd.merge(week_df_1[['Trdwnt']].drop_duplicates(), merged_df[['Trdwnt']].drop_duplicates(), how='inner', on='Trdwnt')

# 根据相同的股票代码和年月进行合并
total_df = pd.merge(week_df_1[week_df_1['Trdwnt'].isin(common_w['Trdwnt'])], merged_df[merged_df['Trdwnt'].isin(common_w['Trdwnt'])], on=['Trdwnt'], how='left')


# In[184]:


# 将 'Trdwnt' 列转换为日期时间类型
total_df['Trdwnt'] = pd.to_datetime(total_df['Trdwnt'] + '-1', format='%Y-%W-%w')

# 根据 'Trdwnt' 列分成相对均匀的三组
total_df['Trdwnt_group'] = pd.qcut(total_df['Trdwnt'], q=3, labels=['Group1', 'Group2', 'Group3'])


# In[187]:


mean_value = total_df['Wretnd_x'].mean()
total_df['Wretnd_x'].fillna(mean_value, inplace=True)


# In[188]:


import statsmodels.api as sm

# 按照Stkcd和Trdwnt_group分组，筛选出Group1的数据
group1_data = total_df[total_df['Trdwnt_group'] == 'Group1']

# 创建一个空的DataFrame来存储回归结果
beta_df = pd.DataFrame(columns=['Stkcd', 'Beta'])

# 对每个Stkcd的Group1数据进行回归分析，计算β值
for stkcd, group_data in group1_data.groupby('Stkcd'):
    X = group_data['Wretnd_y']
    y = group_data['Wretnd_x']
    
    X = sm.add_constant(X)  # 添加常数项
    model = sm.OLS(y, X).fit()  # 普通最小二乘法拟合
    beta = model.params['Wretnd_y']  # 获取斜率参数，即β值
    
    # 将结果添加到beta_df中
    beta_df = beta_df.append({'Stkcd': stkcd, 'Beta': beta}, ignore_index=True)

# 将得到的β值回填到DataFrame中对应的Stkcd行上
total_df = total_df.merge(beta_df, on='Stkcd', how='left')


# In[216]:


# 将 Beta 列分成 10 组并添加分组标签
total_df['Beta_group'] = pd.qcut(total_df['Beta'], q=10, labels=False, duplicates='drop')



# In[222]:


# 对于Group2 对于每一周 计算每一组的平均值 收益 rpt  再新增一列 Wretnd_y-risk_free_return  再回归 用上面同样的方法

total_df['rm_rf'] = total_df['Wretnd_y'] - total_df["risk_free_return"]
# 筛选出 Trdwnt_group 为 Group2 的数据
group2_data = total_df[total_df['Trdwnt_group'] == 'Group2']



# In[224]:


# 按照 Trdwnt 和 Trdwnt_group 分组，计算每组每周的收益平均值
group2_weekly_mean = group2_data.groupby(['Trdwnt', 'Beta_group'])['Wretnd_x'].mean().reset_index()


# In[226]:


# 合并计算得到的每组每周收益平均值到原始的 group2_data 中
group2_data = group2_data.merge(group2_weekly_mean, on=['Trdwnt', 'Beta_group'], how='left', suffixes=('', '_mean'))


# In[228]:


group2_data['rp_rf'] = group2_data['Wretnd_x_mean']-group2_data['risk_free_return']


# In[229]:


# 创建一个空的 DataFrame 来存储回归结果
beta_df_2 = pd.DataFrame(columns=['Beta_group', 'Alpha_p', 'Alpha_p_t_value', 'Beta_p', 'Beta_p_t_value', 'R_squared'])

# 对每个 betagroup 的 Group2 数据进行回归分析，计算 αp、αp 的 t 值、βp、βp 的 t 值和 R-squared
for beta_group, group_data in group2_data.groupby(['Beta_group']):
    X = group_data['rm_rf']
    y = group_data['rp_rf']
    
    X = sm.add_constant(X)  # 添加常数项
    model = sm.OLS(y, X).fit()  # 普通最小二乘法拟合
    
    # 获取回归结果
    alpha_p = model.params['const']  # αp
    alpha_p_t_value = model.tvalues['const']  # αp 的 t 值
    beta_p = model.params['rm_rf']  # βp
    beta_p_t_value = model.tvalues['rm_rf']  # βp 的 t 值
    r_squared = model.rsquared  # R-squared
    
    # 将结果添加到 beta_df_2 中
    beta_df_2 = beta_df_2.append({'Beta_group': beta_group, 'Alpha_p': alpha_p, 
                                  'Alpha_p_t_value': alpha_p_t_value, 'Beta_p': beta_p, 
                                  'Beta_p_t_value': beta_p_t_value, 'R_squared': r_squared}, ignore_index=True)

# # 将得到的结果合并到 total_df 中
# total_df = total_df.merge(beta_df_2, on=['Beta_group', 'Trdwnt'], how='left')

print(beta_df_2)
# In[231]:


group3_data = total_df[total_df['Trdwnt_group'] == 'Group3']

# 按照 Trdwnt 和 Trdwnt_group 分组，计算每组每周的收益平均值
group3_weekly_mean = group3_data.groupby(['Trdwnt', 'Beta_group'])['Wretnd_x'].mean().reset_index()
# 合并计算得到的每组每周收益平均值到原始的 group2_data 中
group3_data = group3_data.merge(group3_weekly_mean, on=['Trdwnt', 'Beta_group'],how='left',suffixes=('', '_mean'))


# In[232]:


group3_data['rp_rf'] = group3_data['Wretnd_x_mean']-group3_data['risk_free_return']


# In[234]:


rp_rf_mean = group3_data.groupby(['Beta_group'])['rp_rf'].mean().reset_index()


# In[235]:


group3_data = group3_data.merge(rp_rf_mean, on=['Beta_group'], how='left', suffixes=('', '_mean'))


# In[240]:


unique_df = group3_data.drop_duplicates(subset=['rp_rf_mean'])


# In[242]:


unique_df.dropna(inplace=True)


# In[244]:


# 假设你的两列分别是 column1 和 column2
column1_unique = unique_df['Beta_group']
column2_unique = unique_df['rp_rf_mean']

# 使用 pd.concat() 合并两列为一个新的 DataFrame
new3_df = pd.concat([column1_unique, column2_unique], axis=1)



# In[245]:


# 假设你的两列分别是 column3 和 column4
column3_unique = beta_df_2['Beta_group']
column4_unique = beta_df_2['Beta_p']

# 使用 pd.concat() 合并两列为一个新的 DataFrame
new4_df = pd.concat([column3_unique, column4_unique], axis=1)



# In[246]:


regression3_df = pd.merge(new3_df, new4_df, on='Beta_group')


# In[254]:


# 创建一个空的 DataFrame 来存储回归结果
beta_df_3 = pd.DataFrame(columns=[ 'Gamma_0', 'Gamma_1','Gamma_0_t_value','Gamma_1_t_value', 'R_squared', 'F_statistic', 'P_value'])

# 对每个 beta_group 进行回归分析，y 是平均收益 rp_rf_mean，X 是 Beta_p
X = regression3_df['Beta_p']
y = regression3_df['rp_rf_mean']

X = sm.add_constant(X)  # 添加常数项
model = sm.OLS(y, X).fit()  # 普通最小二乘法拟合

# 获取回归结果
gamma_0 = model.params['const']  # γ0
gamma_1 = model.params['Beta_p']  # γ1
r_squared = model.rsquared  # R-squared
f_statistic = model.fvalue  # F 检验值
p_value = model.f_pvalue  # F 检验值对应的 P-value
gamma_0_t_value = model.tvalues['const']  # γ0 的 t 值
gamma_1_t_value = model.tvalues['Beta_p']  # γ1 的 t 值

# 将结果添加到 beta_df_2 中
beta_df_3 = beta_df_3.append({'Gamma_0': gamma_0, 
                              'Gamma_1': gamma_1,'Gamma_0_t_value': gamma_0_t_value, 'Gamma_1_t_value': gamma_1_t_value, 'R_squared': r_squared, 
                              'F_statistic': f_statistic, 'P_value': p_value}, ignore_index=True)


# In[255]:


print(beta_df_3)

