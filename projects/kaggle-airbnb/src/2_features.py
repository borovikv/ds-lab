#!/usr/bin/env python
# coding: utf-8

# ## Notebook with feature engineering process

# ### Curated Features list (in addition to columns)
# 
# #### 1 Features based on Sessions actions:  
# 1. [x] Create features event_i with according to:  
#     * event_i means that it's action_info event of order i  
#     * take its first order in session, i.e. if events are show_nan_nan, show_view_p3 then values for show_view_p3 is 2  
#     * normalize by deviding by total number of events in user's session
# 2. [x] COUNT for each action_type
# 3. [x] MEAN, MAX and other descriptive statistics of secs_elapsed deltas
# 
# #### 2 Aggregated on Sessions:  
# 1. [x] COUNT DISTINCT of device_type
# 2. [ ] % time spent on each action type
# 3. [ ] count sessions per each device, MODE of Device type  
# 4. [ ] given that timestamp_first_active is the start of the session, analyze hour (0-23) of activity
# 
# #### 3 Transformed from users:
# 1. [x] Hour of first activity - users['hour_factive'] = users.timestamp_first_active.dt.hour
# 2. [x] date of week of account_created
# 
# **TODO**: use age_gender_bktd and countries data for features generation

# In[1]:


import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm
import numpy as np
from scipy import stats
from collections import Counter

pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
tqdm.pandas()
get_ipython().run_line_magic('load_ext', 'autotime')


# ### 0. Loading Data

# In[4]:


users = pd.read_parquet('../data/processed/users.parquet')
users.shape


# In[5]:


sessions = pd.read_parquet('../data/processed/sessions.parquet')
sessions.shape


# In[6]:


users.head()


# In[7]:


sessions.head()


# ### 2. Getting features based on Sessions

# In[8]:


sessions.secs_elapsed.fillna(-1, inplace=True)
sessions.sort_values(['user_id', 'secs_elapsed'], inplace=True)
sessions.reset_index(drop=True, inplace=True)
sessions.shape


# In[9]:


sessions.head(10)


# In[ ]:





# ### 2.1 Generating features based on action_info Events vs its Order in the session stream with Normalization
# 
# Sessions actions:  
# Create features event_i with according to:  
#     * event_i means that it's action_info event of order i  
#     * take its first order in session, i.e. if events are show_nan_nan, show_view_p3 then values for show_view_p3 is 2  
#     * normalize by deviding by total number of events in user's session  

# In[34]:


actions_info = list(sessions.action_info.unique())
len(actions_info)


# In[35]:


tmp = sessions[['user_id', 'action_info']].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[36]:


tmp['size'] = tmp.action_info.apply(lambda x: len(x))


# In[37]:


tmp.head()


# In[38]:


tmp.columns = ['user_id', 'action_info', 'seassion_length']


# In[39]:


def find_action_info_pos(ai, ais):
    try:
        return ais.index(ai) + 1
    except ValueError:
        return None


# In[40]:


for ai in tqdm(actions_info):
    tmp[f'ai_{ai}'] = tmp.action_info.apply(lambda x: find_action_info_pos(ai, x)) / tmp.size    


# In[41]:


tmp.head()


# ### 2.1.b Adding action_info counts features

# In[42]:


def get_action_info_count(ai, ais):
    return ais.count(ai)


# In[43]:


for ai in tqdm(actions_info):
    tmp[f'count_{ai}'] = tmp.action_info.apply(lambda x: get_action_info_count(ai, x))


# In[44]:


tmp.head()


# In[45]:


tmp.drop('action_info', axis=1, inplace=True)


# #### Checking counts of missing values per each column

# In[26]:


# not_missing = pd.DataFrame(tmp.notna().sum()).reset_index()
# not_missing.columns = ['col', 'counts']
# not_missing['ratio'] = not_missing['counts'].apply(lambda x: round(x / len(users), 4))
# not_missing.shape


# In[27]:


# not_missing.head()


# In[28]:


# threshold = 0.00005
# mask = not_missing.ratio > threshold
# mask.sum()


# #### Dropping all columns that are lower than the above threshold
# Decided not to do that in this iteration

# In[29]:


# keep_columns = not_missing[mask].col.tolist()
# len(keep_columns)


# In[30]:


# keep_columns[0], keep_columns[-1]


# ### 2.1.c Saving features

# In[46]:


# features1 = tmp[keep_columns].copy(deep=True)
features1 = tmp.copy(deep=True)
features1.shape


# ### 2.1.1 Count of each action_type normalized

# In[47]:


col = 'action_type'
col_values = list(sessions[col].unique())
len(col_values)


# In[48]:


tmp = sessions[['user_id', col]].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[49]:


tmp['size'] = tmp[col].apply(lambda x: len(x))


# In[50]:


tmp['counts'] = tmp[col].apply(lambda x: dict(Counter(x)))


# In[51]:


tmp.head()


# In[52]:


tmp = pd.concat([tmp, pd.json_normalize(tmp['counts'])], axis=1)


# In[53]:


tmp.drop(['action_type', 'counts'], axis=1, inplace=True)


# In[54]:


tmp.head()


# In[55]:


cols = list(tmp)[2:]
cols = [f'at_{e}' for e in cols]


# In[56]:


tmp.columns = ['user_id', 'size'] + cols


# In[57]:


for e in cols:
    tmp[e] = tmp[e] / tmp['size']


# In[58]:


tmp.head()


# In[59]:


tmp.drop(['size'], axis=1, inplace=True)


# In[60]:


tmp.fillna(0, inplace=True)


# In[61]:


tmp.head()


# In[62]:


features1a = tmp.copy(deep=True)
features1a.shape


# ### 2.2 Generating features based on seconds elapsed and deltas between info

# In[63]:


tmp = sessions[['user_id', 'secs_elapsed']].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[64]:


tmp.head()


# In[65]:


tmp.secs_elapsed = tmp.secs_elapsed.apply(lambda x: [0] + x[1:])


# In[66]:


tmp['deltas'] = tmp['secs_elapsed'].apply(lambda x: [int(j - i) for i, j in zip(x[:-1], x[1:])])


# In[67]:


tmp.head()


# In[68]:


def get_statistics(x):
    if not x:
        return None, None, None, None
    x = np.array(x)
    return x.mean(), x.std(), x.max(), np.median(x)


# In[69]:


def get_statistics_no_outliers(x):
    if not x:
        return None, None, None, None, None
    x = np.array(x)
    initial_size = len(x)
    x = [e for e in x if e <= x.mean() + x.std()]
    outliers_count = initial_size - len(x)
    x = np.array(x)
    return x.mean(), x.std(), x.max(), np.median(x), outliers_count


# In[70]:


get_statistics(tmp.iloc[0].deltas)


# In[71]:


get_statistics_no_outliers(tmp.iloc[0].deltas)


# In[72]:


tmp = pd.concat([tmp, tmp.deltas.progress_apply(lambda x: pd.Series(get_statistics(x)))], axis=1)
tmp.shape


# In[73]:


tmp.columns = ['user_id', 'secs_elapsed', 'deltas', 'deltas_mean', 'deltas_std', 'deltas_max', 'deltas_median']


# In[74]:


tmp.head()


# In[75]:


tmp = pd.concat([tmp, tmp.deltas.progress_apply(lambda x: pd.Series(get_statistics_no_outliers(x)))], axis=1)
tmp.shape


# In[76]:


tmp.columns = [
    'user_id', 'secs_elapsed', 'deltas', 'deltas_mean', 'deltas_std', 'deltas_max', 'deltas_median', 
    'deltas_no_mean', 'deltas_no_std', 'deltas_no_max', 'deltas_no_median', 'deltas_no_num_outliers'
]


# In[77]:


tmp.head()


# ### 2.2.1 Adding stats on seconds elapsed

# In[79]:


tmp = pd.concat([tmp, tmp.secs_elapsed.progress_apply(lambda x: pd.Series(get_statistics(x)))], axis=1)
tmp.shape


# In[80]:


tmp.columns = [
    'user_id', 'secs_elapsed', 'deltas', 'deltas_mean', 'deltas_std', 'deltas_max', 'deltas_median', 
    'deltas_no_mean', 'deltas_no_std', 'deltas_no_max', 'deltas_no_median', 'deltas_no_num_outliers',
    'secs_elapsed_mean', 'secs_elapsed_std', 'secs_elapsed_max', 'secs_elapsed_median',
]


# In[81]:


tmp.drop(['secs_elapsed', 'deltas'], axis=1, inplace=True)


# In[82]:


features2 = tmp.copy(deep=True)
features2.shape


# In[ ]:





# ### 2.3 Generating features based on device type info

# In[83]:


tmp = sessions[['user_id', 'device_type']].groupby('user_id', as_index=False).agg(set)
tmp.shape


# In[84]:


tmp['size'] = tmp.device_type.apply(lambda x: len(x))


# In[85]:


tmp.drop('device_type', axis=1, inplace=True)


# In[86]:


tmp.head()


# In[87]:


tmp.columns = ['user_id', 'device_count']


# In[88]:


tmp.head()


# In[89]:


features3 = tmp.copy(deep=True)
features3.shape


# ### 3.1 Features based on Users table

# In[90]:


users['dow_registered'] = users.date_account_created.dt.weekday
users['day_registered'] = users.date_account_created.dt.day
users['month_registered'] = users.date_account_created.dt.month
users['year_registered'] = users.date_account_created.dt.year


# In[92]:


users['hr_registered'] = users.timestamp_first_active.dt.hour


# In[96]:


users.age.max()


# In[98]:


users.head()


# In[99]:


mask = (users.age > 1000) & (users.age < 2000)
users.loc[mask, 'age'] = 2015 - users.loc[mask, 'age']
mask.sum()


# In[100]:


users.loc[(users['age'] > 105) | (users['age'] < 14), 'age'] = -1
users['age'].fillna(-1, inplace=True)


# In[101]:


bins = [-1, 20, 25, 30, 40, 50, 60, 75, 85, 105]
users['age_group'] = np.digitize(users['age'], bins, right=True)


# In[102]:


users.sample(5)


# In[103]:


users.shape


# ### 3.1.1. Dropping redundand columns

# In[104]:


users.drop(['date_account_created', 'timestamp_first_active'], axis=1, inplace=True)


# In[105]:


users.columns = ['user_id'] + list(users)[1:]


# In[106]:


users.head()


# In[107]:


users.shape


# #### 4. Assembling all features into one dataset

# In[109]:


df = users.merge(features1, on='user_id', how='left')
df.shape


# In[110]:


df = df.merge(features1a, on='user_id', how='left')
df.shape


# In[111]:


df = df.merge(features2, on='user_id', how='left')
df.shape


# In[112]:


df = df.merge(features3, on='user_id', how='left')
df.shape


# In[113]:


df.to_parquet('../data/processed/features.parquet')


# In[ ]:





# ### 4.1 Splitting into train and test features

# In[2]:


df = pd.read_parquet('../data/processed/features.parquet')
df.shape


# In[3]:


train_users = pd.read_parquet('../data/processed/train_users.parquet')
train_users.shape


# In[4]:


train_features = df[df.user_id.isin(train_users.id)]
train_features.shape


# In[5]:


train_features.to_parquet('../data/processed/train_features.parquet')


# In[ ]:





# In[6]:


test_users = pd.read_parquet('../data/processed/test_users.parquet')
test_users.shape


# In[7]:


test_features = df[df.user_id.isin(test_users.id)]
test_features.shape


# In[8]:


test_features.to_parquet('../data/processed/test_features.parquet')


# In[ ]:




