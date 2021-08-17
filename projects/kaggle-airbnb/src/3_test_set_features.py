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

# In[2]:


users = pd.read_parquet('../data/processed/test_users.parquet')
users.shape


# In[3]:


sessions = pd.read_parquet('../data/processed/sessions_test.parquet')
sessions.shape


# In[4]:


users.date_account_created = users.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
users.timestamp_first_active = users.timestamp_first_active.apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))


# In[5]:


users.head()


# In[6]:


sessions.head()


# ### 2. Getting features based on Sessions

# In[7]:


sessions.secs_elapsed.fillna(-1, inplace=True)
sessions.sort_values(['user_id', 'secs_elapsed'], inplace=True)
sessions.reset_index(drop=True, inplace=True)
sessions.shape


# In[8]:


sessions.head(10)


# In[ ]:





# ### 2.1 Generating features based on action_info Events vs its Order in the session stream with Normalization
# 
# Sessions actions:  
# Create features event_i with according to:  
#     * event_i means that it's action_info event of order i  
#     * take its first order in session, i.e. if events are show_nan_nan, show_view_p3 then values for show_view_p3 is 2  
#     * normalize by deviding by total number of events in user's session  

# In[9]:


actions_info = list(sessions.action_info.unique())
len(actions_info)


# In[10]:


tmp = sessions[['user_id', 'action_info']].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[11]:


tmp['size'] = tmp.action_info.apply(lambda x: len(x))


# In[12]:


tmp.head()


# In[13]:


tmp.columns = ['user_id', 'action_info', 'seassion_length']


# In[14]:


def find_action_info_pos(ai, ais):
    try:
        return ais.index(ai) + 1
    except ValueError:
        return None


# In[15]:


for ai in tqdm(actions_info):
    tmp[f'ai_{ai}'] = tmp.action_info.apply(lambda x: find_action_info_pos(ai, x)) / tmp.size    


# In[16]:


tmp.head()


# In[17]:


tmp.drop('action_info', axis=1, inplace=True)


# #### Checking counts of missing values per each column

# In[18]:


not_missing = pd.DataFrame(tmp.notna().sum()).reset_index()
not_missing.columns = ['col', 'counts']
not_missing['ratio'] = not_missing['counts'].apply(lambda x: round(x / len(users), 4))
not_missing.shape


# In[19]:


not_missing.head()


# In[20]:


threshold = 0.00005
mask = not_missing.ratio > threshold
mask.sum()


# #### Dropping all columns that are lower than the above threshold

# In[21]:


keep_columns = not_missing[mask].col.tolist()
len(keep_columns)


# In[22]:


keep_columns[0], keep_columns[-1]


# In[23]:


features1 = tmp[keep_columns].copy(deep=True)
features1.shape


# ### 2.1.1 Count of each action_type normalized

# In[24]:


col = 'action_type'
col_values = list(sessions[col].unique())
len(col_values)


# In[25]:


tmp = sessions[['user_id', col]].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[26]:


tmp['size'] = tmp[col].apply(lambda x: len(x))


# In[27]:


tmp['counts'] = tmp[col].apply(lambda x: dict(Counter(x)))


# In[28]:


tmp.head()


# In[29]:


tmp = pd.concat([tmp, pd.json_normalize(tmp['counts'])], axis=1)


# In[30]:


tmp.drop(['action_type', 'counts'], axis=1, inplace=True)


# In[31]:


tmp.head()


# In[32]:


cols = list(tmp)[2:]
cols = [f'at_{e}' for e in cols]


# In[33]:


tmp.columns = ['user_id', 'size'] + cols


# In[34]:


for e in cols:
    tmp[e] = tmp[e] / tmp['size']


# In[35]:


tmp.head()


# In[36]:


tmp.drop(['size'], axis=1, inplace=True)


# In[37]:


tmp.fillna(0, inplace=True)


# In[38]:


tmp.head()


# In[39]:


features1a = tmp.copy(deep=True)
features1a.shape


# ### 2.2 Generating features based on seconds elapsed info

# In[40]:


tmp = sessions[['user_id', 'secs_elapsed']].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[41]:


tmp.head()


# In[42]:


tmp.secs_elapsed = tmp.secs_elapsed.apply(lambda x: [0] + x[1:])


# In[43]:


tmp['deltas'] = tmp['secs_elapsed'].apply(lambda x: [int(j - i) for i, j in zip(x[:-1], x[1:])])


# In[44]:


tmp.head()


# In[45]:


def get_statistics(x):
    if not x:
        return None, None, None, None
    x = np.array(x)
    return x.mean(), x.std(), x.max(), np.median(x)


# In[46]:


def get_statistics_no_outliers(x):
    if not x:
        return None, None, None, None, None
    x = np.array(x)
    initial_size = len(x)
    x = [e for e in x if e <= x.mean() + x.std()]
    outliers_count = initial_size - len(x)
    x = np.array(x)
    return x.mean(), x.std(), x.max(), np.median(x), outliers_count


# In[47]:


get_statistics(tmp.iloc[0].deltas)


# In[48]:


get_statistics_no_outliers(tmp.iloc[0].deltas)


# In[49]:


tmp = pd.concat([tmp, tmp.deltas.progress_apply(lambda x: pd.Series(get_statistics(x)))], axis=1)
tmp.shape


# In[50]:


tmp.columns = ['user_id', 'secs_elapsed', 'deltas', 'deltas_mean', 'deltas_std', 'deltas_max', 'deltas_median']


# In[51]:


tmp.head()


# In[52]:


tmp = pd.concat([tmp, tmp.deltas.progress_apply(lambda x: pd.Series(get_statistics_no_outliers(x)))], axis=1)
tmp.shape


# In[53]:


tmp.columns = [
    'user_id', 'secs_elapsed', 'deltas', 'deltas_mean', 'deltas_std', 'deltas_max', 'deltas_median', 
    'deltas_no_mean', 'deltas_no_std', 'deltas_no_max', 'deltas_no_median', 'deltas_no_num_outliers'
]


# In[54]:


tmp.head()


# In[55]:


tmp.drop(['secs_elapsed', 'deltas'], axis=1, inplace=True)


# In[56]:


features2 = tmp.copy(deep=True)
features2.shape


# ### 2.3 Generating features based on device type info

# In[57]:


tmp = sessions[['user_id', 'device_type']].groupby('user_id', as_index=False).agg(set)
tmp.shape


# In[58]:


tmp['size'] = tmp.device_type.apply(lambda x: len(x))


# In[59]:


tmp.drop('device_type', axis=1, inplace=True)


# In[60]:


tmp.head()


# In[61]:


tmp.columns = ['user_id', 'device_count']


# In[62]:


tmp.head()


# In[63]:


features3 = tmp.copy(deep=True)
features3.shape


# ### 3.1 Features based on Users table

# In[64]:


users['dow_registered'] = users.date_account_created.dt.weekday


# In[65]:


users['hr_registered'] = users.timestamp_first_active.dt.hour


# In[66]:


users.sample(5)


# ### 3.1.1. Dropping redundand columns

# In[67]:


users.drop(['date_account_created', 'timestamp_first_active'], axis=1, inplace=True)


# In[68]:


users.columns = ['user_id'] + list(users)[1:]


# In[69]:


users.head()


# In[70]:


users.shape


# #### 4. Assembling all features into one dataset

# In[71]:


df = users.merge(features1, on='user_id', how='inner')
df.shape


# In[72]:


df = df.merge(features1a, on='user_id', how='inner')
df.shape


# In[73]:


df = df.merge(features2, on='user_id', how='inner')
df.shape


# In[74]:


df = df.merge(features3, on='user_id', how='inner')
df.shape


# In[75]:


df.to_parquet('../data/processed/features_test.parquet')


# In[ ]:





# In[ ]:




