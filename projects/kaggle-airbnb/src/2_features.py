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
from scipy import stats
import itertools

pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
tqdm.pandas()
# %load_ext autotime


# ### 0. Loading Data

# In[2]:


users = pd.read_parquet('../data/processed/users.parquet')
users.shape


# In[3]:


sessions = pd.read_parquet('../data/processed/sessions.parquet')
sessions.shape


# In[4]:


users.head()


# In[5]:


sessions.head()


# ### 2. Getting features based on Sessions

# In[6]:


sessions.secs_elapsed.fillna(-1, inplace=True)
sessions.sort_values(['user_id', 'secs_elapsed'], inplace=True)
sessions.reset_index(drop=True, inplace=True)
sessions.shape


# In[7]:


sessions.head(10)


# In[ ]:





# ### 2.1 Generating features based on action_info Events vs its Order in the session stream with Normalization
# 
# Sessions actions:  
# Create features event_i with according to:  
#     * event_i means that it's action_info event of order i  
#     * take its first order in session, i.e. if events are show_nan_nan, show_view_p3 then values for show_view_p3 is 2  
#     * normalize by deviding by total number of events in user's session  

# ### Small change to previous approaches, taking only those action_info, which are present in both train and test datasets, without dropping the rest of action_info

# In[8]:


# actions_info = list(sessions.action_info.unique())
# len(actions_info)
actions_info = set(sessions[sessions.user_id.isin(users[users.train_flag == 1].id)].action_info.unique())
actions_info = actions_info.intersection(set(sessions[sessions.user_id.isin(users[users.train_flag == 0].id)].action_info.unique()))
actions_info = list(actions_info)
len(actions_info)


# In[9]:


# actions_info = list(sessions.action_info.unique())
# len(actions_info)
action_details = set(sessions[sessions.user_id.isin(users[users.train_flag == 1].id)].action_detail.unique())
action_details = action_details.intersection(set(sessions[sessions.user_id.isin(users[users.train_flag == 0].id)].action_detail.unique()))
action_details = list(action_details)
len(action_details)


# In[10]:


sessions.action_detail.nunique()


# In[11]:


tmp = sessions[['user_id', 'action_info']].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[12]:


tmp['size'] = tmp.action_info.apply(lambda x: len(x))


# In[13]:


tmp.head()


# In[14]:


tmp.columns = ['user_id', 'action_info', 'seassion_length']


# In[15]:


def find_action_info_pos(ai, ais):
    try:
        return ais.index(ai) + 1
    except ValueError:
        return None


# In[16]:


res = []
for ai in tqdm(actions_info):
    res.append(pd.DataFrame({f'ai_{ai}': tmp.action_info.apply(lambda x: find_action_info_pos(ai, x))}))
tmp = pd.concat([tmp] + res, axis=1)
tmp.shape


# In[17]:


lst = [1, 2, 1, 2, 3, 4, 1, 3, 4, 2]


# In[18]:


lst.index(2, 1)


# ### 2.1.0 Adding sequencialization of action_info events features

# In[19]:


top20_ai = pd.read_csv('../data/processed/top20_ai_features.csv')
top20_ai.shape


# In[20]:


top20_ai.head(2)


# In[21]:


top20_ai = top20_ai.feature_name.tolist()
len(top20_ai)


# In[22]:


top20_ai = list(itertools.permutations(top20_ai,2))
len(top20_ai), top20_ai[0]


# In[23]:


def count_times_a_before_b(a, b, lst):
    pos_a = lst.index(a) if a in lst else None
    pos_b = lst.index(b, pos_a) if b in lst and pos_a is not None else None
    count = 0
    while pos_a is not None and pos_b is not None:
        count += 1
        pos_a = lst.index(a, pos_a + 1) if a in lst[pos_a + 1:] else None
        pos_b = lst.index(b, pos_a) if b in lst[pos_a:] and pos_a is not None else None
    return count


# In[24]:


res = []
for (a, b) in tqdm(top20_ai):
    res.append(pd.DataFrame({f'ai_{a}_before_{b}': tmp.action_info.apply(lambda x: count_times_a_before_b(a, b, x))}))
tmp = pd.concat([tmp] + res, axis=1)
tmp.shape


# In[25]:


tmp.head()


# ### 2.1.b Adding action_info counts features

# In[26]:


def get_action_info_count(ai, ais):
    return ais.count(ai)


# In[27]:


res = []
for ai in tqdm(actions_info):
    res.append(pd.DataFrame({f'count_{ai}': tmp.action_info.apply(lambda x: get_action_info_count(ai, x))}))
tmp = pd.concat([tmp] + res, axis=1)
tmp.shape


# In[28]:


tmp.head()


# In[29]:


tmp.drop('action_info', axis=1, inplace=True)


# #### Checking counts of missing values per each column

# In[30]:


# not_missing = pd.DataFrame(tmp.notna().sum()).reset_index()
# not_missing.columns = ['col', 'counts']
# not_missing['ratio'] = not_missing['counts'].apply(lambda x: round(x / len(users), 4))
# not_missing.shape


# In[31]:


# not_missing.head()


# In[32]:


# threshold = 0.00005
# mask = not_missing.ratio > threshold
# mask.sum()


# #### Dropping all columns that are lower than the above threshold
# Decided not to do that in this iteration

# In[33]:


# keep_columns = not_missing[mask].col.tolist()
# len(keep_columns)


# In[34]:


# keep_columns[0], keep_columns[-1]


# ### 2.1.c Saving features

# In[35]:


# features1 = tmp[keep_columns].copy(deep=True)
features1 = tmp.copy(deep=True)
features1.shape


# ### 2.1.1 Count of each action_type normalized

# In[36]:


col = 'action_type'
col_values = list(sessions[col].unique())
len(col_values)


# In[37]:


tmp = sessions[['user_id', col]].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[38]:


tmp['size'] = tmp[col].apply(lambda x: len(x))


# In[39]:


tmp['counts'] = tmp[col].apply(lambda x: dict(Counter(x)))


# In[40]:


tmp.head()


# In[41]:


tmp = pd.concat([tmp, pd.json_normalize(tmp['counts'])], axis=1)


# In[42]:


tmp.drop(['action_type', 'counts'], axis=1, inplace=True)


# In[43]:


tmp.head()


# In[44]:


cols = list(tmp)[2:]
cols = [f'at_{e}_counts' for e in cols]


# In[45]:


tmp.columns = ['user_id', 'size'] + cols


# In[46]:


# for e in cols:
#     tmp[e] = tmp[e] / tmp['size']


# In[47]:


tmp.head()


# In[48]:


# tmp.drop(['size'], axis=1, inplace=True)


# In[49]:


tmp.fillna(0, inplace=True)


# In[50]:


tmp.head()


# In[51]:


features1a = tmp.copy(deep=True)
features1a.shape


# ### 2.2 Generating features based on seconds elapsed and deltas between info

# In[52]:


tmp = sessions[['user_id', 'secs_elapsed']].groupby('user_id', as_index=False).agg(list)
tmp.shape


# In[53]:


tmp.head()


# In[54]:


tmp.secs_elapsed = tmp.secs_elapsed.apply(lambda x: [0] + x[1:])


# In[55]:


tmp['deltas'] = tmp['secs_elapsed'].apply(lambda x: [int(j - i) for i, j in zip(x[:-1], x[1:])])


# In[56]:


tmp.head()


# In[57]:


tmp.deltas = tmp.deltas.apply(lambda x: np.array(x) if x else None)


# In[58]:


tmp.secs_elapsed = tmp.secs_elapsed.apply(lambda x: np.array(x) if x else None)


# In[59]:


def get_statistics(x):
    if x is None:
        return None, None, None, None, None, None, None
    m = stats.mode(x)
    return x.mean(), x.std(), x.max(), x.min(), np.median(x), m.mode[0], m.count[0]


# In[60]:


def get_statistics_no_outliers(x):
    if x is None:
        return None, None, None, None, None
    x = np.array(x)
    initial_size = len(x)
    x = x[x  <= x.mean() + x.std()]
    outliers_count = initial_size - len(x)
    m = stats.mode(x)
    return x.mean(), x.std(), x.max(), np.median(x), outliers_count


# In[61]:


get_statistics(tmp.iloc[0].deltas)


# In[62]:


get_statistics_no_outliers(tmp.iloc[0].deltas)


# In[63]:


tmp = pd.concat([tmp, tmp.deltas.progress_apply(lambda x: pd.Series(get_statistics(x)))], axis=1)
tmp.shape


# In[64]:


column_names = ['user_id', 'secs_elapsed', 'deltas', 'deltas_mean', 'deltas_std', 'deltas_max', 
    'deltas_min', 'deltas_median', 'deltas_mode', 'deltas_mode_count']


# In[65]:


tmp.columns = column_names


# In[66]:


tmp.head()


# In[67]:


tmp = pd.concat([tmp, tmp.deltas.progress_apply(lambda x: pd.Series(get_statistics_no_outliers(x)))], axis=1)
tmp.shape


# In[68]:


tmp.columns = column_names + ['deltas_no_mean', 'deltas_no_std', 'deltas_no_max', 'deltas_no_median', 'deltas_no_num_outliers']


# In[69]:


tmp.head(2)


# ### 2.2.1 Adding stats on seconds elapsed

# In[70]:


tmp = pd.concat([tmp, tmp.secs_elapsed.progress_apply(lambda x: pd.Series(get_statistics(x)))], axis=1)
tmp.shape


# In[71]:


tmp.columns = column_names + ['deltas_no_mean', 'deltas_no_std', 'deltas_no_max', 'deltas_no_median', 'deltas_no_num_outliers',
    'secs_elapsed_mean', 'secs_elapsed_std', 'secs_elapsed_max', 'secs_elapsed_min', 'secs_elapsed_median',
    'secs_elapsed_mode', 'secs_elapsed_mode_count'
]


# In[72]:


tmp.drop(['secs_elapsed', 'deltas'], axis=1, inplace=True)


# In[73]:


features2 = tmp.copy(deep=True)
features2.shape


# In[ ]:





# ### 2.3 Generating features based on device type info

# In[74]:


tmp = sessions[['user_id', 'device_type']].groupby('user_id', as_index=False).agg(set)
tmp.shape


# In[75]:


tmp['size'] = tmp.device_type.apply(lambda x: len(x))


# In[76]:


tmp.drop('device_type', axis=1, inplace=True)


# In[77]:


tmp.head()


# In[78]:


tmp.columns = ['user_id', 'device_count']


# In[79]:


tmp.head()


# In[80]:


features3 = tmp.copy(deep=True)
features3.shape


# ### 3.1 Features based on Users table

# In[81]:


users[users.train_flag == 1].date_account_created.dt.year.value_counts()


# In[82]:


users[users.train_flag == 0].date_account_created.dt.year.value_counts()


# In[83]:


users[users.train_flag == 1].date_account_created.dt.month.value_counts()


# In[84]:


users[users.train_flag == 0].date_account_created.dt.month.value_counts()


# In[85]:


users['dow_registered'] = users.date_account_created.dt.weekday
users['day_registered'] = users.date_account_created.dt.day
users['month_registered'] = users.date_account_created.dt.month
users['year_registered'] = users.date_account_created.dt.year


# In[86]:


users['hr_registered'] = users.timestamp_first_active.dt.hour


# In[87]:


users.age.max()


# In[88]:


users.head()


# In[89]:


mask = (users.age > 1000) & (users.age < 2000)
users.loc[mask, 'age'] = 2015 - users.loc[mask, 'age']
mask.sum()


# In[90]:


users.loc[(users['age'] > 105) | (users['age'] < 14), 'age'] = -1
users['age'].fillna(-1, inplace=True)


# In[91]:


bins = [-1, 20, 25, 30, 40, 50, 60, 75, 85, 105]
users['age_group'] = np.digitize(users['age'], bins, right=True)


# In[92]:


users.sample(5)


# In[93]:


users.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 3.1.1. Dropping redundand columns

# In[94]:


users.drop(['date_account_created', 'timestamp_first_active'], axis=1, inplace=True)


# In[95]:


users.columns = ['user_id'] + list(users)[1:]


# In[96]:


users.head()


# In[97]:


users.shape


# #### 4. Assembling all features into one dataset

# In[98]:


df = users.merge(features1, on='user_id', how='left')
df.shape


# In[99]:


df = df.merge(features1a, on='user_id', how='left')
df.shape


# In[100]:


df = df.merge(features2, on='user_id', how='left')
df.shape


# In[101]:


df = df.merge(features3, on='user_id', how='left')
df.shape


# In[102]:


df.to_parquet('../data/processed/features.parquet')


# In[ ]:





# ### 4.1 Splitting into train and test features

# In[ ]:


train_features = df[df.train_flag == 1]
train_features.shape


# In[ ]:


train_features.to_parquet('../data/processed/train_features.parquet')


# In[ ]:





# In[ ]:


test_features = df[df.train_flag == 0]
test_features.shape


# In[ ]:


test_features.to_parquet('../data/processed/test_features.parquet')


# In[ ]:




