#!/usr/bin/env python
# coding: utf-8

# ### Pre-processing steps based on EDA

# In[1]:


import pandas as pd
from datetime import datetime


# ### 0 Loading data

# In[2]:


df = pd.read_csv('../data/original/train_users_2.csv')
df.shape


# In[3]:


df.date_account_created = df.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df.timestamp_first_active = df.timestamp_first_active.apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))
df.date_first_booking = df.date_first_booking.apply(lambda x: datetime.strptime(x, '%Y-%m-%d') if pd.notna(x) else None)


# In[4]:


df.head()


# In[5]:


test = pd.read_csv('../data/original/test_users.csv')
test.shape


# In[6]:


test.head()


# In[7]:


ss = pd.read_csv('../data/original/sessions.csv')
ss.shape


# In[8]:


ss.head()


# ### 1 Checking whether sessions are present for all users. 
# ### It appears that around sessions data is available for only 34% of users

# In[9]:


ids = set(df.id.tolist()).intersection(set(ss.user_id.tolist()))
ids = list(ids)
len(ids)


# In[10]:


len(ids) * 100 / df.id.nunique()


# ### 2 Checking whether sessions are present for all **TEST** users. 
# ### It appears that sessions data is available for almost all test users

# In[11]:


ids = set(test.id.tolist()).intersection(set(ss.user_id.tolist()))
ids = list(ids)
len(ids)


# In[12]:


len(ids) * 100 / test.id.nunique()


# ### 3 Dropping redundand data

# ### 3.1 We need to drop users with no sesssions

# In[13]:


ids = set(df.id.tolist()).intersection(set(ss.user_id.tolist()))
df = df[df.id.isin(ids)]
df.reset_index(drop=True, inplace=True)
df.shape


# ### 3.2 We need to drop date_first_booking, as it is null for test set

# In[14]:


test.date_first_booking.isna().sum(), len(test)


# In[15]:


df.drop('date_first_booking', inplace=True, axis=1)
test.drop('date_first_booking', inplace=True, axis=1)
df.shape, test.shape


# ### 3.3 We need to drop sessions without any user_ids

# In[16]:


ss.user_id.isna().sum()


# In[17]:


ss = ss.loc[ss.user_id.notna()]
ss.reset_index(drop=True, inplace=True)
ss.shape


# ### 3.4 Checking whether all action, action_types etc are present in both datasets

# ### 3.4.1 Dropping all actions that are not present in both datasets

# In[36]:


ss['action_info'] = ss[['action', 'action_type', 'action_detail']].apply(lambda x: f'{x[0]}_{x[1]}_{x[2]}', axis=1)


# In[47]:


ss.sample(20)


# In[48]:


ss_train = ss[ss.user_id.isin(df.id.tolist())]
ss_train.reset_index(drop=True, inplace=True)
ss_train.shape


# In[49]:


ss_test = ss[ss.user_id.isin(test.id.tolist())]
ss_test.reset_index(drop=True, inplace=True)
ss_test.shape


# In[50]:


ss_train.nunique()


# In[51]:


ss_test.nunique()


# In[52]:


action_info_set = set(ss_train.action_info.unique()).intersection(set(ss_test.action_info.unique()))
len(actions_set)


# In[ ]:





# In[53]:


ss_train = ss_train[ss_train.action_info.isin(action_info_set)]
ss_train.reset_index(drop=True, inplace=True)
ss_train.shape


# In[54]:


ss_test = ss_test[ss_test.action_info.isin(action_info_set)]
ss_test.reset_index(drop=True, inplace=True)
ss_test.shape


# In[ ]:





# In[ ]:





# In[ ]:





# ### 4. Saving data

# In[60]:


df.to_parquet('../data/processed/train_users_2.parquet')
test.to_parquet('../data/processed/test_users.parquet')


# In[61]:


ss_train.to_parquet('../data/processed/sessions_train.parquet')
ss_test.to_parquet('../data/processed/sessions_test.parquet')


# In[ ]:




