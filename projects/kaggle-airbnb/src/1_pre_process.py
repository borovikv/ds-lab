#!/usr/bin/env python
# coding: utf-8

# ### Pre-processing steps based on EDA

# In[1]:


import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm

tqdm.pandas()


# ### 0 Loading data

# In[2]:


train_users = pd.read_csv('../data/original/train_users_2.csv')
test_users = pd.read_csv('../data/original/test_users.csv')
train_users.shape, test_users.shape


# In[5]:


users = pd.concat([train_users, test_users], axis=0)
users.shape


# In[6]:


users['train_flag'] = 1
users.loc[users.id.isin(test_users.id), 'train_flag'] = 0
users.train_flag.value_counts()


# In[7]:


users.date_account_created = users.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
users.timestamp_first_active = users.timestamp_first_active.apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))
users.date_first_booking = users.date_first_booking.apply(lambda x: datetime.strptime(x, '%Y-%m-%d') if pd.notna(x) else None)


# In[8]:


users.head()


# In[9]:


sessions = pd.read_csv('../data/original/sessions.csv')
sessions.shape


# In[10]:


sessions.head()


# ### 1 Checking whether sessions are present for all users. 
# ### It appears that around sessions data is available for only 34% of train users

# In[11]:


ids = set(train_users.id.tolist()).intersection(set(sessions.user_id.tolist()))
len(ids), len(ids) * 100 / train_users.id.nunique()


# ### 2 Checking whether sessions are present for all **TEST** users. 
# ### It appears that sessions data is available for almost all test users

# In[12]:


ids = set(test_users.id.tolist()).intersection(set(sessions.user_id.tolist()))
len(ids), len(ids) * 100 / test_users.id.nunique()


# ### 3 Dropping redundand data

# ### 3.1 We need to drop users with no sesssions

# In[13]:


users.shape


# In[14]:


test_users[~test_users.id.isin(sessions.user_id)].id.nunique()


# ### 3.1.1 Decided to leave users without sessions, as we would have dropped 428 users

# In[15]:


# users = users[users.id.isin(sessions.user_id)]
# users.reset_index(drop=True, inplace=True)
# users.shape


# ### 3.2 We need to drop date_first_booking, as it is null for test set

# In[16]:


users.drop('date_first_booking', inplace=True, axis=1)
users.shape


# ### 3.2.1 Need to update train_users and test_users dataframes after cleaning
# No longer needed, as we did not drop any users

# In[17]:


# train_users.shape, test_users.shape


# In[15]:


# train_users = train_users[train_users.id.isin(sessions.user_id)]
# train_users.reset_index(drop=True, inplace=True)
# train_users.shape


# In[ ]:


# test_users = test_users[test_users.id.isin(sessions.user_id)]
# test_users.reset_index(drop=True, inplace=True)
# test_users.shape


# ### 3.3 We need to drop sessions without any user_ids

# In[20]:


sessions.user_id.isna().sum(), (~sessions.user_id.isin(users.id)).sum()


# In[21]:


sessions = sessions[sessions.user_id.notna()]
sessions.reset_index(drop=True, inplace=True)
sessions.shape


# ### 3.4 Checking whether all action, action_types etc are present in both datasets

# ### 3.4.1 Concatenating action, action_type, action_detail, and splitting into train and test sessions set

# In[23]:


sessions['action_info'] = sessions['action'].astype(str) + '_' + sessions['action_type'].astype(str) + '_' + sessions['action_detail'].astype(str)


# In[24]:


sessions.head()


# In[25]:


sessions.nunique()


# In[ ]:


sessions_test = sessions[sessions.user_id.isin(test_users.id.tolist())]
sessions_test.reset_index(drop=True, inplace=True)
sessions_test.shape


# In[ ]:


sessions_train.nunique()


# In[21]:


sessions_test.nunique()


# ### 3.5 Checking action, action_type, action_detail, device_type, action_info unique values for test and train users

# In[26]:


cols = ['action', 'action_type', 'action_detail', 'device_type', 'action_info']


# In[27]:


for col in cols:
    train_set = set(sessions_train[col].unique())
    test_set = set(sessions_test[col].unique())
    if train_set != test_set:
        print(f'Discrepancy found for: {col}')
        print(f'train size: {len(train_set)}, test size: {len(test_set)}')
        print(f'Present in train but missing in test:\n{train_set - test_set}')
        print(f'Present in test but missing in train:\n{test_set - train_set}')


# ### 3.5.1 Due to the fact that discrepancies where found, we are taking only those sessions that have action_info in both sets
# Not doing that any longer, as we might drop 4 users sessions because of this, resulting in total decrease of 432 users (428 + 4)

# In[28]:


# action_info_set = set(sessions_train.action_info.unique()).intersection(set(sessions_test.action_info.unique()))
# len(action_info_set)


# In[29]:


# unique_user_ids = sessions.user_id.nunique()
# unique_user_ids


# In[30]:


# sessions = sessions[sessions.action_info.isin(action_info_set)]
# sessions.reset_index(drop=True, inplace=True)
# sessions.shape


# ### 3.5.3 Checking if the submission files are missing 4 test_users

# In[38]:


submission = pd.read_csv('../data/results/submission1.csv')
submission.shape


# In[41]:


submission.head()


# In[39]:


original_test_users = pd.read_csv('../data/original/test_users.csv')
original_test_users.shape


# In[42]:


original_test_users.head()


# ### It appears that there are 432 test users we did not submit any results
# Conclusion: not to drop non-intersecting action_info and users without any sessions

# In[43]:


# len(set(original_test_users.id.unique()) - set(submission.id.unique()))


# In[44]:


# len(set(submission.id.unique()) - set(original_test_users.id.unique()))


# In[53]:


# ss_train = ss_train[ss_train.action_info.isin(action_info_set)]
# ss_train.reset_index(drop=True, inplace=True)
# ss_train.shape


# In[54]:


# ss_test = ss_test[ss_test.action_info.isin(action_info_set)]
# ss_test.reset_index(drop=True, inplace=True)
# ss_test.shape


# In[ ]:





# In[ ]:





# In[ ]:





# ### 4. Saving data

# In[26]:


sessions.to_parquet('../data/processed/sessions.parquet')


# In[27]:


users.to_parquet('../data/processed/users.parquet')


# In[ ]:





# In[ ]:





# In[ ]:




