#!/usr/bin/env python
# coding: utf-8

# ## Notebook to generate Pandas Profiling reports, for better Exploratory Data Analysis
# 
# #### To install pandas profiling package, run the following command in Jupyter cell:  
# ! pip install pandas-profiling[notebook]

# In[1]:


import pandas as pd
from pandas_profiling import ProfileReport
from datetime import datetime

pd.options.display.float_format = "{:.2f}".format
get_ipython().run_line_magic('load_ext', 'autotime')


# ### 1. Creating profiling report for age_gender_bkts file

# In[5]:


df = pd.read_csv('../data/original/age_gender_bkts.csv')
df.shape


# In[6]:


df.head()


# In[13]:


df[df.country_destination == 'AU'].population_in_thousands.sum()


# In[15]:


df[(df.country_destination == 'AU') & (df.gender == 'female')]


# In[ ]:





# In[8]:


profile = ProfileReport(df, title="Age Gender Buckets Profiling Report", explorative=True)
profile.to_file('../reports/age_gender_bkts.html')


# ### 2. Viewing countries file

# In[11]:


df = pd.read_csv('../data/original/countries.csv')
df.shape


# In[13]:


df


# ### 3. Creating profiling report for entire sessions file failed, creating for its subsample

# In[32]:


df = pd.read_csv('../data/original/sessions.csv')
df.shape


# In[33]:


df.head()


# In[34]:


df.describe()


# In[35]:


profile = ProfileReport(df.sample(100_000).reset_index(drop=True), title="Users Profiling Report", explorative=True)
profile.to_file('../reports/sessions.html')


# ### 4. Creating profiling report for train_users_2 file

# In[16]:


df = pd.read_csv('../data/original/train_users_2.csv')
df.shape


# In[7]:


df.head()


# In[16]:


df.date_account_created = df.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[25]:


df.timestamp_first_active = df.timestamp_first_active.apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))


# In[28]:


df.date_first_booking = df.date_first_booking.apply(lambda x: datetime.strptime(x, '%Y-%m-%d') if pd.notna(x) else None)


# In[29]:


df.head()


# In[30]:


profile = ProfileReport(df, title="Users Profiling Report", explorative=True)
profile.to_file('../reports/train_users_2.html')


# In[ ]:





# In[18]:


df[df.age > 1000].country_destination.value_counts()


# In[37]:


df[df.date_first_booking.isna()].country_destination.value_counts()


# In[ ]:





# In[ ]:





# \[ DCG_k=\sum_{i=1}^k\frac{2^{rel_i}-1}{\log_2{\lefti+1\righti+1\right}}, \]
# \[ nDCG_k=\frac{DCG_k}{IDCG_k}, \]

# In[ ]:





# ### 5. Creating profiling report for train_users_2 file

# In[19]:


df = pd.read_csv('../data/original/test_users.csv')
df.shape


# In[20]:


df.head()


# In[21]:


df.date_account_created = df.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[22]:


df.timestamp_first_active = df.timestamp_first_active.apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M%S'))


# In[23]:


df.date_first_booking = df.date_first_booking.apply(lambda x: datetime.strptime(x, '%Y-%m-%d') if pd.notna(x) else None)


# In[24]:


df.head()


# In[28]:


profile = ProfileReport(df, title="Test Users Profiling Report", explorative=True)
profile.to_file('../reports/test_users.html')

