#!/usr/bin/env python
# coding: utf-8

# ### An attempt to balance classes in training dataset

# In[5]:


import sys
sys.path.append('../src/')
import filter_features as ff
import pandas as pd


# In[6]:


df = pd.read_parquet('../data/processed/train_features.parquet')
df.shape


# In[7]:


df['nan_counts'] = df.isnull().sum(axis=1)


# In[8]:


df[['country_destination', 'nan_counts']].head()


# In[13]:


df.country_destination.value_counts()


# In[11]:


df = df.sort_values(by=['country_destination', 'nan_counts'], ascending=[True, True])
df.reset_index(drop=True, inplace=True)


# In[12]:


df.head()


# In[20]:


tmp = df.groupby('country_destination', as_index=False).head(10000)
tmp.shape


# In[21]:


tmp.country_destination.value_counts()


# In[23]:


df.to_parquet('../data/processed/train_features_undersample.parquet')
df.shape


# In[ ]:




