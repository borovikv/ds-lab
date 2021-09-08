#!/usr/bin/env python
# coding: utf-8

# ### Filtering features from Low Variance, High Correlation and Empty Values Columns

# In[1]:


import sys
sys.path.append('../src/')
import filter_features as ff
import pandas as pd


# In[7]:


df = pd.read_parquet('../data/processed/train_features.parquet')
df.shape


# In[4]:


empty_columns = ff.get_empty_columns(df, threshold=0.999)
len(empty_columns)


# ### 1. Getting and loading Feature Importance file

# In[3]:


model_path = '/home/jovyan/projects/kaggle-airbnb/models/model_2021_09_06_14_22_11.cbm'
x_pickle_path = '/home/jovyan/projects/kaggle-airbnb/data/processed/x_test.pickle'
y_pickle_path = '/home/jovyan/projects/kaggle-airbnb/data/processed/y_test.pickle'
cat_features = [
    'gender',
    'signup_method',
    'signup_flow',
    'language',
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'signup_app',
    'first_device_type',
    'first_browser',
    'dow_registered',
    'hr_registered',
    'age_group',
    'dow_registered',
    'day_registered',
    'month_registered',
    'year_registered',
]
save_to = '/home/jovyan/projects/kaggle-airbnb/data/processed/features_importance.csv'


# In[4]:


ff.generate_feature_importance_file(model_path, x_pickle_path, y_pickle_path, cat_features, save_to)


# In[5]:


fi = ff.get_normalized_feature_weights(save_to)
fi = [{'feature_name': k, 'weight': v} for k, v in fi.items()]
fi = pd.DataFrame(fi)
fi['w_sum'] = fi.weight.cumsum()
fi.shape


# In[8]:


fi.head(10)


# In[7]:


fi[fi.feature_name == 'gender']


# ### 1.1 If we remove empty_cols columns we would lose approximately 1.4% of feature importance

# In[9]:


fi[fi.feature_name.isin(empty_columns)].weight.sum()


# In[ ]:





# In[ ]:





# ### 2. Generating Low variance columns

# In[10]:


low_variance_columns = ff.get_low_variance_columns(df, threshold=0.0001)
len(low_variance_columns)


# In[11]:


fi[fi.feature_name.isin(low_variance_columns)].weight.sum()


# In[12]:


remove_columns = set(empty_columns).union(set(low_variance_columns))
len(remove_columns)


# In[13]:


'country_destination' in remove_columns


# In[ ]:





# ### 3. Generating highly correlated columns set

# In[14]:


fi_weights = ff.get_normalized_feature_weights(save_to)
len(fi_weights), fi_weights['age']


# In[15]:


df.drop(remove_columns, axis=1, inplace=True)
df.shape


# In[16]:


cdf = ff.get_correlated_features_pairs(df, fi_weights, threshold=0.95)
cdf.shape


# In[17]:


cdf.head()


# In[18]:


ccs = ff.get_connected_components(cdf)
len(ccs)


# In[19]:


ccs[-5]


# In[20]:


corr = abs(df.corr())


# In[21]:


ix = -5
corr[ccs[ix]][corr.index.isin(ccs[ix]) | corr.index.isin(ccs[ix])]


# In[25]:


fi[fi.feature_name.isin(ccs[ix])].head()


# In[27]:


# len(highly_correlated_features)


# In[22]:


highly_correlated_features = ff.get_highly_correlated_removal_candidates(ccs, fi_weights, verbose=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


remove_columns = remove_columns.union(set(highly_correlated_features))
len(remove_columns)


# In[ ]:


'country_destination' in remove_columns


# In[ ]:





# ### 4. Dropping filtered columns and saving data. Repeating same process for test set

# In[27]:


remove_columns = list(set(remove_columns).intersection(set(df)))
df.drop(remove_columns, axis=1, inplace=True)
df.shape


# In[28]:


test_df = pd.read_parquet('../data/processed/test_features.parquet')
test_df.shape


# In[29]:


test_df.drop(remove_columns, axis=1, inplace=True)
test_df.shape


# In[ ]:





# In[32]:


df.to_parquet('../data/processed/train_features_uncorr.parquet')


# In[33]:


test_df.to_parquet('../data/processed/test_features_uncorr.parquet')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


fi[fi.feature_name.str.startswith('ai_')].head(20)


# In[ ]:





# In[ ]:





# In[2]:


users = pd.read_parquet('../data/processed/users.parquet')
users.shape


# In[3]:


sessions = pd.read_parquet('../data/processed/sessions.parquet')
sessions.shape


# In[5]:


df = pd.read_parquet('../data/processed/train_features.parquet')
df.shape


# In[7]:


df.country_destination.value_counts(normalize=True)


# In[ ]:


df.country_destination.value_counts(dropna=False)


# In[ ]:


(NDF, US, other) - (Universum - (NDF, US, other))


# In[4]:


df.head()


# In[ ]:


'action_info' in list


# In[3]:


df.country_destination.value_counts()


# In[13]:


other_countries_ai = df[~df.country_destination.isin({'NDF', 'US', 'other'})].action_info.tolist()


# In[ ]:




