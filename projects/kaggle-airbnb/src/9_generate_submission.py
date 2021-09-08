#!/usr/bin/env python
# coding: utf-8

# ## Generatig predicitons for the submission dataset

# In[1]:


from catboost import CatBoostClassifier, Pool
import pandas as pd
from collections import Counter
from tqdm.notebook import tqdm
import numpy as np
import glob


# ### 0. Loading and preparing data

# In[2]:


# df = pd.read_parquet('../data/processed/test_features.parquet')
# df = pd.read_parquet('../data/processed/test_features_uncorr.parquet')
df = pd.read_parquet('../data/processed/test_features.parquet')
df.shape


# In[4]:


df['nan_counts'] = df.isnull().sum(axis=1)


# In[5]:


if 'train_flag' in df:
    df.drop('train_flag', inplace=True, axis=1)
    df.reset_index(drop=True, inplace=True)
    df.shape


# In[6]:


df.head()


# In[7]:


df.drop('country_destination', axis=1, inplace=True)


# In[8]:


x = df.drop('user_id', axis=1)


# In[9]:


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


# In[10]:


cat_features_remained = list(set(cat_features).intersection(set(df)))
cat_features_removed = set(cat_features) - set(df)
len(cat_features), len(cat_features_remained), len(cat_features_removed)                                      


# In[11]:


for col in cat_features_remained:
    x[col].fillna('', inplace=True)
    x[col] = x[col].astype('category')


# ### 2. Loading model

# In[12]:


def find_latest_model(path='../models/', ext='cbm'):
    files = glob.glob(f"{path}*.{ext}")
    files = [file for file in files if len(file) > 25]
    files = sorted(files)
    return files[-1]


# In[13]:


path = find_latest_model()
path


# In[14]:


model = CatBoostClassifier()
model.load_model(path)


# ### 3.3 Predicting 5 Countries of Destination per each user

# In[15]:


classes = list(model.classes_)
classes


# In[16]:


preds = model.predict_proba(x)
preds = preds.tolist()


# In[17]:


preds_df = pd.DataFrame({'preds': preds})
preds_df['amax'] = preds_df.preds.apply(lambda x: max(x))


# In[41]:


preds_df.head()


# In[22]:


def find_max_n_indeces(x, n=1):
    n_indeces = sorted(x, reverse=True)[:n]
    n_indeces = [x.index(el) for el in n_indeces]
    return n_indeces


def get_best_n_result(x, n=5):
    n_indeces = find_max_n_indeces(x, n=n)
    best_n = [classes[ix] for ix in n_indeces]
    return best_n


# In[23]:


find_max_n_indeces(preds_df.loc[0].preds, n=5), get_best_n_result(preds_df.loc[0].preds, n=5)


# In[43]:


find_max_n_indeces(preds_df.loc[0].preds, n=5), get_best_n_result(preds_df.loc[0].preds, n=5)


# In[24]:


preds_df['best5'] = preds_df.preds.apply(lambda x: get_best_n_result(x, n=5))


# In[25]:


preds_df.head()


# In[45]:


preds_df.head()


# In[26]:


preds_df['id'] = df['user_id']


# In[27]:


preds_df.head()


# In[28]:


submission = preds_df[['id', 'best5']].explode('best5')
submission.shape


# In[29]:


submission.head()


# In[ ]:





# In[ ]:





# ### 4 Saving NDCG aware submission

# In[30]:


submission.columns = ['id', 'country']


# In[31]:


path[-24:-4]


# In[32]:


submission.to_csv(f'../data/results/submission{path[-24:-4]}.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


def get_sum_of_top_n(x, n=1):
    return sum(sorted(x, reverse=True)[:n])


# In[39]:


preds_df['sum5'] = preds_df.preds.apply(lambda x: get_sum_of_top_n(x, 5))
preds_df['sum4'] = preds_df.preds.apply(lambda x: get_sum_of_top_n(x, 4))


# In[104]:


preds_df.head()


# In[129]:


s = preds_df[['id', 'preds']].explode('preds').copy(deep=True)
s.shape


# In[130]:


s.preds = s.preds.astype(float)


# In[131]:


classes


# In[132]:


s['country'] = classes * int(len(s) / len(classes))


# In[133]:


s.head()


# In[134]:


type(s.loc[0].preds)


# In[135]:


s = s.sort_values(['id', 'preds'], ascending=[True, False])
s.reset_index(drop=True, inplace=True)


# In[136]:


ss = s[['id', 'preds']].groupby('id', as_index=False).preds.cumsum()


# In[137]:


s['preds_sum'] = ss


# In[138]:


s.head(15)


# In[183]:


best = s.groupby('id', as_index=False).head(4)
best.shape


# In[184]:


best.head()


# In[185]:


result = s[s.preds_sum < 0.90]
result.shape, s.shape, result.id.nunique(), s.id.nunique()


# In[186]:


result = pd.concat([best, result])
result = result.drop_duplicates(['id', 'country'])
result.shape, s.shape, result.id.nunique(), s.id.nunique()


# In[187]:


result[['id', 'country']].to_csv('../data/results/submission_cumsum_head3_90.csv', index=False)
result.shape


# In[ ]:





# In[ ]:





# In[33]:


'gender' in df


# In[4]:


df.shape


# In[5]:


users = pd.get_dummies(df[['gender', 'user_id', 'country_destination']], columns=['gender'])


# In[6]:


users.head()


# In[ ]:




