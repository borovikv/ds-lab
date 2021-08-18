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


df = pd.read_parquet('../data/processed/test_features.parquet')
df.shape


# In[3]:


df.drop('train_flag', inplace=True, axis=1)
df.shape


# In[4]:


df.reset_index(drop=True, inplace=True)


# In[5]:


df.head()


# In[6]:


df.drop('country_destination', axis=1, inplace=True)


# In[7]:


x = df.drop('user_id', axis=1)


# In[8]:


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
]


# In[9]:


for col in cat_features:
    x[col].fillna('', inplace=True)
    x[col] = x[col].astype('category')


# ### 2. Loading model

# In[10]:


def find_latest_model(path='../models/', ext='cbm'):
    files = glob.glob(f"{path}*.{ext}")
    files = [file for file in files if len(file) > 25]
    files = sorted(files)
    return files[-1]


# In[11]:


path = find_latest_model()
path


# In[12]:


model = CatBoostClassifier()
model.load_model(path)


# ### 3. Predicting Country of Destination

# In[10]:


# x_pool = Pool(x, cat_features=cat_features)


# In[11]:


# preds = model.predict(x, prediction_type='Class')
# preds = [el[0] for el in preds.tolist()]


# In[12]:


# Counter(preds)


# ### 3.2 Predicting Country of Destination using multiple predicitons where applicable

# In[13]:


# classes = list(model.classes_)
# classes

# preds = model.predict_proba(x)
# preds = preds.tolist()
# # preds = [el[0] for el in preds]
# preds_df = pd.DataFrame(preds)
# preds_df.shape

# preds_df['preds'] = preds

# preds_df['amax'] = preds_df.preds.apply(lambda x: max(x))

# preds_df.head()

# def find_nth_max_index(x, ix=1):
#     second_max = sorted(x)[-ix]
#     return x.index(second_max)


# def get_nth_best_result(x, ix=1):
#     return classes[find_nth_max_index(x, ix=ix)]

# find_nth_max_index(preds_df.loc[0].preds), get_nth_best_result(preds_df.loc[0].preds)

# preds_df['second'] = preds_df.preds.apply(lambda x: get_nth_best_result(x, ix=2))
# preds_df['first'] = preds_df.preds.apply(lambda x: get_nth_best_result(x))

# preds_df.head()

# preds_df['result'] = preds_df[['first', 'second']].apply(lambda x: list(x), axis=1)

# preds_df.head()

# mask = preds_df.amax < 0.65
# mask.sum()

# preds_df.loc[~mask, 'result'] = preds_df.loc[~mask, 'first']

# preds_df['id'] = df['user_id']

# preds_df.sample(10, random_state=42)

# submission = preds_df[['id', 'result']].explode('result')
# submission.shape

# submission.head()

# submission[submission.id == 'ycr4e6e5qv']


# ### 3.3 Predicting 5 Countries of Destination per each user

# In[13]:


classes = list(model.classes_)
classes


# In[14]:


preds = model.predict_proba(x)
preds = preds.tolist()


# In[15]:


preds_df = pd.DataFrame({'preds': preds})
preds_df['amax'] = preds_df.preds.apply(lambda x: max(x))


# In[16]:


preds_df.head()


# In[17]:


def find_max_n_indeces(x, n=1):
    n_indeces = sorted(x, reverse=True)[:n]
    n_indeces = [x.index(el) for el in n_indeces]
    return n_indeces


def get_best_n_result(x, n=5):
    n_indeces = find_max_n_indeces(x, n=n)
    best_n = [classes[ix] for ix in n_indeces]
    return best_n


# In[18]:


find_max_n_indeces(preds_df.loc[0].preds, n=5), get_best_n_result(preds_df.loc[0].preds, n=5)


# In[19]:


preds_df['best5'] = preds_df.preds.apply(lambda x: get_best_n_result(x, n=5))


# In[20]:


preds_df.head()


# In[21]:


preds_df['id'] = df['user_id']


# In[22]:


preds_df.head()


# In[23]:


submission = preds_df[['id', 'best5']].explode('best5')
submission.shape


# In[24]:


submission.head()


# In[ ]:





# In[ ]:





# ### 4 Saving NDCG aware submission

# In[25]:


submission.columns = ['id', 'country']


# In[26]:


path[-24:-4]


# In[27]:


submission.to_csv(f'../data/results/submission{path[-24:-4]}.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




