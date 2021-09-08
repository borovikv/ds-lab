#!/usr/bin/env python
# coding: utf-8

# In[1]:


from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
get_ipython().run_line_magic('load_ext', 'autotime')


# ### 0. Loading test data and model

# In[2]:


x_test = pd.read_pickle('../data/processed/x_test.pickle')
y_test = pd.read_pickle('../data/processed/y_test.pickle')
x_test.shape, y_test.shape


# In[7]:


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


# In[8]:


model = CatBoostClassifier()
model.load_model('../models/model_2021_09_06_14_22_11.cbm')


# In[9]:


test_pool = Pool(x_test, label=y_test,cat_features=cat_features)


# In[10]:


fi = model.get_feature_importance(test_pool)


# In[11]:


fi = pd.DataFrame({'feature_name': list(x_test), 'weight': fi})
fi = fi.sort_values('weight', ascending=False).reset_index(drop=True)


# In[12]:


fi.head(20)


# In[11]:


fi.head(20)


# In[ ]:





# In[ ]:




