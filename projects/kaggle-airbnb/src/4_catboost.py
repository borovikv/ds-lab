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
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
get_ipython().run_line_magic('load_ext', 'autotime')


# ### 0. Loading training data

# In[2]:


# df = pd.read_parquet('../data/processed/train_features.parquet')
# df = pd.read_parquet('../data/processed/train_features_uncorr.parquet')
df = pd.read_parquet('../data/processed/train_features_undersample.parquet')
df.shape


# In[3]:


if 'train_flag' in list(df):
    df.drop('train_flag', inplace=True, axis=1)
    df.shape


# In[4]:


y, x = df.pop('country_destination'), df
x.drop('user_id', axis=1, inplace=True)


# In[ ]:





# ### 1. Splitting tain data into stratified train, validation and test sets

# In[5]:


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


# In[6]:


cat_features_remained = list(set(cat_features).intersection(set(df)))
cat_features_removed = set(cat_features) - set(df)
len(cat_features), len(cat_features_remained), len(cat_features_removed)                                      


# In[7]:


for col in cat_features_remained:
    try:
        if col not in list(x):
            print(col)
            continue
        if str(x[col].dtype) != 'category':
            x[col].fillna('', inplace=True)
            x[col] = x[col].astype('category')
            
    except Exception as e:
        print(col)
        print(e)


# In[8]:


x_train_large, x_test, y_train_large, y_test = train_test_split(
    x, 
    y, 
    train_size=0.9, 
    random_state=42,
    stratify=y
)
x_train_large.shape, x_test.shape


# In[9]:


x_train, x_validation, y_train, y_validation = train_test_split(
    x_train_large, 
    y_train_large, 
    train_size=0.8,
    random_state=42,
    stratify=y_train_large
)
x_train.shape, x_validation.shape


# ### 1.1 Checking that the ratios of the target class are invariant

# In[10]:


# y_train.value_counts() / len(y_train)


# In[11]:


# y_validation.value_counts() / len(y_validation)


# ### 2. Model Training

# In[10]:


classes = np.unique(y_train)
classes


# In[11]:


weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weights


# In[12]:


class_weights = dict(zip(classes, weights))


# In[13]:


model = CatBoostClassifier(
    iterations=100,
    random_seed=42,
#     learning_rate=0.25,
    custom_loss=['AUC', 'Accuracy'],
    loss_function='MultiClass',
#     class_weights=class_weights,
#     depth=7,
#     eval_metric
)


# In[35]:


model.fit(
    x_train, y_train,
    cat_features=cat_features_remained,
    eval_set=(x_validation, y_validation),
    early_stopping_rounds=50,
    use_best_model=True,
    verbose=False,
    plot=True
);


# ### 3. Testing model on test sets

# In[19]:


y_pred = model.predict(x_test)


# In[20]:


labels = y_validation.value_counts().index.tolist()
labels


# In[21]:


cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='all')


# ### Confusion Matrix when iterations=500

# In[22]:


plt.figure(figsize = (20, 15))
sns.heatmap(cm, annot=True, fmt='g');


# In[25]:


plt.figure(figsize = (20, 15))
sns.heatmap(cm, annot=True, fmt='g');


# ### Confusion Matrix when using class weights

# In[19]:


plt.figure(figsize = (20, 15))
sns.heatmap(cm, annot=True, fmt='g');


# In[18]:


plt.figure(figsize = (20, 15))
sns.heatmap(cm, annot=True, fmt='g');


# In[19]:


plt.figure(figsize = (20, 15))
sns.heatmap(cm, annot=True, fmt='g');


# In[ ]:





# ### 4. Training on x_train_large, i.e. on 90% of training data instead of 70%

# In[14]:


model.fit(
    x_train_large, y_train_large,
    cat_features=cat_features_remained,
    eval_set=(x_test, y_test),
    early_stopping_rounds=10,
    use_best_model=True,
    verbose=False,
    plot=True
);


# In[15]:


y_pred = model.predict(x_test)


# In[16]:


accuracy_score(y_true=y_test, y_pred=y_pred)


# In[20]:


accuracy_score(y_true=y_test, y_pred=y_pred)


# In[25]:


accuracy_score(y_true=y_test, y_pred=y_pred)


# In[26]:


accuracy_score(y_true=y_test, y_pred=y_pred)


# In[ ]:


[a1, a2, a2, a3, a1, a2]


# In[ ]:





# ### 5. Saving Model

# In[17]:


def save_model(model, ext='cbm'):
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if ext == 'cbm':
        print(f'saving model with ts: {ts}')
        model.save_model(f'../models/model_{ts}.cbm')


# In[18]:


save_model(model)


# In[ ]:





# In[ ]:





# 1. Features set reduction by
#  - columns with nan values over 99%
#  - low variance removel
#  - highly correlated features > 0.80 (both with absolute)
# 2. Change submission generating, take top X with sum of probabilities > 90%

# In[ ]:





# In[19]:


with open('../data/processed/x_test_uncorr.pickle', 'wb') as f:
    pickle.dump(x_test, f)


# In[20]:


with open('../data/processed/y_test_uncorr.pickle', 'wb') as f:
    pickle.dump(y_test, f)


# In[ ]:




