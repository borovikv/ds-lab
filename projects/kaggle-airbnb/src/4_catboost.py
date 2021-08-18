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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
get_ipython().run_line_magic('load_ext', 'autotime')


# ### 0. Loading training data

# In[2]:


df = pd.read_parquet('../data/processed/train_features.parquet')
df.shape


# In[3]:


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


x.gender.dtype


# In[7]:


for col in cat_features:
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


# In[ ]:


x_train, x_validation, y_train, y_validation = train_test_split(
    x_train_large, 
    y_train_large, 
    train_size=0.8,
    random_state=42,
    stratify=y_train_large
)
x_train.shape, x_validation.shape


# ### 1.1 Checking that the ratios of the target class are invariant

# In[9]:


# y_train.value_counts() / len(y_train)


# In[10]:


# y_validation.value_counts() / len(y_validation)


# ### 2. Model Training

# In[11]:


classes = np.unique(y_train)
classes


# In[12]:


weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weights


# In[13]:


class_weights = dict(zip(classes, weights))


# In[9]:


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


# In[21]:


model.fit(
    x_train, y_train,
    cat_features=cat_features,
    eval_set=(x_validation, y_validation),
    early_stopping_rounds=50,
    use_best_model=True,
    verbose=False,
    plot=True
);


# ### 3. Testing model on test sets

# In[22]:


y_pred = model.predict(x_test)


# In[23]:


labels = y_validation.value_counts().index.tolist()
labels


# In[24]:


cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='all')


# ### Confusion Matrix when iterations=500

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

# In[10]:


model.fit(
    x_train_large, y_train_large,
    cat_features=cat_features,
    eval_set=(x_test, y_test),
    early_stopping_rounds=10,
    use_best_model=True,
    verbose=False,
    plot=True
);


# In[11]:


y_pred = model.predict(x_test)


# In[12]:


accuracy_score(y_true=y_test, y_pred=y_pred)


# In[13]:


accuracy_score(y_true=y_test, y_pred=y_pred)


# In[ ]:





# ### 5. Saving Model

# In[13]:


def save_model(model, ext='cbm'):
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if ext == 'cbm':
        print(f'saving model with ts: {ts}')
        model.save_model(f'../models/model_{ts}.cbm')


# In[14]:


save_model(model)


# In[ ]:




