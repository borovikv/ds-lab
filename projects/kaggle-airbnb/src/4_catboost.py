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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
get_ipython().run_line_magic('load_ext', 'autotime')


# ### 0. Loading training data

# In[2]:


df = pd.read_parquet('../data/processed/train_features.parquet')
df.shape


# In[3]:


y, x = df.pop('country_destination'), df
x.drop('user_id', axis=1, inplace=True)


# In[ ]:





# ### 1. Splitting tain data into stratified train, validation and test sets

# In[4]:


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


# In[5]:


x.gender.dtype


# In[6]:


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


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    train_size=0.9, 
    random_state=42,
    stratify=y
)
x_train.shape, x_test.shape


# In[8]:


x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, 
    y_train, 
    train_size=0.8,
    random_state=42,
    stratify=y_train
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


# In[20]:


model = CatBoostClassifier(
    iterations=500,
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





# ### 5. Saving Model

# In[26]:


model.save_model('../models/model3.cbm')


# In[ ]:




