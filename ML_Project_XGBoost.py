#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install imbalanced-learn')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import preprocessing


# In[2]:


train=pd.read_csv("pre_processed_train.csv")
test=pd.read_csv("pre_processed_test.csv")


# In[3]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud","Unnamed: 0"], inplace = True)


# In[4]:


test.drop(axis = 1, labels = ["Unnamed: 0"], inplace = True)


# In[5]:


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)


# In[6]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
X_train = scale.fit_transform(train1)
X_test=scale.transform(test) 


# In[7]:


abc = XGBClassifier(
        n_estimators=5000,
        max_depth=12,
        learning_rate=0.002,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-1,
        eval_metric='auc',
        tree_method='gpu_hist'
    )       
abc.fit(X_train,ytrain1)


# In[9]:


y_pred1 = abc.predict_proba(X_test)
y_pred1.sum()


# In[ ]:


y_pred = np.array([])
y_pred = []
for i in range(len(y_pred1)):
  if(y_pred1[i][0]<0.06):
    y_pred.append(1)
  else:
    y_pred.append(0)


# In[ ]:





# In[10]:


import csv
with open('xgboost.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_pred1:
        writer.writerow([idx,i])
        idx=idx+1


# In[ ]:




