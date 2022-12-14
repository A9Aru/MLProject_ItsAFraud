#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
#import statements

# In[4]:


train=pd.read_csv("pre_processed_train.csv")
test=pd.read_csv("pre_processed_test.csv")
#loading the pre-processed datasets

# In[5]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud","Unnamed: 0"], inplace = True)
#separating target column and the rest of dataset
# In[6]:


test.drop(axis = 1, labels = ["Unnamed: 0"], inplace = True)


# In[7]:


over = SMOTE(sampling_strategy=0.05)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)

#doing over sampling and under sampling to make the dataset more balanced
# In[8]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
X_train = scale.fit_transform(train1)
X_test=scale.transform(test) 
#scaling the dataset

# In[9]:


abc = XGBClassifier(
        colsample_bytree= 0.75,
        gamma= 0.65, 
        learning_rate= 0.1,
        max_depth= 20,
        reg_alpha= 0.4,  
        objective="binary:logistic",
        n_estimators=8000,
        njobs=-1
    )       
abc.fit(X_train,ytrain1)
#fitting the model for training

# In[10]:


y_pred1 = abc.predict_proba(X_test)
y_pred1.sum()
#predicting the outputs

# In[11]:


y_pred = np.array([])
y_pred = []
for i in range(len(y_pred1)):
  if(y_pred1[i][0]<0.35):
    y_pred.append(1)
  else:
    y_pred.append(0)
#threshold setting from probabilities

# In[12]:


sum(y_pred)


# In[13]:


import csv
with open('xgboost_finalsub.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_pred:
        writer.writerow([idx,i])
        idx=idx+1
#saving the data in submittable csv files 

# In[ ]:




