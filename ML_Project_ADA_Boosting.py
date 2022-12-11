#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install imbalanced-learn')
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier


# In[2]:


train=pd.read_csv("pre_processed_train.csv")
test=pd.read_csv("pre_processed_test.csv")


# In[3]:


t1=train


# In[4]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud","Unnamed: 0"], inplace = True)


# In[5]:


test.drop(axis = 1, labels = ["Unnamed: 0"], inplace = True)


# In[6]:


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)


# In[7]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
X_train = scale.fit_transform(train1)
X_test=scale.transform(test) 


# In[9]:


abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
grid_params = {'base_estimator__max_depth':[10],
               'base_estimator__min_samples_leaf':[10],
               'n_estimators':[10, 100, 500],
               'learning_rate':[0.001,0.1,1.0]
}
gs=GridSearchCV(abc,grid_params,verbose=3,cv=3,n_jobs=-1,scoring='roc_auc')
gs=gs.fit(X_train,ytrain1)


# In[10]:


print("Tuned Hyperparameters :", gs.best_params_)
print("Accuracy :",gs.best_score_)


# In[11]:


y_pred1 = gs.predict(X_test)
y_pred1.sum()


# In[12]:


import csv
with open('adaboost.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_pred1:
        writer.writerow([idx,i])
        idx=idx+1


# In[ ]:




