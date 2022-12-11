#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


# In[2]:


train=pd.read_csv("pre_processed_train.csv")
test=pd.read_csv("pre_processed_test.csv")


# In[3]:


train


# In[4]:


t1=train


# In[5]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud","Unnamed: 0"], inplace = True)


# In[9]:


test.drop(axis = 1, labels = ["Unnamed: 0"], inplace = True)


# In[6]:


over = SMOTE(sampling_strategy=0.04)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)


# In[7]:


ytrain


# In[10]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
X_train = scale.fit_transform(train1)
X_test=scale.transform(test) 


# In[19]:


solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l1','l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
max_iter=[100000]
grid = dict(solver=solvers,penalty=penalty,C=c_values,max_iter=max_iter)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[20]:


logreg = LogisticRegression()
grid_search = GridSearchCV(estimator=logreg, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, ytrain1)


# In[21]:


print("Tuned Hyperparameters :", grid_result.best_params_)
print("Accuracy :",grid_result.best_score_)


# In[22]:


classifier = LogisticRegression(C=0.1,penalty='l2',solver="lbfgs",max_iter=100000)
classifier.fit(X_train, ytrain1)
y_pred1 = classifier.predict(X_test)


# In[23]:


y_pred1.sum()


# In[17]:


import csv
with open('log_reg.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_pred1:
        writer.writerow([idx,i])
        idx=idx+1

