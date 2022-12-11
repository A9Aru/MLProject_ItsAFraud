#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


# In[ ]:


train=pd.read_csv("pre_processed_train.csv")
test=pd.read_csv("pre_processed_test.csv")


# In[ ]:


t1=train


# In[ ]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud","Unnamed: 0"], inplace = True)


# In[ ]:


test.drop(axis = 1, labels = ["Unnamed: 0"], inplace = True)


# In[ ]:


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
X_train = scale.fit_transform(train1)
X_test=scale.transform(test) 


# In[ ]:


grid_params = {  
'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
} 
gs=GridSearchCV(MultinomialNB(),grid_params,verbose=1,cv=3,n_jobs=-1,scoring='roc_auc')
gs=gs.fit(X_train,ytrain1)


# In[ ]:


print("Tuned Hyperparameters :", gs.best_params_)
print("Accuracy :",gs.best_score_)

