#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


# In[25]:


train=pd.read_csv("pre_processed_train.csv")
test=pd.read_csv("pre_processed_test.csv")


# In[26]:


train


# In[27]:


t1=train


# In[28]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud","Unnamed: 0"], inplace = True)


# In[29]:


test.drop(axis = 1, labels = ["Unnamed: 0"], inplace = True)


# In[30]:


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)


# In[31]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
X_train = scale.fit_transform(train1)
X_test=scale.transform(test) 


# In[34]:


grid_params={'n_neighbors':[3,11,19],
             'weights': ['distance'],
             'metric': ['manhattan']
}
gs=GridSearchCV(KNeighborsClassifier(),grid_params,verbose=1,cv=3,n_jobs=-1,scoring='roc_auc')
gs=gs.fit(X_train,ytrain1)


# In[35]:


print("Tuned Hyperparameters :", gs.best_params_)
print("Accuracy :",gs.best_score_)


# In[36]:


y_pred1 = gs.predict(X_test)
y_pred1.sum()


# In[ ]:


classifier = KNeighborsClassifier(metric='manhattan',n_neighbors=19,weights='distance')
classifier.fit(X_train, ytrain1)
y_pred1 = classifier.predict(X_test)


# In[37]:


import csv
with open('knn.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_pred1:
        writer.writerow([idx,i])
        idx=idx+1


# In[39]:


np.linspace(10,100,num=10)

