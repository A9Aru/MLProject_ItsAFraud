#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
#imported libraries

# In[3]:


train = pd.read_csv("/kaggle/input/train-data/pre_processed_train.csv")
test=pd.read_csv("/kaggle/input/train-data/pre_processed_test.csv")
#loaded the pre-processed datasets

# In[4]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud","Unnamed: 0"], inplace = True)
#separated target-columns in the dataset

# In[5]:


test.drop(axis = 1, labels = ["Unnamed: 0"], inplace = True)


# In[9]:


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)

#performed over sampling and under sampling to balance out the dataset
# In[10]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
X_train = scale.fit_transform(train1)
X_test=scale.transform(test) 
#scaled the data

# In[25]:


from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
model = Sequential() 
model.add(Dense(128, activation='relu', input_dim=116))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()]) 
model.summary()
#made a 2 layer neural network with activation functions as Relu and Sigmoid using keras libraries

# In[26]:


model.fit(X_train, ytrain1,epochs=20, batch_size=100)
y_pred1 = model.predict(X_test)
y_pred1.sum()

#train the model and then predicted the results
# In[32]:


y_pred = np.array([])
y_pred = []
for i in range(len(y_pred1)):
  if(y_pred1[i][0]>0.35):
    y_pred.append(1)
  else:
    y_pred.append(0)

#classifies the probalities using a threshold value
# In[33]:


y_pred


# In[35]:


import csv
with open('neural_netwrok.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_pred:
        writer.writerow([idx,i])
        idx=idx+1

#saved the results into a submittable csv file
# In[28]:


max(y_pred1)

