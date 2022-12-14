#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


np.logspace(-3,3,7)


# In[ ]:


get_ipython().system('pip install imbalanced-learn')
import imblearn


# In[ ]:


get_ipython().system('pip install xgboost')
import xgboost as xgb


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 
import seaborn as sns
train=pd.read_csv("/content/drive/MyDrive/Data/train.csv")
test=pd.read_csv("/content/drive/MyDrive/Data/test.csv")
train.head()


# **Pre-Processing & EDA**
# 
# First we check how much missing data does each column have.

# In[ ]:


limit=(train["isFraud"].sum()/len(train))*100
train["isFraud"].sum()


# In[ ]:


columns=[]
emptytrainpercent=[]
for i in train:
    columns.append(i)
    percent=(train[i].isnull().sum())/len(train)
    percent=percent*100
    emptytrainpercent.append(percent)
list_of_tuples = list(zip(columns, emptytrainpercent))
emptytrainpercentdf=pd.DataFrame(list_of_tuples,columns=["Column Name","Empty %"])
emptytrainpercentdf=emptytrainpercentdf.sort_values(by="Empty %",ascending=False)
emptytrainpercentdf


# In[ ]:


columns=[]
emptytestpercent=[]
for i in test:
    columns.append(i)
    percent=(test[i].isnull().sum())/len(test)
    percent=percent*100
    emptytestpercent.append(percent)
list_of_tuples1 = list(zip(columns, emptytestpercent))
emptytestpercentdf=pd.DataFrame(list_of_tuples1,columns=["Column Name","Empty %"])
emptytestpercentdf=emptytestpercentdf.sort_values(by="Empty %",ascending=False)
emptytestpercentdf


# We remove all the columns which have more missing data then no. of frauds (very high amount of missing data, so columns are not useful)
# 

# In[ ]:


for ind in emptytrainpercentdf.index:
    if((100-emptytrainpercentdf["Empty %"][ind])<limit):
        train.drop([emptytrainpercentdf["Column Name"][ind]], axis=1,inplace=True)
        test.drop([emptytrainpercentdf["Column Name"][ind]], axis=1,inplace=True)
    else:
        break


# Now we will try to reduce no. of V columns as they are the ones which are high in number and can be reduced. We will be start by sorting them based on the missing value percent.

# In[ ]:


vcols=emptytrainpercentdf.loc[emptytrainpercentdf["Column Name"].str.startswith("V")]
vcols


# In[ ]:


vcols["Empty %"]=vcols["Empty %"].astype(int)
vcols


# We observe that many of this columns have the same exact percent of missing data.
# We will make groups according to that and make a correlation chart.

# In[ ]:


temp1=vcols["Empty %"][196]
vgroups=[]
tempgroup=[]
for i in vcols.index:
    if(vcols["Empty %"][i]==temp1):
        tempgroup.append(vcols["Column Name"][i])
    else:
        vgroups.append(tempgroup)
        tempgroup=[]
        tempgroup.append(vcols["Column Name"][i])
        temp1=vcols["Empty %"][i]
vgroups.append(tempgroup)
len(vgroups)


# In[ ]:


vgroups[0].sort()
vtemp=train[vgroups[0]]


# In[ ]:


#plt.figure(figsize=(30,30))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group0 = ["V140", "V142", "V164", "V165", "V322", "V323", "V324", "V331", "V332", "V333", "V145", "V150", "V151", "V159", "V160", "V147", "V149", "V153", "V154", "V155", "V156", "V157", "V158", "V162", "V163", "V327", "V329", "V330", "V334", "V335", "V337", "V338"]
train.drop(axis = 1, labels = drop_group0, inplace = True)
test.drop(axis = 1, labels = drop_group0, inplace = True)
vgroup0 = []


# In[ ]:





# In[ ]:


vgroups[1].sort()
vtemp = train[vgroups[1]]


# In[ ]:


#plt.figure(figsize = (30,30))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group1 = ["V218", "V219", "V231", "V232", "V233", "V273", "V275", "V225", "V253", "V230", "V243", "V248", "V254", "V235", "V236", "V244", "V257", "V249", "V262", "V263", "V264", "V269", "V268", "V276", "V277"]
train.drop(axis = 1, labels = drop_group1, inplace = True)
test.drop(axis = 1, labels = drop_group1, inplace = True)
vgroup1 = []


# In[ ]:





# In[ ]:


vgroups[2].sort()
vtemp = train[vgroups[2]]


# In[ ]:


#plt.figure(figsize = (40,40))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group2 = ["V168", "V177", "V178", "V179", "V202", "V204", "V211", "V212", "V213", "V191", "V196", "V207", "V190", "V199", "V182", "V183", "V203", "V192", "V193", "V189", "V197", "V198", "V201", "V214", "V215", "V222", "V245", "V259", "V239", "V251", "V256", "V270", "V271"]
train.drop(axis = 1, labels = drop_group2, inplace = True)
test.drop(axis = 1, labels = drop_group2, inplace = True)
vgroup2 = []


# In[ ]:





# In[ ]:


vgroups[3].sort()
vtemp = train[vgroups[3]]


# In[ ]:


#plt.figure(figsize = (10,10))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group3 = ["V11", "V5"]
train.drop(axis = 1, labels = drop_group3, inplace = True)
test.drop(axis = 1, labels = drop_group3, inplace = True)
vgroup3 = []


# In[ ]:


vgroups[4].sort()
vtemp = train[vgroups[4]]


# In[ ]:


#plt.figure(figsize = (10,10))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group4 = ["V36", "V38", "V40", "V42", "V50", "V45", "V49", "V50", "V52"]
train.drop(axis = 1, labels = drop_group4, inplace = True)
test.drop(axis = 1, labels = drop_group4, inplace = True)
vgroup4 = []


# In[ ]:


vgroups[5].sort()
vtemp = train[vgroups[5]]


# In[ ]:


#plt.figure(figsize = (10, 10))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group5 = ["V76", "V92", "V93", "V94", "V81", "V84", "V83", "V93", "V87", "V91"]
train.drop(axis = 1, labels = drop_group5, inplace = True)
test.drop(axis = 1, labels = drop_group5, inplace = True)
vgroup5 = []


# In[ ]:


vgroups[6].sort()
vtemp = train[vgroups[6]]


# In[ ]:


#plt.figure(figsize = (15, 15))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group6 = ["V54", "V58", "V71", "V72", "V73", "V74", "V60", "V63", "V64", "V62", "V70"]
train.drop(axis = 1, labels = drop_group6, inplace = True)
test.drop(axis = 1, labels = drop_group6, inplace = True)
vgroup6 = []


# In[ ]:


vgroups[7].sort()
vtemp = train[vgroups[7]]


# In[ ]:


#plt.figure(figsize = (15, 15))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group7 = ["V13", "V16", "V31", "V32", "V33", "V34", "V18", "V21", "V22", "V20", "V28"]
train.drop(axis = 1, labels = drop_group7, inplace = True)
test.drop(axis = 1, labels = drop_group7, inplace = True)
vgroup7 = []


# In[ ]:


vgroups[8].sort()
vtemp = train[vgroups[8]]


# In[ ]:


#plt.figure(figsize = (50, 50))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group8 = ["V287", "V102", "V103", "V126", "V127", "V128", "V132", "V133", "V134", "V279", "V280", "V293", "V294", "V295", "V306", "V307", "V308", "V316", "V317", "V318", "V95", "V96", "V97", "V106", "V297", "V299", "V296", "V298", "V299", "V111", "V112", "V119", "V309", "V311", "V310", "V312", "V135", "V136", "V99", "V289", "V292", "V301", "V302", "V303", "V315", "V319", "V320"]
train.drop(axis = 1, labels = drop_group8, inplace = True)
test.drop(axis = 1, labels = drop_group8, inplace = True)
vgroup8 = []


# In[ ]:


Cs = [x for x in train.columns.values.tolist() if x.startswith("C")]
ctemp = train[Cs]


# In[ ]:


#plt.figure(figsize = (10, 10))
#sns.heatmap(ctemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group_c = ["C1", "C2", "C4", "C6", "C7", "C8", "C11", "C12", "C14", "C9"]
train.drop(axis = 1, labels = drop_group_c, inplace = True)
test.drop(axis = 1, labels = drop_group_c, inplace = True)


# In[ ]:


d_clmn = [x for x in train.columns.values.tolist() if (x.startswith("D"))]
d_temp = train[d_clmn]


# In[ ]:


#plt.figure(figsize = (10, 10))
#sns.heatmap(d_temp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group_d = ["D2", "D6", "D12", "D7"]
train.drop(axis = 1, labels = drop_group_d, inplace = True)
test.drop(axis = 1, labels = drop_group_d, inplace = True)


# In[ ]:


Vs = [x for x in train.columns.values.tolist() if x.startswith("V")]
vtemp = train[Vs]


# In[ ]:


#plt.figure(figsize = (50, 50))
#sns.heatmap(vtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group_v = ["V29","V30","V48","V69","V90","V41","V53","V57","V79","V39","V43","V59","V80","V85","V68","V89","V51","V65","V88","V90"]
train.drop(axis = 1, labels = drop_group_v, inplace = True)
test.drop(axis = 1, labels = drop_group_v, inplace = True)


# In[ ]:


otherthanv=['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','C3','C5','C10','C13','D1','D3','D4','D10','D11','D15','M1','M2','M3','M4','M6']
otvtemp = train[otherthanv]


# In[ ]:


#plt.figure(figsize = (50, 50))
#sns.heatmap(otvtemp.corr(), vmin=-1, cmap="coolwarm", annot=True)


# In[ ]:


drop_group_otv = ['TransactionDT']
train.drop(axis = 1, labels = drop_group_otv, inplace = True)
test.drop(axis = 1, labels = drop_group_otv, inplace = True)


# In[ ]:


sns.pairplot(train,vars=["TransactionAmt"],hue="isFraud")


# In[ ]:


train["isFraud"].sum()


# In[ ]:


train.boxplot(by ='isFraud', column =['TransactionAmt'], grid = False)


# In[ ]:


train.boxplot( column =['TransactionAmt'], grid = False)


# In[ ]:


skewness=train.skew()


# In[ ]:


kurt=train.kurt()


# In[ ]:


kurt=kurt.to_frame()
skewness=skewness.to_frame()


# In[ ]:


kurt=kurt.sort_values(by=0)
skewness=skewness.sort_values(by=0)


# In[ ]:


for i in skewness.index:
    if(skewness[0][i]>5.1):
        train[i]=np.sqrt(train[i])
        test[i]=np.sqrt(test[i])
skewness=train.skew().to_frame().sort_values(by=0)
skewness


# In[ ]:


skewness=train.skew().to_frame().sort_values(by=0)
skewness


# In[ ]:


for i in skewness.index:
    if(skewness[0][i]<-4):
        train[i]=np.square(train[i])
        test[i]=np.square(test[i])
skewness=train.skew().to_frame().sort_values(by=0)
skewness


# In[ ]:


for ind in skewness.index:
    if(skewness[0][ind]<-4):
        train.drop([ind], axis=1,inplace=True)
        test.drop([ind], axis=1,inplace=True)
skewness=train.skew().to_frame().sort_values(by=0)
skewness


# In[ ]:


for ind in skewness.index:
    if(skewness[0][ind]>10):
        train.drop([ind], axis=1,inplace=True)
        test.drop([ind], axis=1,inplace=True)
skewness=train.skew().to_frame().sort_values(by=0)
skewness


# In[ ]:


train


# In[ ]:


columns=[]
emptytrainpercent=[]
for i in train:
    columns.append(i)
    percent=(train[i].isnull().sum())/len(train)
    percent=percent*100
    emptytrainpercent.append(percent)
list_of_tuples = list(zip(columns, emptytrainpercent))
emptytrainpercentdf=pd.DataFrame(list_of_tuples,columns=["Column Name","Empty %"])
emptytrainpercentdf=emptytrainpercentdf.sort_values(by="Empty %",ascending=False)
emptytrainpercentdf


# In[ ]:


for ind in emptytrainpercentdf.index:
    if((100-emptytrainpercentdf["Empty %"][ind])<25 and emptytrainpercentdf["Column Name"][ind]!="R_emaildomain"):
        train.drop([emptytrainpercentdf["Column Name"][ind]], axis=1,inplace=True)
        test.drop([emptytrainpercentdf["Column Name"][ind]], axis=1,inplace=True)


# In[ ]:


columns=[]
emptytrainpercent=[]
for i in train:
    columns.append(i)
    percent=(train[i].isnull().sum())/len(train)
    percent=percent*100
    emptytrainpercent.append(percent)
list_of_tuples = list(zip(columns, emptytrainpercent))
emptytrainpercentdf=pd.DataFrame(list_of_tuples,columns=["Column Name","Empty %"])
emptytrainpercentdf=emptytrainpercentdf.sort_values(by="Empty %",ascending=False)
emptytrainpercentdf


# In[ ]:


train["isFraud"].sum()


# In[ ]:


t1=train["P_emaildomain"].unique()


# In[ ]:


out = [d for _, d in train.groupby('isFraud')]


# In[ ]:


t2=out[1]["P_emaildomain"].unique()


# In[ ]:


l=[]
for i in train.index:
    l.append(0)
l2=[]
for i in test.index:
    l2.append(0)
train["P"]=l
test["P"]=l2


# In[ ]:


for i in train.index:
    if(train["P_emaildomain"][i] in t2):
        train["P"][i]=1
for i in test.index:
    if(test["P_emaildomain"][i] in t2):
        test["P"][i]=1     


# In[ ]:


drop_group_p = ['P_emaildomain']
train.drop(axis = 1, labels = drop_group_p, inplace = True)
test.drop(axis = 1, labels = drop_group_p, inplace = True)


# In[ ]:


t1=train["R_emaildomain"].unique()
t2=out[1]["R_emaildomain"].unique()
l=[]
for i in train.index:
    l.append(0)
l2=[]
for i in test.index:
    l2.append(0)
train["R"]=l
test["R"]=l2
for i in train.index:
    if(train["R_emaildomain"][i] in t2):
        train["R"][i]=1
for i in test.index:
    if(test["R_emaildomain"][i] in t2):
        test["R"][i]=1
drop_group_r = ['R_emaildomain']
train.drop(axis = 1, labels = drop_group_r, inplace = True)
test.drop(axis = 1, labels = drop_group_r, inplace = True)  


# In[ ]:


train["M4"].fillna("M3",inplace=True)
test["M4"].fillna("M3",inplace=True)


# In[ ]:


for col in train.columns:
    if pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_float_dtype(train[col]):
        if train[col].nunique()<=10:
            train[col].fillna(train[col].median(),inplace=True)
            if(col!="isFraud"):
                test[col].fillna(train[col].median(),inplace=True)
        else:
            train[col].fillna(train[col].mean(),inplace=True)
            if(col!="isFraud"):
                test[col].fillna(train[col].mean(),inplace=True)


# In[ ]:


columns=[]
emptytrainpercent=[]
for i in train:
    columns.append(i)
    percent=(train[i].isnull().sum())/len(train)
    percent=percent*100
    emptytrainpercent.append(percent)
list_of_tuples = list(zip(columns, emptytrainpercent))
emptytrainpercentdf=pd.DataFrame(list_of_tuples,columns=["Column Name","Empty %"])
emptytrainpercentdf=emptytrainpercentdf.sort_values(by="Empty %",ascending=False)
emptytrainpercentdf


# In[ ]:


train["M1"].value_counts()


# In[ ]:


train["M1"].fillna("T",inplace=True)
test["M1"].fillna("T",inplace=True)


# In[ ]:


train["M2"].value_counts()


# In[ ]:


train["M2"].fillna("N",inplace=True)
test["M2"].fillna("N",inplace=True)


# In[ ]:


train["M3"].value_counts()


# In[ ]:


train["M3"].fillna("N",inplace=True)
test["M3"].fillna("N",inplace=True)


# In[ ]:


train["M6"].value_counts()


# In[ ]:


train["M6"].fillna("N",inplace=True)
test["M6"].fillna("N",inplace=True)


# In[ ]:


train["card4"].value_counts()


# In[ ]:


train["card4"].fillna("others",inplace=True)
test["card4"].fillna("others",inplace=True)


# In[ ]:


train["card6"].value_counts()


# In[ ]:


train["card6"].fillna("debit",inplace=True)
test["card6"].fillna("debit",inplace=True)


# In[ ]:


columns=[]
emptytrainpercent=[]
for i in train:
    columns.append(i)
    percent=(train[i].isnull().sum())/len(train)
    percent=percent*100
    emptytrainpercent.append(percent)
list_of_tuples = list(zip(columns, emptytrainpercent))
emptytrainpercentdf=pd.DataFrame(list_of_tuples,columns=["Column Name","Empty %"])
emptytrainpercentdf=emptytrainpercentdf.sort_values(by="Empty %",ascending=False)
emptytrainpercentdf


# In[ ]:


train["M5"].value_counts()


# In[ ]:


train["M5"].fillna("N",inplace=True)
test["M5"].fillna("N",inplace=True)


# In[ ]:


train["M7"].value_counts()


# In[ ]:


train["M7"].fillna("F",inplace=True)
test["M7"].fillna("F",inplace=True)


# In[ ]:


train["M8"].value_counts()


# In[ ]:


train["M8"].fillna("N",inplace=True)
test["M8"].fillna("N",inplace=True)


# In[ ]:


train["M9"].value_counts()


# In[ ]:


train["M9"].fillna("F",inplace=True)
test["M9"].fillna("F",inplace=True)


# In[ ]:


columns=[]
emptytestpercent=[]
for i in test:
    columns.append(i)
    percent=(test[i].isnull().sum())/len(test)
    percent=percent*100
    emptytestpercent.append(percent)
list_of_tuples1 = list(zip(columns, emptytestpercent))
emptytestpercentdf=pd.DataFrame(list_of_tuples1,columns=["Column Name","Empty %"])
emptytestpercentdf=emptytestpercentdf.sort_values(by="Empty %",ascending=False)
emptytestpercentdf


# In[ ]:


label_encoder = LabelEncoder()
one_hot_columns=[]
for col in train.columns:
    if not (pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_float_dtype(train[col])):
        if train[col].nunique()<=2:
            train[col] = label_encoder.fit_transform(train[col])
            test[col] = label_encoder.fit_transform(test[col])
        else:
            one_hot_columns.append(col)
train= pd.get_dummies(train, columns =one_hot_columns)
test= pd.get_dummies(test, columns =one_hot_columns)


# In[ ]:


train


# In[ ]:


test


# In[ ]:


tol1=train


# In[ ]:


for col in train.columns:
    if pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_float_dtype(train[col]):
        if train[col].nunique()>10 and train[col].nunique()!=train.shape[0]:
            upper_limit = np.percentile(train[col],98)
            lower_limit = np.percentile(train[col],2)
            print(col)
            print(upper_limit)
            print(lower_limit)
            tol1[col] = np.where(train[col]>upper_limit,
                                  upper_limit,
                                  np.where(train[col]<lower_limit,
                                        lower_limit,
                                        train[col]
                                    )
                                )


# In[ ]:


train.to_csv("/pre_processed_train.csv")


# In[ ]:


test.to_csv("/pre_processed_test.csv")


# In[ ]:


ytrain=train["isFraud"]
train.drop(axis = 1, labels = ["isFraud"], inplace = True)


# In[ ]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over = SMOTE(sampling_strategy=0.04)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)
train1, ytrain1 = pipeline.fit_resample(train, ytrain)


# In[ ]:


train1


# In[ ]:


ytrain1


# In[ ]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
train1= scale.fit_transform(train1) 


# In[ ]:


train1


# In[ ]:


test1 = scale.transform(test) 


# In[ ]:


test1


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 1,max_iter=100000)
classifier1 = LogisticRegression(random_state = 1,max_iter=100000)
classifier.fit(train1, ytrain1)
y_pred1 = classifier.predict(test1)


# In[ ]:


y_pred1.sum()


# In[ ]:


import csv
with open('/log_reg.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_pred1:
        writer.writerow([idx,i])
        idx=idx+1


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100)
clf1 = RandomForestClassifier(n_estimators = 100)
clf.fit(train1, ytrain1)
y_predrf1 = clf.predict(test1)


# In[ ]:


y_predrf1.sum()


# In[ ]:


import csv
with open('/ran_for.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_predrf1:
        writer.writerow([idx,i])
        idx=idx+1


# In[ ]:


from sklearn import linear_model
SGDClf = linear_model.SGDClassifier(max_iter = 100000, tol=1e-3,penalty = "elasticnet")
SGDClf1 = linear_model.SGDClassifier(max_iter = 100000, tol=1e-3,penalty = "elasticnet")
SGDClf.fit(train1, ytrain1)
y_predsvc1 = SGDClf.predict(test1)


# In[ ]:


y_predsvc1.sum()


# In[ ]:


import csv
with open('/ans2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in y_predsvc1:
        writer.writerow([idx,i])
        idx=idx+1


# In[ ]:


xgb_cl1 = xgb.XGBClassifier(learning_rate=1e-3,max_depth=5,subsample=0.75,colsample_bytree=0.75)
xgb_cl2 = xgb.XGBClassifier(learning_rate=1e-3,max_depth=5,subsample=0.75,colsample_bytree=0.75)
xgb_cl1.fit(train1, ytrain1)
preds1 = xgb_cl1.predict(test1)


# In[ ]:


preds1.sum()


# In[ ]:


import csv
with open('/ans3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","isFraud"])
    idx=0
    for i in preds1:
        writer.writerow([idx,i])
        idx=idx+1

