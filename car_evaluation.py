import pandas as pd 
import os
 
os.chdir('C:\\Users\\HP\\MACHINE LEARNING codes\\MINI PROJECTS\\ML Projects ANTRIX\\TARUSHI')

data = pd.read_csv('car.data')
data.shape
 
headerList = ['Buying','Maint','Doors','Persons','Lug_boot','Safety','Acceptance']

data.to_csv("car.data", header=headerList, index= False)
data2 = pd.read_csv("car.data")

data2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

"' ELECTRONIC DESIGN EVALUATION '"
data.head()

data.info()
for i in data.columns:
    print(data[i].unique(),"\t",data[i].nunique())

for i in data.columns:
    print(data[i].value_counts())
    
sns.countplot(data['Acceptance'])

#  this is an unbalanced multiclass classification problem
for i in data.columns[:-1]:
    plt.figure(figsize=(12,6))
    plt.title("For feature '%s'"%i)
    sns.countplot(data[i],hue=data['Acceptance'])

# label encoding

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    data[i]=le.fit_transform(data[i])

data.head()
for i in data.columns:
    print(data[i].value_counts)
    
# HeatMap
fig=plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True)

#  it can be seen that most of the columns shows very weak correlation with 'Acceptance'
X = data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MODEL SELECTION 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# LOGISTIC-REGESSION

logreg=LogisticRegression(solver='newton-cg',multi_class='multinomial')
logreg.fit(X_train,y_train)
pred=logreg.predict(X_test)
logreg.score(X_test,y_test)

pred, y_test
 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(pred,y_test))

# knn classifier 
knn=KNeighborsClassifier(n_jobs=-1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
knn.score(X_test,y_test)


print(classification_report(y_test,pred))

# random forest classifier 
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_jobs=-1,random_state=51)
rfc.fit(X_train,y_train)
print(rfc.score(X_test,y_test))

"'SCORE OF RANDOM FOREST IS BEST. HENCE THIS IS  THE SELECTED MODEL'"

