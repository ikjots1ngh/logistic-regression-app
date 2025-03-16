import streamlit as st


import pandas as pd
df=pd.read_csv(r"C:\Users\Welcome\Desktop\Titanic_train.csv")
df

df.head(),df.tail()

df.info(),df.describe,df.shape

st.write(type(df))

z=df.corr(numeric_only=True)
z

import seaborn as sns
sns.heatmap(z,annot=True)

df

df.isnull().sum()

#drop na values from dataset
df=df.dropna()
df

df=df.drop("Cabin",axis=1)

df=df.drop("Embarked",axis=1)

df=df.drop("Name",axis=1)

df=df.drop("Sex",axis=1)

df=df.drop("Ticket",axis=1)

df=df.drop("PassengerId",axis=1)

df

df.shape

summary=df.describe().T
summary

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
df.hist(bins=20,figsize=(15,10),color='skyblue',edgecolor='black')
plt.suptitle("histogram of numerical values",fontsize=16)
st.pyplot(fig)

sns.pairplot(df)
st.pyplot(fig)

st.write(type(plt.figure)) 

df

X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

y

X

from sklearn.linear_model import LogisticRegression

#Logistic regression and fit the model
classifier = LogisticRegression()
classifier.fit(X,y)

#Predict for X dataset
y_pred = classifier.predict(X)
y_pred

len(y_pred)

classifier.predict([[1,0,34,12.5,2],[0,1,25,15.6,4]])

classifier.predict(X)

y_pred_df= pd.DataFrame({'actual': y,
                         'predicted': classifier.predict(X)})

y_pred_df

len(y_pred_df[(y_pred_df["actual"]== 1) & (y_pred_df["predicted"]==1)])

# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred_df["predicted"])
print (confusion_matrix)

import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);

y_pred_df["predicted"].value_counts()

#Classification report
## Precision – What percent of your predictions were correct?
## Recall – What percent of the positive cases did you catch? 
## F1 score – What percent of positive predictions were correct?
from sklearn.metrics import classification_report
st.write(classification_report(y,y_pred))

classifier.predict_proba(X)

# intercept value
classifier.intercept_

# Other coeffieicents
classifier.coef_

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba (X)[:,1])

auc = roc_auc_score(y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.legend()
st.pyplot(fig)

