#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

st.title("Logistic Regression Assignment")

# Load dataset
df = pd.read_csv(r"C:\Users\Welcome\Desktop\Titanic_train.csv")
st.write(df.head())

# Display dataset info
st.write(df.info())
st.write(df.describe())
st.write("Shape of dataset:", df.shape)

# Correlation heatmap
z = df.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(z, annot=True, ax=ax)
fig, ax = plt.subplots()
st.pyplot(fig)

# Drop unnecessary columns and handle missing values
df = df.dropna()
df = df.drop(columns=["Cabin", "Embarked", "Name", "Sex", "Ticket", "PassengerId"])
st.write("Cleaned Data Shape:", df.shape)

# Summary statistics
summary = df.describe().T
st.write(summary)

# Histograms
fig, ax = plt.subplots(figsize=(15, 10))
df.hist(bins=20, color='skyblue', edgecolor='black', ax=ax)
plt.suptitle("Histogram of numerical values", fontsize=16)
fig, ax = plt.subplots()
st.pyplot(fig)

# Pairplot
fig = sns.pairplot(df)
fig, ax = plt.subplots()
st.pyplot(fig)

# Define features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train Logistic Regression model
classifier = LogisticRegression()
classifier.fit(X, y)

# Predictions
y_pred = classifier.predict(X)
y_pred_df = pd.DataFrame({'actual': y, 'predicted': y_pred})
st.write(y_pred_df.head())

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['0', '1'])
ax.yaxis.set_ticklabels(['0', '1'])
fig, ax = plt.subplots()
st.pyplot(fig)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba(X)[:, 1])
auc = roc_auc_score(y, y_pred)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='red', label='Logit model (AUC = %0.2f)' % auc)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate or [1 - True Negative Rate]')
ax.set_ylabel('True Positive Rate')
ax.legend()
fig, ax = plt.subplots()
st.pyplot(fig)


# In[ ]:




