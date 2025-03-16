#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Logistic Regression Assignment")

# Load dataset
df = pd.read_csv("Titanic_train.csv")

st.write("### Shape of dataset:", df.shape)

# Display a heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))  # Ensure fig is defined
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)  # Display the figure

# Histogram of numerical values
st.write("### Histogram of Numerical Columns")
fig, ax = plt.subplots(figsize=(10, 6))  # Ensure fig is defined
df.hist(bins=20, figsize=(10, 6), color="skyblue", edgecolor="black", ax=ax)
st.pyplot(fig)  # Display the figure

# Pairplot
st.write("### Pairplot of the Data")
fig = sns.pairplot(df)  # sns.pairplot automatically creates a figure
st.pyplot(fig)  # Display the figure

# Confusion Matrix
st.write("### Confusion Matrix")
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

df = df.dropna()  # Drop missing values
X = df.drop("Survived", axis=1)
y = df["Survived"]

classifier = LogisticRegression()
classifier.fit(X, y)
y_pred = classifier.predict(X)

conf_matrix = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(6, 4))  # Define fig before using it
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
st.pyplot(fig)  # Display the figure


# In[ ]:




