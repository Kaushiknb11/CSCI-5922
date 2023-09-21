#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:30:09 2023

@author: kaushiknarasimha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


df = pd.read_csv('/Users/kaushiknarasimha/Documents/Academics/SEM3/NNs/A1_Data_Kaushik_Narasimha_Bukkapatnam.csv')
print(df)

#visualizing the data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(df['Credit Score'], df['Income'], df['Age'], s=50)
ax.set_xlabel('Credit Score')
ax.set_ylabel('Income')
ax.set_zlabel('Age')
plt.title('Dataset')
plt.show()




df1=df.copy()

#Normalizing the data using min-max method
df1['Credit Score']= (df1['Credit Score'] - df1['Credit Score'].min()) /(df1['Credit Score'].max()-df1['Credit Score'].min())
df1['Income']= (df1['Income'] - df1['Income'].min()) /(df1['Income'].max()-df1['Income'].min())
df1['Age']= (df1['Age'] - df1['Age'].min()) /(df1['Age'].max()-df1['Age'].min())
print(df1)

# creating a mapping dictionary to change labels to binary
mapping = {'Approved': 1, 'Denied': 0}
# converting the values in the "Loan Status" column
df1['Loan Status'] = df['Loan Status'].map(mapping)
print(df1)
# Renaming the "Loan Status" column to "LABEL"
df1 = df1.rename(columns={'Loan Status': 'LABEL'})
print(df1)

# storing the actual lables as y
y = df1['LABEL'].values
y = np.transpose([y])
print(y)

# extracting the feature columns (everything except the labels) as a numpy array
X = df1.drop(columns=['LABEL']).values
# Display the numpy array
print(X)

# defining the sigmoid function
def sigmoid(s, deriv=False):
    if (deriv == True):
        return s * (1 - s)
    else:
        return 1 / (1 + np.exp(-s)) 

# initializing the parameters
w = np.array([[1, 1, 1]])
b = 0
learning_rate = 0.1
epochs = 100
n = len(df1)

#defining loss function (binary cross entropy)
def LCE(y, y_hat):
    return -(np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))))/n

# define gradient descent to optimize the parameters
def gradient_descent(X, y, w, b, learning_rate, epochs):
    m = len(y)
    losses = []

    for i in range(epochs):
        print("Epoch number:\n", i)
        z=X @ w.T + b
        print("z:\n", z)
        #applying the sigmoid funtion to all values of z to get the predicted values
        y_hat = sigmoid(z)
        print("y_hat:\n", y_hat)

        # Calculate gradients
        dL_dw = (1 / m) * np.dot(X.T, (y_hat - y))
        dL_db = (1 / m) * np.sum(y_hat - y)
        print("The update for w is\n",dL_dw)
        print("The update for b is\n", dL_db)
    

        # Update parameters
        w = w - (learning_rate * dL_dw.T)
        b = b - (learning_rate * dL_db)
        print("The new w is\n", w)
        print("The new b is\n", b)
        
        # Calculate and store the loss
        loss = LCE(y, y_hat)
        losses.append(loss)
        print("The loss for this epoch is\n", loss)

    return w, b, losses, y_hat


# performing gradient descent to optimize parameters
w, b, losses, y_hat = gradient_descent(X, y, w, b, learning_rate, epochs)

#plotting the loss over epochs
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Reduction Over Epochs')
plt.show()

#getting the predicted labels
print("Y_hat:\n",y_hat)
y_hat_labels = (y_hat >= 0.5).astype(int)
print("y_hat_labels:\n",y_hat_labels)

#creating confusion matrix
cm = confusion_matrix(y, y_hat_labels)
print("confusion matrix:",cm)

ax= plt.subplot()
sns.heatmap(cm,fmt='g', ax=ax, cmap='Blues')  
# labels, title and ticks
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Declined", "Approved"])
ax.yaxis.set_ticklabels(["Declined", "Approved"])
# displaying all the confusion matrix values within the cells
for i in range(len(cm)):
    for j in range(len(cm[i])):
        ax.text(j + 0.5, i + 0.5, str(cm[i][j]), va='center', ha='center', fontsize=12)
        
plt.show()