#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:00:26 2023

@author: kaushiknarasimha
"""

import numpy as np
import pandas as pd

def min_max_scaling(data):
    # Calculate the minimum and maximum values for each feature
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Perform min-max scaling for each feature
    scaled_data = (data - min_vals) / (max_vals - min_vals)

    return scaled_data


def sigmoid(X):
    lines = []
    for x in X:
        print(f"- 1/1+exp(-{np.round(x,3)})")
        line = f"1/1+exp(-{np.round(x,3)})"
        lines.append(line)
        
#     print(np.array(lines))
    print()
    print()
    return 1/(1+np.exp(-X))

def compute_forward(W,b,X):
    store_lines = []
    
#      store_lines = []
#     for x, weight in zip(X, W):
#         lines = []
#         for element in x:
#             line = f"{np.round(element,2)} x {weight} + {b}"
#             lines.append(line)
#         store_lines.append(lines)

    for x in X:
        lines = []  # Initialize lines for this x
        for weight, element in zip(W, x):
            line = f"{np.round(element, 2)} x {weight} + {b}"
            lines.append(line)

        # Append the lines to store_lines only if they haven't been added before
        if lines not in store_lines:
            store_lines.append(lines)

    # Print the stored lines
    final_lines = []
    for lines in store_lines:
        final_line = ''
        for line in lines:
            final_line += line
        final_lines.append(final_line)

#     final_lines

    print('Z=\n',end='')
    print("[",end="")
    for line in final_lines:
        print(line)
    
    print("]")
    print()
    
    z = np.dot(X,W) + b
    return z

def compute_loss(y, yhat):
    
    loss = []
    print('Loss Computing - Steps')
    for y_true, y_pred in zip(y, yhat):
        print(f"- {np.round(y_true,2)} * log({np.round(y_pred,2)}) + (1 - {np.round(y_true,2)}) * log(1 - {np.round(y_pred,2)})")
        bce = (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss.append(bce)
    print('--'*25)

    
    return loss

def compute_gradients(w, y, yhat, x):
    
        print('Computing Gradients dLCE/dw')
        print('--'*25)
        print()
        
        y = np.array(y)
        m = x.shape[0]
        yhat_y = yhat-y
        lr = 1 
        print('Learning Rate (α) = 1')
        print(f"y: {y},\nyhat: {yhat},\nyhat-y: {yhat_y}")
        
        print()
        print('dLCE/dw: 1/m ( yhat - y)T @ X') 
        print()
        print(f"1/{m} x {yhat_y} x \n{x}")
        
        dlce_dw = 1/m*(yhat_y) @ x
        print()
        print(f"dLCE/dw: {dlce_dw}")
        print()
        print(f'Updated weights (w) = w - lr * dlce/dw',end=" = ")
        print(f"{w} - {dlce_dw}")
        updated_weights = w - lr * dlce_dw
        print(f'Updated Weights = {updated_weights}')       
        return updated_weights
        
        
    


def main(x,y,w,b, mode="initial"):
    
    print()
    print('--'*25)
    if mode == "initial":
        print(f'Forward pass with Initial Weights as : {w}, bias = {b}')
    elif mode == 'optimized':
        print(f'Forward pass with Optimized Weights : {w}, bias = {b}')
    
    print('--'*25)
    z = compute_forward(w,b,x)
    print("z = ", list(z))
    print('--'*25)
    print('yhat for z')
    yhat = sigmoid(z)
    print('yhat i.e S(z):',yhat,'y:',y.values)
    print('--'*25)


    loss = compute_loss(y, yhat)
    print(f'loss, for weigts {w} and bias b {b}:\n',loss)
    print('--'*25)
    
    print('total loss = -1/m sum_i_m (individual_loss)')
    m = x.shape[0]
    print('Number of samples, m = ',m)
    print()
    total_loss = -1/m*sum(loss)
    print(f'total loss: {total_loss}')
    print('--'*25)
    print()
    
    return yhat
    
    

    
    
    


data = pd.read_csv('/Users/kaushiknarasimha/Documents/Academics/SEM3/NNs/A1_Data_Kaushik_Narasimha_Bukkapatnam.csv')

#print('original_data')
#print('--'*25)
#print(data)

# creating a mapping dictionary to change labels to binary
mapping = {'Approved': 1, 'Denied': 0}
# converting the values in the "Loan Status" column
data['Loan Status'] = data['Loan Status'].map(mapping)

# X, y  # Change it to feature columns and target label of dataset in use.
X = data[['Credit Score','Income','Age']]
y = data['Loan Status']




print('Input Matrix')
print('--'*25)
print(X.values)
print('--'*25)
print('Target Label')
print(y.values)

print('--'*25)

# Scaling MinMax
scaled_X = min_max_scaling(X.values)
scaled_X 

w = [1,1,1]
b = 0
yhat = main(scaled_X, y, w, b, mode='initial')
optimized_weights = compute_gradients(w, y, yhat, scaled_X)

yhat = main(scaled_X, y, optimized_weights, b, mode='optimized')


# w_n1 = [ 0.2, 0.1, 0.3 ]
# b_n1 = 0
# main(scaled_X, y, w_n1, b_n1)

# w_n2 = [ 0.4, 0.8, 1 ]
# b_n2 = 0

# main(scaled_X, y, w_n2, b_n2)