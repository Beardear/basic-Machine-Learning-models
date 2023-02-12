#!/usr/bin/python3
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sympy import N
import math
import time



def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # assign the proper value to binary_error:
    
    x_full = np.insert(X, 0, 1, axis = 1)
    # y_hat = np.sign(np.matmul(w, np.transpose(x_full)))
    y_hat = 1 / (1+np.power(math.e, -np.matmul(w, np.transpose(x_full))))
    y_hat[y_hat>0.5]=1
    y_hat[y_hat<0.5]=-1
    
    binary_error = np.sum(y_hat!=y)/len(y)
    
    
    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # assign the proper values to t, w, and e_in:
    w = w_init
    x_full = np.insert(X, 0, 1, axis = 1)
    grad = 0
    e_in = 0
    for t in range(max_its):
        # for i in range(len(y)):
            # grad += -(1/len(y)) * (y[i]*x_full[i,:])/(1+np.power(math.e, y[i] * np.matmul(w, np.transpose(x_full[i,:]))))
        grad = -(1/len(y)) * (y[:,np.newaxis]*x_full) / (1 + np.power(math.e, y * np.matmul(w, np.transpose(x_full))))[:,np.newaxis]
        grad = np.sum(grad, axis = 0)
            
        w = w - eta * grad
        if(np.all(np.abs(grad)<grad_threshold)):
            print("---converge at iteration step %d, eta is %s, max_iter is %d---" % (t, eta, max_its))
            break
            
        else:
            grad = 0
        if(t%200000==0):
            print(t)
    
    for i in range(len(y)):
        e_in += (1/len(y)) * np.log(1+np.power(math.e, -y[i] * np.matmul(w, np.transpose(x_full[i,:]))))
    
    return t, w, e_in

def normalize_features(data):
    mean = np.mean(data, axis = 0)
    std = np.sqrt(np.var(data, axis = 0))
    normalized_data = (data - mean)/std
    
    return normalized_data
    
    

def main():
    Normalize = True
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')
    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')
    
    # Your code here
    train_data['heartdisease::category|0|1'][train_data['heartdisease::category|0|1']==0]=-1
    test_data['heartdisease::category|0|1'][test_data['heartdisease::category|0|1']==0]=-1
    
        
    w_init = np.zeros(14)
    w_init[0] = 0
    
    train = train_data.to_numpy()
    test = test_data.to_numpy()
    # train = train_data.values
    # test = test_data.values
    
    x_train = train[:,:-1]
    y_train = train[:,-1]
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    
    if (Normalize):
        mean = np.mean(x_train, axis = 0)
        std = np.sqrt(np.var(x_train, axis = 0))
        x_train = (x_train - mean)/std
        x_test = (x_test - mean)/std
        print('data normalized.')
        
        max_its = [int(1e6)]
        grad_threshold = 1e-6
        # eta = [0.01, 0.1, 1, 4, 7, 7.5, 7.6, 7.7]
        eta = 7.7
        
    else:
        max_its = [int(1e4), int(1e5), int(1e6)]
        grad_threshold = 1e-3
        eta = [1e-5]
        
    
    
    # cnt = 0
    t = []
    e_in = []
    binary_error = []
    binary_error_test = []
    w = []
    
    for iters in max_its:
        # ----------------------------------------------
        # ---------------------training-----------------
        # ----------------------------------------------
        start_time = time.time()
        
        t_temp,w_temp,e_in_temp = logistic_reg(x_train, y_train, w_init, iters, eta, grad_threshold)
        print("--- %s seconds when training %d iterations ---" % (time.time() - start_time, iters+1))
        t.append(t_temp)
        w.append(w_temp)
        e_in.append(e_in_temp)
        binary_error.append(find_binary_error(w_temp, x_train, y_train))
        print(f'in-sample error is {e_in}')
        print(f'training binary error is {binary_error}')
        # ----------------------------------------------
        # ---------------------testing------------------
        # ----------------------------------------------
        binary_error_test.append(find_binary_error(w_temp, x_test, y_test))
        print(f'testing binary error is {binary_error_test}')
        
    
    
    

if __name__ == "__main__":
    # main()
    a = input().split(' ')
    print(a)