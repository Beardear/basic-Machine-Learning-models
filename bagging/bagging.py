#!/usr/bin/python3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier

def databagging(X_train,y_train,n_sample):
    sample = []
    indexs = []
    labels = []
    for i in range(n_sample):
        index = randrange(n_sample)
        sample.append(X_train[index])
        indexs.append(index)
        
    sample = np.array(sample)
    labels = [y_train[i] for i in indexs]
    labels = np.array(labels)
    
    # Out of Bag Sample
    # Data Pieces selected into training (no duplication)
    indexs = list(set(indexs))
    
    ref_list = range(n_sample)
    oob_list = list(set(ref_list) - set(indexs))
    
    X_train_copy = deepcopy(X_train)
    y_train_copy = deepcopy(y_train)
    oob_sample = np.delete(X_train_copy, indexs,axis = 0)
    oob_label = np.delete(y_train_copy,indexs)
    
    return sample,labels,oob_sample,oob_label,oob_list
    
    
def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: Here I use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function
    
    # DataBagging

    n_sample = len(y_train)
    trees = [] 
    index_list = []
    
    # Out-of-Bag Error List
    err_array = []
    [err_array.append([]) for i in range(n_sample)]                 
    error_count = 0
    
    ### It is important to maintain a 2D-array for sample status in each row
    ### Assume a 2D array with size of n_sample x 2
    ### The first column maintains the error count
    ### The second column maintains the total count
    ### Each Row --> A piece of training data
    
    for i in range(num_bags):
        
        trainingdata, traininglabels, oob_data, oob_label, indexs = databagging(X_train, y_train, n_sample)
        
        tree = DecisionTreeClassifier(criterion='entropy')
        tree = tree.fit(trainingdata,traininglabels)
        
        trees.append(tree)
        
        
        # Out-of-bag Error Processing
        o_predict = tree.predict(oob_data)
        
        # Loop Pointer Definition
        i = 0
        
        for label in indexs:
           
            err_array[label].append(o_predict[i])
            
            i += 1
            
            '''
            # Total Count
            err_array[label,1] += 1
            
            # Error Count
            if o_predict[i] != oob_label[i]:
                err_array[label,0] += 1
            
            i += 1
            '''
        
        index_list = index_list + indexs
        index_list = list(set(index_list))
    
    
    # OOB_Error based on Err_array
    #sample_error_sum = np.array([err_array[index,0] / err_array[index,1]  for index in index_list]).sum()
    estimated_ooblabel = [max(err_array[index],key = err_array[index].count) for index in index_list]
    
    j = 0
    for index in index_list:
        if estimated_ooblabel[j] != y_train[index]:
            error_count += 1
        j += 1
    
    
    #sample_error_sum = np.array([row[0]/row[1]  for row in err_array]).sum()
    oob_error = error_count / len(index_list)
    
    
    # Bagging DecisionTree Test Prediction
    index = 0
    error = 0
    
    for row in X_test:
  
        p_res = []
        
        for tree in trees:
            res = tree.predict(row.reshape(1,-1))
            p_res.append(res)
        
        label = max(p_res,key=p_res.count)
        
        if label != y_test[index]:
            error += 1
        
        index += 1
      
        
    # Test Error
    n_test = len(y_test)
    test_error = error / n_test
    
    
    
    print('Test_Error when Num_Bags =',num_bags,'is',test_error)
    
        
    out_of_bag_error = oob_error
    
    '''
    # Reference Test
    clf = RandomForestClassifier(n_estimators = num_bags, oob_score= True)
    clf = clf.fit(X_train,y_train)
    ref_oob = clf.oob_score_
    '''
    
    
    return out_of_bag_error, test_error


def single_decision_tree(X_train, y_train, X_test, y_test):
    single_tree = DecisionTreeClassifier(criterion='entropy')
    single_tree = single_tree.fit(X_train,y_train)
    train_error = 1 - single_tree.score(X_train,y_train)
    test_error = 1- single_tree.score(X_test,y_test)
    
    return train_error,test_error
    
    
    
    
    
def main_hw4():
    # Load data
    
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')
    
    
    # Question1: 1 versus 3
    # Extract '1' and '3' data pieces from Train/Test Set
    
    q1_train = []
    for lines in og_train_data:
        if lines[0] == 1 or lines[0] == 3:
            q1_train.append(lines)
    
    q1_test = []
    for lines in og_test_data:
        if lines[0] == 1 or lines[0] == 3:
            q1_test.append(lines)
    
    # Question2: 3 versus 5
    q2_train = []
    for lines in og_train_data:
        if lines[0] == 3 or lines[0] == 5:
            q2_train.append(lines)
    
    q2_test = []
    for lines in og_test_data:
        if lines[0] == 3 or lines[0] == 5:
            q2_test.append(lines)
    
    q1_train = np.array(q1_train)
    q1_test = np.array(q1_test)
    q2_train = np.array(q2_train)
    q2_test = np.array(q2_test)
    
    

    
    
    question = 'q2'
    
    if question == 'q1':
        
        # Split data
        y_train = q1_train[:,0]
        y_test = q1_test[:,0]
        X_train = np.delete(q1_train, 0 ,axis = 1)
        X_test = np.delete(q1_test, 0 , axis = 1)
        
        Q1_OOBE = []

        for i in range(1,201):
            # Run bagged trees
            out_of_bag_error, test_error = bagged_trees(X_train, y_train, X_test, y_test, i)
            Q1_OOBE.append(out_of_bag_error)
        
        plt.plot(Q1_OOBE)
        plt.xlabel('Num of Bags')
        plt.ylabel('OOB Error')
        plt.title('1 versus 3')
        plt.show()
        
        
        train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)
        
        print('/')
        print('Train_E of single decision tree is:',train_error)
        print('Test_E of single decision tree is:',test_error)
        
        
    else:
        
        # Split data
        y_train = q2_train[:,0]
        y_test = q2_test[:,0]
        X_train = np.delete(q2_train,0,axis = 1)
        X_test = np.delete(q2_test, 0, axis = 1)
    
        Q2_OOBE = []

        for i in range(1,201):
            # Run bagged trees
            out_of_bag_error, test_error = bagged_trees(X_train, y_train, X_test, y_test, i)
            Q2_OOBE.append(out_of_bag_error)
        
        plt.plot(Q2_OOBE)
        plt.xlabel('Num of Bags')
        plt.ylabel('OOB Error')
        plt.title('3 versus 5')
        plt.show()
        
        
        train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)
        
        print('/')
        print('Train_E of single decision tree is:',train_error)
        print('Test_E of single decision tree is:',test_error)
        
        
if __name__ == "__main__":
    main_hw4()

