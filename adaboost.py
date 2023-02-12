#!/usr/bin/python3
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def compute_error(pred, labels):
    return np.mean(pred != labels)


def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use
    N, d = X_train.shape #N = 1663, d= 256
    N_test, _ = X_test.shape #(430, 256)
    
    D = np.zeros((N, n_trees)) #each column is a D_t, each element in D_t is a D_t(n)
    D[:,0] = 1/N #initialize D_1
    training_error = np.zeros(n_trees)
    test_error = np.zeros(n_trees)
    alpha = np.zeros(n_trees)
    clfs = [] #use a list to store decision stumps
    predict_training_all = 0
    predict_test_all = 0
    for t in range(n_trees):
        # train
        clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1)
        clf.fit(X_train, y_train, D[:, t])
        clfs.append(clf)
        
        epsilon = 1 - clf.score(X_train, y_train, sample_weight = D[:, t])
        alpha[t] = 1/2 * np.log((1-epsilon)/epsilon)
        # predict
        predict_training = clf.predict(X_train)
        predict_test = clf.predict(X_test)
        predict_training_all += alpha[t] * predict_training
        predict_test_all += alpha[t] * predict_test
        output_training = np.sign(predict_training_all)
        output_test = np.sign(predict_test_all)
        training_error[t] = compute_error(output_training, y_train)
        test_error[t] = compute_error(output_test, y_test)
        
        print("#trees=", t + 1, "E_train=", training_error[t], "E_test=", test_error[t])
        if t < (n_trees -1) :
            gamma = np.sqrt((1-epsilon)/epsilon)
            Z = gamma * epsilon + 1/gamma * (1-epsilon)
            D[:,t+1] = 1/Z * D[:,t] * np.exp(-alpha[t] * y_train * predict_training)
            
    
    # train_error = 0
    # test_error = 0

    return training_error, test_error


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees = 200
    problem = "3vs5" # 1vs3 or 3vs5
    
    
    if problem == "1vs3":
        # Split data
        X_train = og_train_data[np.logical_or(og_train_data[:,0] == 1, og_train_data[:,0]==3)][:,1:]
        y_train = og_train_data[np.logical_or(og_train_data[:,0] == 1, og_train_data[:,0]==3)][:,0]
        X_test = og_test_data[np.logical_or(og_test_data[:,0] == 1, og_test_data[:,0]==3)][:,1:]
        y_test = og_test_data[np.logical_or(og_test_data[:,0] == 1, og_test_data[:,0]==3)][:,0]
    elif problem =="3vs5":
        X_train = og_train_data[np.logical_or(og_train_data[:,0] == 3, og_train_data[:,0]==5)][:,1:]
        y_train = og_train_data[np.logical_or(og_train_data[:,0] == 3, og_train_data[:,0]==5)][:,0]
        X_test = og_test_data[np.logical_or(og_test_data[:,0] == 3, og_test_data[:,0]==5)][:,1:]
        y_test = og_test_data[np.logical_or(og_test_data[:,0] == 3, og_test_data[:,0]==5)][:,0]

    if problem == "1vs3":
        y_train -= 2
        y_test -= 2
    elif problem == "3vs5":
        y_train -= 4
        y_test -= 4
        
    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)
    
    # plot results
    if problem == "1vs3":
        plt.title("1 versus 3")
    elif problem == "3vs5":
        plt.title("3 versus 5")
    
    x = [1,2,3,4,5,6,7,8,9,10]
    loss = [890.9874, 129.7339, 70.5128, 53.4660, 46.4309, 43.1683, 40.4465, 39.0341, 37.7531, 36.4761]
    x = list(range(num_trees))
    plt.plot(x, train_error, label = 'train error')
    plt.plot(x, test_error, label = 'test error')
    plt.xlabel('num of trees')
    plt.ylabel('error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main_hw5()
