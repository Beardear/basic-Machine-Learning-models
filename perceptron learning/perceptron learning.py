#!/usr/bin/python2.7
import numpy as np
import matplotlib.pyplot as plt



def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for
    
    # Your code here, assign the proper values to w and iterations:
    x = np.transpose(data_in[:,:-1])
    
    y_star = np.transpose(data_in[:,-1])
    w = np.zeros(11)
    # y = np.sign(np.random.uniform(-1,1, np.size(y_star)))
    iterations = 0
    
    while(np.any(y_star!=np.sign(np.matmul(w,x)))):
        for i in range(x.shape[1]):
            if y_star[i]*(np.sign(w.dot(x[:,i])))<=0:
                w = w + y_star[i]*x[:,i]
                iterations += 1
    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    # Your code here, assign the values to num_iters and bounds_minus_ni:
    for i in range(num_exp):
        print(i)
        x = np.random.uniform(-1,1,[d+1,N])
        x[0,:] = 1
        
        w_star = np.random.uniform(0,1, 11)
        w_star[0] = 0
        
        y_star = np.sign(np.matmul(w_star, x))
        
        data_in =np.concatenate((np.transpose(x),y_star[:,np.newaxis]),axis=1)
        
        [w, iterations] = perceptron_learn(data_in)
        
        rou = min(y_star*np.matmul(w_star,x))
        mode = np.zeros(N)
        for j in range(N):
            mode[j] = np.linalg.norm(x[:,j])
        R = max(mode)
        
        t_bound = R**2 *  (np.linalg.norm(w_star))**2 / rou**2
        
        num_iters[i] = iterations
        bounds_minus_ni[i] = t_bound - iterations
        
    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()