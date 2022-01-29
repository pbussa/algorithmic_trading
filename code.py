import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from AlgoTradingTD3_solver import SolveObstacleProblem

def generatePaths(noOfPaths, noOfSteps, T, sigma, xi_0, theta, kappa):        
    # Fixing random seed
    np.random.seed(1)
        
    Z = np.random.normal(0.0,1.0,[noOfSteps])
    X = np.zeros([noOfSteps+1])
 
    time = np.zeros([noOfSteps+1])
    
    X[0] = xi_0
    
    dt = T / float(noOfSteps) #delta_t

    for i in range(0,noOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if noOfPaths > 1:
            Z[i] = (Z[i] - np.mean(Z[i])) / np.std(Z[i])
            
        X[i+1] = np.exp(-kappa * dt)*X[i] + theta * (1 - np.exp(-kappa * dt)) + sigma * np.power((1 - np.exp(-kappa * dt))/(2*kappa), 0.5)*Z[i]
        time[i+1] = time[i] +dt

 
    paths = {"time":time,"X":X}
    return paths

def modelCalibration(x, y):
    #linear regression on xi_k+1, xi_k 
    model = LinearRegression().fit(x, y)
    a = model.coef_
    b = model.intercept_

    kappa_implied = -np.log(a)
    theta_implied = b / (1 - np.exp(-kappa_implied))

    error = y - a*x - b
    sigma_implied = np.std(error) / np.power((1 - np.exp(-2*kappa_implied))/(2*kappa_implied), 0.5)
    return(kappa_implied, theta_implied, sigma_implied)


def simulateStrategy(entryValue, exitValue, T, X):
    c = 0.01
    pos = 0
    cash = 0
    cumpnl = 0
    numberTrades = 0
    for t in range(1, T+1):
        if pos == 0 and X[t] <= entryValue:
            numberTrades += 1
            pos = 1
            cash = cash - X[t] - c
        elif pos == 1 and X[t] >= exitValue:
            pos = 0
            cash = cash + X[t] - c
        else:
            pos = pos
            cash = cash
        cumpnl = cash + pos*X[t] 
    average_pnl = cumpnl/numberTrades
    return average_pnl



def main():
    #Question 2 
    noOfPaths = 1
    noOfSteps = 1000
    T = 1000
    kappa = 0.5
    sigma = 0.5
    xi_0 = 0
    theta = 1

    paths = generatePaths(noOfPaths, noOfSteps, T, sigma, xi_0, theta, kappa)
    timeGrid = paths["time"]
    X = paths["X"]

    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
  

    #Question 3
    print("Third question")
    x = X[0:999].reshape(-1, 1)
    y = X[1:1000]
    [kappa_implied, theta_implied, sigma_implied] = modelCalibration(x,y)
    print(kappa_implied, theta_implied, sigma_implied)


    #Question 4
    print("Fourth question")
    data = pd.read_csv("simdata.csv", sep = ",", index_col=False)
    x = data['xi_t'][:-1].values
    y = data['xi_t'][1:].values
    
    x = x.reshape(-1,1)
    [kappa_implied, theta_implied, sigma_implied] = modelCalibration(x,y)
    print(kappa_implied, theta_implied, sigma_implied)

    #Question 5
    #function simulateStrategy

    #Question 6
    print("Sixth question")
    noOfPaths = 1
    noOfSteps = 1000000
    T = 1000000
    kappa = 0.5
    sigma = 0.5
    xi_0 = 0
    theta = 1

    X = generatePaths(noOfPaths, noOfSteps, T, sigma, xi_0, theta, kappa)["X"]

    entry = [-0.4, 0, 0.4]
    exit = [0.5, 1, 1.5, 2]
    result = []
    for i in range(3):
        for j in range(4):
            avg = simulateStrategy(entry[i], exit[j], T, X)
            result.append(avg)
            print(avg, entry[i], exit[j])
            if avg >= max(result):
                best = avg
                en = entry[i]
                ex = exit[j]
    print('best', best, 'entry', en, 'exit', ex)


    #Question 7
    print("Seventh question")
    xi_min = -2
    xi_max = 2
    n = 1000
    kappa = 0.5
    c = 0.01
    rho = 0.01
    sigma = 0.5
    theta = 1

    xi = np.array([])

    for k in range (n):
        xi_k = xi_min + ((k)/(n+1)) * (xi_max - xi_min)
        xi = np.append(xi, xi_k)
    phi_H_pos = xi - c*(np.ones(xi.size))  
    phi_H_neg = -xi - c*(np.ones(xi.size))  

    H_pos = SolveObstacleProblem(kappa, rho, sigma, theta, xi, phi_H_pos)
    H_neg = SolveObstacleProblem(kappa, rho, sigma, theta, xi, phi_H_neg)
    
    phi_G = np.array([])
    for i in range(H_pos.size):
        m = max(H_neg[i] + xi[i] - c, H_pos[i] - xi[i] - c)
        phi_G = np.append(phi_G, m) 

    G = SolveObstacleProblem(kappa, rho, sigma, theta, xi, phi_G)
    
    epsilon = 10**(-10)
    for i in range(G.size):
        if G[i] > phi_G[i] + epsilon:
            xi_k0 = xi[i]
            break
    for i in range(G.size-1, 1, -1):
        if G[i] > phi_G[i] + epsilon:
            xi_k1 = xi[i]
            break
    
    print('Best entry', xi_k0)
    print('Best exit', xi_k1)

main()

