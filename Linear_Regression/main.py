# This code demo simple fuction: f(x) = ax + b
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def computeGradient(X,y,theta):
    Grad =  X.T.dot(X.dot(theta) - y) / X.shape[0]
    return Grad

def updateWeights(w, Grad, eta):
    w = w - eta * Grad
    return w

def computeloss(X, y, theta):
    loss = np.linalg.norm(X.dot(theta)-y) ** 2  / X.shape[0] * 0.5
    return loss

def isConverged(w1, w2, eps):
    return np.linalg.norm(w1-w2) < eps

def BGD(xy, theta_init, eta, eps, maxIter):
    theta_init = np.array([theta_init]).T
    thetas = [theta_init]
    GDs = []
    losses = []
    X = np.array([xy[0]]).T
    X = np.concatenate((np.ones((1, X.shape[0])).T, X), axis=1)
    y = np.array([xy[1]]).T
    
    for i in range(maxIter):
      GDs.append(computeGradient(X, y, thetas[-1]))
      losses.append(computeloss(X, y, thetas[-1]))
      new_thetas = updateWeights(thetas[-1], GDs[-1], eta)
      if isConverged(thetas[-1], new_thetas, eps):
        break
      thetas.append(updateWeights(thetas[-1], GDs[-1], eta))

    return np.array(thetas), np.array(GDs), np.array(losses)

def animate(i):
    x_gd = np.array(x_line)
    y_gd = [BGDWeights[i][0][0] + BGDWeights[i][1][0] * j for j in x_line]
    line.set_data(x_gd, y_gd)
    return line,

fig, ax = plt.subplots()
ax.set_title('Gradient Descent for Linear Regression')
x = [1, 2, 3, 4, 6, 10, 30, 44, 12, 21, 9, 15]
y = [2, 4, 5, 9, 12, 25, 55, 92, 30, 35, 21, 20]

initialWeights = [0, 0]
stepSize = 0.001
convergenceTol = 1e-3
numIterations = 100
data = [x, y]

# Fit the linear regression model
BGDWeights, BGDs, BGDlosses = BGD(data, initialWeights, stepSize, convergenceTol, numIterations)
interceptBGD, slopeBGD = BGDWeights[-1][0][0], BGDWeights[-1][1][0] # interceptBGD, slopeBGD stand for b, a in function f(x) = ax + b

# Draw initial points and initial line
plt.plot(x, y, 'ro', label='Initial points')
plt.plot([-10, 50], [initialWeights[0] + initialWeights[1] * i for i in initialWeights], 'r', label='Initial line')

# Visualize the weight update affect the line
for weight in BGDWeights:
    x = [-10, 50]
    y = [weight[0][0] + weight[1][0] * i for i in x]
    plt.plot(x, y, 'k', alpha=0.5)

# Draw the best line 
x_line = [-10, 50]
y_line = [interceptBGD + slopeBGD * i for i in x_line]
plt.plot(x_line, y_line, label='Regression line')

# Test prediction
x_predicts = [-7, 36, 25]
y_predicts = [interceptBGD + slopeBGD * i for i in x_predicts]
plt.plot(x_predicts, y_predicts, 'k^', label='Prediction points')

# Draw animation
line,   = ax.plot([],[], 'b')
iters = np.arange(1,len(BGDWeights), 1)
line_animation = FuncAnimation(fig, animate, iters, interval=100, blit=True)

plt.legend()
plt.show()