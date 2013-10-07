import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as slope

#problem 6

def XOR(D, X):
    for x in X:
        if x in D:
            print x + ':' + str(D[x])
        else:
            numOnes = x.count('1')
            if numOnes % 2 == 1:
                D[x] = 1
            else:
                D[x] = 0
    return D



D = {'000':0,'001': 1, '010': 1, '011': 0, '100': 1}
X = ['000', '001', '010', '011', '100', '101', '110', '111']
##newD = problem5h1(D,X)
##print newD
##def problem7(point1, point2, points):
##    ansArray = []
##    deltaX = point1['x'] - point2['x']
##    deltaY = point1['y'] - point2['y']
##    xArray = linspace(point1['x'], point2['x'], 10)
##    if deltaX > 0 and deltaY > 0:
##        for i in range(0,9):
##            randP = {'x':random.uniform(-1,1),y:random.uniform(-1,1)}
##            if randP['x'] > point1['x']

# code for perceptron for NAND function
# taken from wikipedia
##threshold = 0.5
##learning_rate = 0.1
##weights = [0, 0, 0]
##training_set = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]
## 
##def dot_product(values, weights):
##    return sum(value * weight for value, weight in zip(values, weights))
## 
##while True:
##    print('-' * 60)
##    error_count = 0
##    for input_vector, desired_output in training_set:
##        print(weights)
##        result = dot_product(input_vector, weights) > threshold
##        error = desired_output - result
##        if error != 0:
##            error_count += 1
##            for index, value in enumerate(input_vector):
##                weights[index] += learning_rate * error * value
##    if error_count == 0:
##        break

# weights - a numpy array of weights values, should only have two indices
# learnRate - a float containing learn rate constant
# thresh - threshold for perceptron
# xyValues - list with tuples, x in first tuple, y in second tuple
# D - list of expected values, that are ints
# dIndex - int 
def perceptron(weights, learnRate, thresh, xyValues, D, dIndex):
    for xVal, yVal in xyValues
        xy = np.array([xVal, yVal])
        y = np.dot(weights.transpose(), xVals)
        n = 0
        if y > D[dIndex]:
            n = 1
        else:
            n = -1
        error = D[dIndex] - n
        correction = learnRate * error
        Ds = correction* np.ones((1, 2))
        if (1 / y) * sum(Ds - xy) < thresh:
            break
        else 
            return perception(weights, learnRate, thresh, xyValues, D, dIndex + 1)
        
# initialize values
weights = np.zeros((1,2))

thresh = 0.5
learnRate = 0.1
points = []

# create random point to determine sign
x1 = random.uniform(-1,1)
y1 = random.uniform(-1,1)
x2 = random.uniform(-1,1)
y2 = random.uniform(-1,1)
point1 = {'x':x1,'y':y1}
point2 = {'x':x2,'y':y2}

# get inputs for training data
for i in range(0,9):
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    points.append((x,y))

# can't find d?
# create d from plot
xVal = np.linspace(point1['x'], point2['x'], 10)
yVal = np.linspace(point1['y'], point2['y'], 10)
fit = np.polyfit(xVal, yVal, 1)
plt.plot(xVal,fit)
for randX, randY in points:
    plt.plot(randX, randY, 'ro')
plt.axis([-1,1,-1,1])
plt.show()

