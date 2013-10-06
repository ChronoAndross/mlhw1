import math
import random
import numpy as np
import matplotlib.pyplot as plt

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

# initialize values
weights = []
for j in range(0,9):
    weights.append(0)

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
plt.plot(xVal,yVal)
for randX, randY in points:
    plt.plot(randX, randY, 'ro')
plt.axis([-1,1,-1,1])
plt.show()

