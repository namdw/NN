
import base
import numpy as np
import math
import matplotlib.pyplot as plt

'''
Simple true/false test
'''
## initializing a blank network exmamples
testNet = base.NN()
testNet2 = base.NN(2,2,[4,4])
testNet3 = base.NN(func='relu')

print("number of nodes : ", testNet.numX, testNet.numW, testNet.numY)


X = [1,1]
Y = [0,1]

## do forward calculation on network
print("Before train (sigmoid): ",testNet.forward(X))
print("Before train (ReLu)   : ",testNet3.forward(X))

# print("weights : ", testNet.W)

for i in range(10):
	testNet.train(X,Y)
	testNet3.train(X,Y)


print("After train (sigmoid) : ",testNet.forward(X))
print("After train (ReLu)    : ",testNet3.forward(X))


'''
Test case for function estimator
'''
sinNet = base.NN(1,1,[5,5])

X = np.arange(0,2*math.pi, 0.1)
Y = np.cos(X)

for i in range(50):
	for j in range(len(X)):
		sinNet.train(X[j], Y[j])

Yhat = np.zeros(Y.shape)
for i, x in enumerate(X):
	Yhat[i] = sinNet.forward(x)

plt.figure(1)
plt.plot(X,Y)

plt.figure(2)
plt.plot(X,Yhat)

plt.show()