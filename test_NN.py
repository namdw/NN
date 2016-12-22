
import base
import numpy as np
import math

## initializing a blank network
testNet = base.NN()

print("number of nodes : ", testNet.numX, testNet.numW, testNet.numY)


X = [1,1]
Y = [0,1]

## do forward calculation on network
print("Before train : ",testNet.forward(X))

# print("weights : ", testNet.W)

for i in range(1000):
	testNet.train(X,Y)


print("After train : ",testNet.forward(X))


