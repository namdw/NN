
import base
import numpy as np
import math

## initializing a blank network
testNet = base.NN()

print("number of nodes : ", testNet.numX, testNet.numW, testNet.numY)

## do forward calculation on network
print("First forward : ",testNet.forward([1,1]))

# print("weights : ", testNet.W)

X = [1,1]
Y = [1,0]

for i in range(10):
	testNet.train(X,Y)


print(testNet.forward([1,1]))


