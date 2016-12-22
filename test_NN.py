
import base
import numpy as np
import math

## initializing a blank network
testNet = base.NN()

print(testNet.numX, testNet.numW, testNet.numY)

## do forward calculation on network
print(testNet.forward([1,1]))

# X = [1,1]
# Y = [1,1]

# for i in range(10):
# 	testNet.train(X,Y)
	
# print(testNet.forward([1,1]))


