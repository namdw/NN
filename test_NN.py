
import base
import numpy as np
import math

## initializing a blank network
testNet = base.NN()

print(testNet.numX, testNet.numW, testNet.numY)

## do forward calculation on network
print(testNet.forward([1,3]))

