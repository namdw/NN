
import base
import numpy as np
import math
import matplotlib.pyplot as plt
import time

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

for i in range(100):
	testNet.train(X,Y)
	testNet3.train(X,Y)


print("After train (sigmoid) : ",testNet.forward(X))
print("After train (ReLu)    : ",testNet3.forward(X))


'''
Test case for function approximator
'''
sinNet = base.NN(1,1,[10], func='relu')

X = np.arange(0,math.pi*2.0, 0.2)
Y = (np.sin(X))/2.0+0.5

num_iter = 500
sample_x = np.zeros(num_iter)
sample_y = np.zeros(num_iter)
f = plt.figure(1)
ax = f.gca()
f.show()
# print(sinNet.W)
for i in range(num_iter):
	x = np.random.rand()*math.pi*2.0
	y = (np.sin(x))/2.0+0.5
	sinNet.train(x,y, 0.01)
	sample_x[i] = x
	sample_y[i] = y

	Yhat = np.zeros(Y.shape)
	for i, x in enumerate(X):
		Yhat[i] = sinNet.forward(x)

	ax.cla()
	ax.plot(X, Y, X, Yhat, sample_x, sample_y, 'o')
	ax.axis([0, 7.0, 0, 1.2])
	f.canvas.draw()


	time.sleep(0.01)
	# print(sinNet.W)
