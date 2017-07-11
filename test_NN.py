
import base
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random

'''
Simple true/false test
'''
## initializing a blank network exmamples
testNet = base.NN(2,2,[5,5], func='sigmoid', weight=5, dropout=1.0)
testNet3 = base.NN(2,2,[128,128], func='relu', weight=5, dropout=0.8)

print("number of nodes : ", testNet.numX, testNet.numW, testNet.numY)


X = [1,0]
Y = [0,1]

## do forward calculation on network
print("Before train (sigmoid): ",testNet.forward(X))
print("Before train (ReLu)   : ",testNet3.forward(X))

# print("weights : ", testNet.W)

for i in range(1000):
	testNet.train(X,Y,0.1)
	testNet3.train(X,Y,0.1)


print("After train (sigmoid) : ",testNet.forward(X))
print("After train (ReLu)    : ",testNet3.forward(X))


'''
Test case for function approximator
'''
DO = False
GRAPHICS = True
if(DO==True):
	sinNet = base.NN(1,1,[128,256,128], func='lrelu', dropout=0.8, weight='xavier')

	X = np.arange(-2*math.pi,2*math.pi, 0.01)
	# Y = np.sin(X)
	Y = np.sin(X)

	num_iter = 2000
	num_epoch = 5
	sample_x = np.zeros(num_iter)
	sample_y = np.zeros(num_iter)
	
	# print(sinNet.W)
	for i in range(num_iter):
		x = np.random.rand()*math.pi*6.0-3*math.pi
		# y = np.sin(x)
		y = np.sin(x) + 0.1 * (2*random.random()-1)
		if (x!=0 and y!=0):
			for k in range(num_epoch):
				sinNet.train([x],[y], 0.02)
			sample_x[i] = x
			sample_y[i] = y

		if((i==1 or i==num_iter-1) and GRAPHICS):
			Yhat = np.zeros(Y.shape)
			for j, x in enumerate(X):
				Yhat[j] = sinNet.forward(x)
			f = plt.figure(1)
			ax = f.gca()
			f.show()
			ax.cla()
			# ax.plot(sample_x, sample_y, 'o', X, Y, X, Yhat)
			ax.plot(X, Y, X, Yhat)
			# ax.axis([-3.5, 3.5, -1.2, 1.2])
			# ax.axis([-3.s5, 3.5, -0.2, 3.5])
			f.canvas.draw()
			f.savefig(str(sinNet.func)+str(i)+"iter_weight"+str(sinNet.weight)+"_drop"+str(sinNet.dropout)+".png")

			# print(sinNet.W)
		# time.sleep(0.01)


	# f.savefig(str(num_iter)+" iterations.png")
