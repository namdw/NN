
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

layer1 = base.Layer('input',2)
layer2 = base.Layer('hidden',5, func='lrelu', dropout=0.8)
layer3 = base.Layer('hidden',5, func='sigmoid', dropout=0.8)
layer4 = base.Layer('output',2)

buildNet = base.NNb()
buildNet.addLayer(layer1)
buildNet.addLayer(layer2)
buildNet.addLayer(layer3)
buildNet.addLayer(layer4)

print("number of nodes : ", testNet.numX, testNet.numW, testNet.numY)


X = [1,0]
Y = [0,1]

## do forward calculation on network
print("Before train (sigmoid): ",testNet.forward(X))
print("Before train (ReLu)   : ",buildNet.forward(X))

# print("weights : ", testNet.W)

for i in range(1000):
	testNet.train(X,Y,0.001)
	buildNet.train(X,Y,0.001)


print("After train (sigmoid) : ",testNet.forward(X))
print("After train (ReLu)    : ",buildNet.forward(X))


'''
Test case for function approximator
'''
DO = True
GRAPHICS = True
if(DO==True):
	input_layer = base.Layer('input',1)
	hidden_layer1 = base.Layer('hidden',128,func='lrelu',dropout=0.8,weight='xavier',weight_scale=10.0)
	hidden_layer2 = base.Layer('hidden',256,func='lrelu',dropout=0.8,weight='xavier',weight_scale=10.0)
	hidden_layer3 = base.Layer('hidden',128,func='lrelu',dropout=0.8,weight='xavier',weight_scale=10.0)
	output_layer = base.Layer('output',1)
	sinNet = base.NNb()
	sinNet.addLayer(input_layer)
	sinNet.addLayer(hidden_layer1)
	sinNet.addLayer(hidden_layer2)
	sinNet.addLayer(hidden_layer3)
	sinNet.addLayer(output_layer)
	# sinNet = base.NN(1,1,[128,256,128], func='lrelu', dropout=0.8, weight=10)

	X = np.arange(-math.pi/2,math.pi/2, 0.01)
	# Y = np.sin(X)
	Y = np.sin(X*4)

	num_iter = 2000
	num_epoch = 5
	sample_x = np.zeros(num_iter)
	sample_y = np.zeros(num_iter)
	
	# print(sinNet.W)
	for i in range(num_iter):
		x = np.random.rand()*math.pi/2-math.pi/4
		# x = (math.pi*6/100*num_iter)%(math.pi*6)-math.pi*3
		# y = np.sin(x)
		y = np.sin(x*4) + 0.1 * (2*random.random()-1)
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
			f.savefig("buildnet"+str(i)+"iter.png")

			# print(sinNet.W)
		# time.sleep(0.01)


	# f.savefig(str(num_iter)+" iterations.png")
