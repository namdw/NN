
import base
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import copy

'''
Simple true/false test
'''
## initializing a blank network exmamples
testNet = base.NN(2,2,[5,5], func='sigmoid', weight=5, dropout=1.0)
testNet3 = base.NN(2,2,[128,128], func='relu', weight=5, dropout=0.8)

layer1 = base.Layer('input',2)
layer2 = base.Layer('hidden',5, func='lrelu', weigth='xavier', dropout=0.8)
layer3 = base.Layer('hidden',5, func='sigmoid', weight='xavier', dropout=0.8)
layer4 = base.Layer('output',2, dropout=0.8, weightt='xavier')

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
plt.ion()
DO = True
GRAPHICS = True
if(DO==True):
	input_layer = base.Layer('input',1)
	hidden_layer1 = base.Layer('hidden',128,func='lrelu',dropout=0.8,weight='xavier',weight_scale=2.0,optimizer='Vanilla')
	hidden_layer2 = base.Layer('hidden',256,func='lrelu',dropout=0.8,weight='xavier',weight_scale=2.0,optimizer='Vanilla')
	hidden_layer3 = base.Layer('hidden',128,func='lrelu',dropout=0.8,weight='xavier',weight_scale=2.0,optimizer='Vanilla')
	output_layer = base.Layer('output',1,dropout=0.8,weight='xavier',weight_scale=2.0,optimizer='Vanilla')
	sinNet = base.NNb()
	sinNet.addLayer(input_layer)
	sinNet.addLayer(hidden_layer1)
	sinNet.addLayer(hidden_layer2)
	sinNet.addLayer(hidden_layer3)
	sinNet.addLayer(output_layer)
	# sinNet2 = base.NN(1,1,[128,256,128], func='lrelu', dropout=0.8, weight='xavier')
	sinNet2 = copy.deepcopy(sinNet)
	for layer in sinNet2.layers:
		layer.optimizer = 'RMSprop'

	# for layer in sinNet2.layers:
	# 	print(layer.func, layer.optimizer)
	# print(sinNet2.func, sinNet2.dropout, sinNet2.weight)

	X = np.arange(-math.pi/2,math.pi/2, 0.01)
	# Y = np.sin(X)
	Y = np.sin(X*4)

	num_iter = 10000
	num_epoch = 3
	sample_x = np.zeros(num_iter)
	sample_y = np.zeros(num_iter)
	
	f = plt.figure(1)
	ax = f.gca()
	f.show()
	ax.plot(X, Y)
	line1 = ''
	line2 = ''
	# print(sinNet.W)
	for i in range(num_iter):
		x = np.random.rand()*math.pi-math.pi/2
		y = np.sin(x*4) + 0.05 * (2*random.random()-1)
		if (x!=0 and y!=0):
			for k in range(num_epoch):
				sinNet.train([x],[y], 0.001)
				sinNet2.train([x],[y], 0.001)
			sample_x[i] = x
			sample_y[i] = y

		#if(((i+1)%1000==0 or i==0) and GRAPHICS):
		## if(GRAPHICS):
			Yhat = np.zeros(Y.shape)
			Yhat2 = np.zeros(Y.shape)
			for j, x in enumerate(X):
				Yhat[j] = sinNet.forward(x)
				Yhat2[j] = sinNet2.forward(x)
			# ax.cla()
			# ax.plot(sample_x, sample_y, 'o', X, Y, X, Yhat)
			if(line1!='' and line2!=''):
				line1.remove()
				line2.remove()
			line1, = ax.plot(X, Yhat, 'tab:orange')
			line2, = ax.plot(X, Yhat2, 'tab:green')
			ax.plot(sample_x[i], sample_y[i], 'ob')
			ax.axis([-2.0, 2.0, -1.5, 1.5])
			plt.title(str(i))
			f.canvas.draw()
			#f.savefig("buildnet"+str(i)+"iter.png")


	# f.savefig(str(num_iter)+" iterations.png")
	print(sinNet.layers[1].m)