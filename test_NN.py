
import base
import numpy as np
import math
import matplotlib.pyplot as plt
import time

'''
Simple true/false test
'''
## initializing a blank network exmamples
testNet = base.NN(func='sigmoid')
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
DO = True
if(DO==True):
<<<<<<< HEAD
	sinNet = base.NN(1,1,[128], func='relu')
=======
	sinNet = base.NN(1,1,[128,128], func='lrelu')
>>>>>>> 632962e45a3c98924711682ff0b1ad42ee8cd691

	X = np.arange(-math.pi,math.pi, 0.2)
	# Y = np.sin(X)
	Y = abs(X)

<<<<<<< HEAD
	num_iter = 500
=======
	num_iter = 200
>>>>>>> 632962e45a3c98924711682ff0b1ad42ee8cd691
	num_epoch = 3
	sample_x = np.zeros(num_iter)
	sample_y = np.zeros(num_iter)
	
	# print(sinNet.W)
	for i in range(num_iter):
		x = np.random.rand()*math.pi*2.0-math.pi
		# y = np.sin(x)
		y = abs(x)
		if (x!=0 and y!=0):
			for k in range(num_epoch):
				sinNet.train([x],[y], 0.2)
			sample_x[i] = x
			sample_y[i] = y

		Yhat = np.zeros(Y.shape)
		for j, x in enumerate(X):
			Yhat[j] = sinNet.forward(x)


		if(i==1 or i==num_iter-1):
			f = plt.figure(1)
			ax = f.gca()
			f.show()
			ax.cla()
			# ax.plot(sample_x, sample_y, 'o', X, Y, X, Yhat)
			ax.plot(X, Y, X, Yhat)
			# ax.axis([-3.5, 3.5, -1.2, 1.2])
			ax.axis([-3.5, 3.5, -0.2, 3.5])
			f.canvas.draw()
			f.savefig(str(i)+" iterations.png")

			print(sinNet.W)
		time.sleep(0.01)
	# f.savefig(str(num_iter)+" iterations.png")
