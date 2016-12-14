
import numpy as np 
import math

class NN():
	'''
	Create a Custom Neural Network
	input:
	numX, numW, numY
	numX(int) : number of inputs nodes of the network. Default 2
	numW([int]) : list of number of weigth for each hidden layer. [numW1, numW2,...,numWH]. Default [5]
	numY(int) : number of outputs to the network. Default 2
	'''
	def __init__(self, *args):
		## Default values
		self.numX = 2
		self.numW = [5]
		self.numY = 2

		## Initialize input variables
		if (len(args)>0):
			self.numX = args[0]
			if(len(args)>1):
				self.numY = args[1]
				if(len(args)>2):
					self.numW = args[2]
		self.numH = len(self.numW)-1 # number of hidden layers

		## Initialize input, weigths, and outputs of the network
		self.X = np.zeros([1, self.numX])
		self.W = [0] * (len(self.numW)+1)
		for i in range(len(self.W)):
			if (i==0):
				self.W[i] = np.random.rand(self.numX, self.numW[i])
			elif (i==len(self.W)-1):
				self.W[i] = np.random.rand(self.numW[i-1], self.numY)
			else:
				self.W[i] = np.random.rand(self.numW[i-1], self.numW[i])
		self.Y = np.zeros([self.numY,1])

	'''
	Sigmoid function for an array
	X(float[]) : array of values to calculate sigmoid
	'''	
	def sigmoid(self, X):
		result = np.zeros(X.size)
		for i in range(len(X)):
			result[i] = 1 / (1+math.exp(-1*X[i]))
		return result

	'''
	NN forward propagation
	X(float[]) : array of inputs to calculate the forward propagation
	'''
	def forward(self, X):
		node_in = X
		for i in range(len(self.W)):
			node_in = self.sigmoid(np.dot(node_in, self.W[i]))
		return node_in

	def train(self, X, Y, numEpoch):
		for i in range(numEpoch):
			# do training...
			pass