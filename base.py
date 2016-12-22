
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
		self.numH = len(self.numW) # number of hidden layers

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
		self.B = [0] * (len(self.numW)+1)
		for i in range(len(self.B)):
			if (i==len(self.B)-1):
				self.B[i] = np.random.rand(1,self.numY)
			else:
				self.B[i] = np.random.rand(1,self.numW[i])
		self.Y = np.zeros([self.numY,1])

	'''
	Sigmoid function for an array
	X(float[]) : array of values to calculate sigmoid
	'''	
	def sigmoid(self, X):
		result = np.zeros(X.shape)
		for i in range(len(X[0])):
			result[0][i] = 1 / (1+math.exp(-1*X[0][i])) # sigmoid function 
		return result

	'''
	NN forward propagation
	X(float[]) : array of inputs to calculate the forward propagation
	'''
	def forward(self, X):
		a = np.array([X])
		for i in range(len(self.W)):
			z = np.dot(a, self.W[i]) / np.prod(a.shape) + self.B[i] # z = a*w + b
			a = self.sigmoid(z) # a = sigma(a*w)
		return a

	def train(self, X, Y):
		X = np.array([X])
		Y = np.array([Y])
		n = 1.0
		A = [0]*(len(self.W)+1)
		Z = [0]*len(self.W)
		A[0] = X
		for i in range(len(self.W)):
			Z[i] = np.dot(A[i], self.W[i]) / np.prod(A[i].shape) + self.B[i] # z = a*w + b
			A[i+1] = self.sigmoid(Z[i]) # a = sigma(a*w)
		D = [0]*(len(self.W)+1)
		D[0] = (A[-1]-Y) / np.prod(A[-1].shape) * A[-1]*(1-A[-1])
		# D[0] = 0.5 * (A[-1]-Y)**2 / np.prod(A[-1].shape) * A[-1]*(1-A[-1])

		for i in range(len(self.W)):
			D[i+1] = np.transpose(np.dot(self.W[len(self.W)-i-1], np.transpose(D[i]))) * A[-2-i]*(1-A[-2-i])
			self.W[-1-i] = self.W[-1-i] - n * np.dot(np.transpose(A[-2-i]), D[i])
			self.B[-1-i] = self.B[-1-i] - n * D[i]

		# print(D)

			
