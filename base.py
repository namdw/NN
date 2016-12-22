
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
		a = np.array([X])
		for i in range(len(self.W)):
			z = np.dot(a, self.W[i])
			print(z)
			# a = self.sigmoid(z[0])
		return a

	def train(self, X, Y):

		n = 0.5
		A = [0]*(len(self.W)+1)
		Z = [0]*len(self.W)
		A[0] = np.array([X])
		for i in range(len(self.W)):
			Z[i] = np.dot(A[i], self.W[i])
			A[i+1] = self.sigmoid(Z[i])

		D = [0]*(len(self.W)+1)
		D[0] = 1/2 * (A[-1]-Y)**2 * A[-1]*(1-A[-1])

		for i in range(len(self.W)):
			D[i+1] = np.dot(self.W[len(self.W)-i-1], D[i])
			print(A[-2-i].shape, D[i].shape)
			self.W[-1-i] = self.W[-1-i] - np.dot(A[-2-i], np.transpose(D[i]))
			
