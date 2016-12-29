
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
	def __init__(self, *args, **kwargs):
		## Default values
		self.numX = 2
		self.numW = [5]
		self.numY = 2
		self.func = 'sigmoid'

		## Initialize input variables
		if (len(args)>0):
			self.numX = args[0]
			if(len(args)>1):
				self.numY = args[1]	
				if(len(args)>2):
					if(type(args[2])==type(1)):
						self.numW = [args[2]]
					else:
						self.numW = args[2]
		self.numH = len(self.numW) # number of hidden layers

		
		for name, value in kwargs.items():
			if (name=='func' and (value=='sigmoid' or value=='relu')):
				self.func = value


		## Initialize input, weigths, and outputs of the network
		self.X = np.zeros([1, self.numX])
		self.W = [0] * (len(self.numW)+1)
		for i in range(len(self.W)):
			if (i==0):
				self.W[i] = 10.0*np.random.rand(self.numX, self.numW[i])-5
			elif (i==len(self.W)-1):
				self.W[i] = 10.0*np.random.rand(self.numW[i-1], self.numY)-5
			else:
				self.W[i] = 10.0*np.random.rand(self.numW[i-1], self.numW[i])-5
		self.B = [0] * (len(self.numW)+1)
		for i in range(len(self.B)):
			if (i==len(self.B)-1):
				self.B[i] = 1.0*np.random.rand(1,self.numY)
			else:
				self.B[i] = 1.0*np.random.rand(1,self.numW[i])
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

	def relu(self, X):
		result = np.zeros(X.shape)
		for i in range(len(X[0])):
			result[0][i] = max([0,X[0][i]])
		return result

	'''
	NN forward propagation
	X(float[]) : array of inputs to calculate the forward propagation
	'''
	def forward(self, X):
		a = np.array([X])
		for i in range(len(self.W)):
			z = np.dot(a, self.W[i]) / np.prod(a.shape) + self.B[i] # z = a*w + b
			if(self.func=='sigmoid'):
				a = self.sigmoid(z) # a = sigma(a*w)
			elif(self.func=='relu'):
				a = self.relu(z)
			else:
				a = self.sigmoid(z)
		return a

	def train(self, X, Y, *arg):
		X = np.array([X])
		Y = np.array([Y])
		if(len(arg)==1):
			n = arg
		else:
			n = 1.0
		A = [0]*(len(self.W)+1)
		Z = [0]*len(self.W)
		A[0] = X
		for i in range(len(self.W)):
			Z[i] = np.dot(A[i], self.W[i]) / np.prod(A[i].shape) + self.B[i] # z = a*w + b
			if(self.func=='sigmoid'):
				A[i+1] = self.sigmoid(Z[i]) # a = sigma(a*w)
			elif(self.func=='relu'):
				A[i+1] = self.relu(Z[i])
			else:
				A[i+1] = self.sigmoid(Z[i])
		D = [0]*(len(self.W)+1)
		D[0] = (A[-1]-Y) / np.prod(A[-1].shape) * A[-1]*(1-A[-1])
		# D[0] = 0.5 * (A[-1]-Y)**2 / np.prod(A[-1].shape) * A[-1]*(1-A[-1])

		for i in range(len(self.W)):
			D[i+1] = np.transpose(np.dot(self.W[len(self.W)-i-1], np.transpose(D[i]))) * A[-2-i]*(1-A[-2-i])
			self.W[-1-i] = self.W[-1-i] - n * np.dot(np.transpose(A[-2-i]), D[i])
			self.B[-1-i] = self.B[-1-i] - n * D[i]

			
