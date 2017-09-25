
import numpy as np 
import math

class NN(object):
	'''
	Create a Custom Neural Network
	input:
	numX, numW, numY
	numX(int) : number of inputs nodes of the network. Default 2
	numW([int]) : list of number of weight for each hidden layer. [numW1, numW2,...,numWH]. Default [5]
	numY(int) : number of outputs to the network. Default 2
	'''
	def __init__(self, *args, **kwargs):
		## Default values
		self.numX = 2
		self.numW = [5]
		self.numY = 2
		self.func = 'relu' # default function set to relu
		self.epoch = 3
		self.weight = 10
		self.dropout = 1.0
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epx = 1e-8
		self.m = 0
		self.v = 0
		self.optimizer = 'Vanilla'

		## Initialize input variables
		if (len(args)>0):
			self.numX = int(args[0])
			if(len(args)>1):
				self.numY = int(args[1])	
				if(len(args)>2):
					if(type(args[2])==type(1)):
						self.numW = [int(args[2])]
					else:
						self.numW = args[2]
		self.numH = len(self.numW) # number of hidden layers

		
		for name, value in kwargs.items():
			# if (name=='func' and (value=='sigmoid' or value=='relu')):
			if(name=='func'):
				self.func = value
				print('activation :', value)
			if(name=='epoch'):
				self.epoch = int(value)
			if(name=='weight'):
				if(value=='xavier'):
					self.weight = value
				else:
					self.weight = float(value)
			if(name=='dropout'):# and value < 1.0 and value > 0.0):
				self.dropout = float(value)


		## Initialize input, weights, and outputs of the network
		self.X = np.zeros([1, self.numX])
		self.W = [0] * (len(self.numW)+1)
		self.B = [0] * (len(self.numW)+1)
		if(self.weight=='xavier'):
			for i in range(len(self.W)):
				if (i==0):
					self.W[i] = np.random.normal(0, 2/(self.numX+self.numY), (self.numX, self.numW[i]))
				elif (i==len(self.W)-1):
					self.W[i] = np.random.normal(0, 2/(self.numX+self.numY), (self.numW[i-1], self.numY))
				else:
					self.W[i] = np.random.normal(0, 2/(self.numX+self.numY),(self.numW[i-1], self.numW[i]))
			for i in range(len(self.B)):
				if (i==len(self.B)-1):
					self.B[i] = np.random.normal(0, 2/(self.numX+self.numY), (1,self.numY))
				else:
					self.B[i] = np.random.normal(0, 2/(self.numX+self.numY), (1,self.numW[i]))
		else:
			W_scale = self.weight
			W_offset = W_scale/2.0
			for i in range(len(self.W)):
				if (i==0):
					self.W[i] = W_scale*np.random.rand(self.numX, self.numW[i])-W_offset
				elif (i==len(self.W)-1):
					self.W[i] = W_scale*np.random.rand(self.numW[i-1], self.numY)-W_offset
				else:
					self.W[i] = W_scale*np.random.rand(self.numW[i-1], self.numW[i])-W_offset
			for i in range(len(self.B)):
				if (i==len(self.B)-1):
					self.B[i] = W_scale*np.random.rand(1,self.numY)-W_offset
				else:
					self.B[i] = W_offset*np.random.rand(1,self.numW[i])-W_offset
		self.Y = np.zeros([self.numY,1])

	'''
	Sigmoid function for an array
	X(float[]) : array of values to calculate sigmoid
	'''	
	def sigmoid(self, X):
		result = np.zeros(X.shape)
		for i in range(len(X[0][:])):
			try:
				result[0][i] = 1 / (1+math.exp(-1*X[0][i])) # sigmoid function 
			except OverflowError:
				print("Over flow", X[0][i])
		return result

	def relu(self, X):
		result = np.zeros(X.shape)
		for i in range(len(X[0][:])):
			result[0][i] = max([0,X[0][i]])
		return result

	def relu2(self, X):
		result = X
		return result

	def lrelu(self, X):
		result = np.zeros(X.shape)
		for i in range(len(X[0][:])):
			result[0][i] = X[0][i] if X[0][i] > 0 else 0.001*X[0][i]
		return result

	'''
	NN forward propagation
	X(float[]) : array of inputs to calculate the forward propagation
	'''
	def forward(self, X):
		if(type(X)==type([])):
			X = np.array([X])
		a = X
		for i in range(len(self.W)):
			z = np.dot(a / np.prod(a.shape), self.W[i]) + self.B[i] # z = a*w + b
			if (i==len(self.W)-1):
				return z
			else:
				if(self.func=='sigmoid'):
					a = self.sigmoid(z) # a = sigma(a*w)
				elif(self.func=='relu'):
					a = self.relu(z)
				elif(self.func=='reul2'):
					a = self.relu2(z)
				elif(self.func=='lrelu'):
					a = self.lrelu(z)
				else:
					a = self.relu(z)
				# if(self.dropout<1.0):
				# 	a = np.multiply(a, np.random.binomial([p for p in np.ones(a.shape)], self.dropout)[0] * (1.0/(1-self.dropout)))
		return a

	def train(self, X, Y, *arg):
		if(type(X)==type([])):
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
			Z[i] = np.dot(A[i] / np.prod(A[i].shape), self.W[i]) + self.B[i] # z = a*w + b
			if (i==(len(self.W)-1)):
				A[i+1] = Z[i]
			else:
				if(self.func=='sigmoid'):
					A[i+1] = self.sigmoid(Z[i]) # a = sigma(a*w)
				elif(self.func=='relu'):
					A[i+1] = self.relu(Z[i])
				elif(self.func=='relu2'):
					A[i+1] = self.relu2(Z[i])
				elif(self.func=='lrelu'):
					A[i+1] = self.lrelu(Z[i])
				else:
					A[i+1] = self.relu(Z[i])
				if(self.dropout<1.0):
					dropout_list = np.random.binomial([p for p in np.ones(A[i+1].shape)], self.dropout)
					if(np.sum(dropout_list)==0):
						dropout_list[np.random.randint(len(dropout_list))] = 1
					A[i+1] = np.multiply(A[i+1], dropout_list[0]) #/ np.sum(dropout_list) #* (1.0/(1-self.dropout)))
		
		D = [0]*(len(self.W)+1)
		# D[0] = (A[-1]-Y) * A[-1]*(1-A[-1])
		D[0] = (A[-1]-Y)
		# D[0] = 0.5 * (A[-1]-Y)**2 / np.prod(A[-1].shape) * A[-1]*(1-A[-1])
		np.seterr(all='raise')
		for i in range(len(self.W)):
			if (self.func=='sigmoid'):
				D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.multiply(A[-2-i],(1-A[-2-i])))
			elif (self.func=='relu'):
				try:
					D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.array([1 if element>0 else 0 for element in A[-2-i][0]]))
				except:
					print(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))))
			elif (self.func=='relu2'):
				D[i+1] = np.transpose(np.dot(self.W[-i-1], np.transpose(D[i])))
			elif (self.func=='lrelu'):
				D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.array([1 if element>0 else 0.001 for element in A[-2-i][0]]))
			else:
				D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.array([1 if element>0 else 0 for element in A[-2-i][0]]))
			dx = (np.dot(np.transpose(A[-2-i]), D[i]))
			if (self.optimizer=='Vanilla'):
				self.W[-1-i] = self.W[-1-i] - n * dx
				self.B[-1-i] = self.B[-1-i] - n * D[i]
			elif (self.optimizer=='ADAM'):
				try:
					self.m = self.beta1*self.m + (1-self.beta1)*dx
					self.v = self.beta2*self.v + (1-self.beta2)*(dx**2)
					self.W[-1-i] = self.W[-1-i] - n * self.m / (np.sqrt(self.v)+self.eps)
				except:
					print('wrong')
					# print(self.layers[-1-i].m, self.layers[-1-i].v)
				self.B[-1-i] = self.B[-1-i] - n * D[i]
			else: # use Vanilla
				self.W[-1-i] = self.W[-1-i] - n * dx
				self.B[-1-i] = self.B[-1-i] - n * D[i]

class NNb(NN):
	'''
	Create a Custom Neural Network
	input:
	numX, numW, numY
	numX(int) : number of inputs nodes of the network. Default 2
	numW([int]) : list of number of weight for each hidden layer. [numW1, numW2,...,numWH]. Default [5]
	numY(int) : number of outputs to the network. Default 2
	'''
	def __init__(self):
		## Default values
		self.numX = 0
		self.numW = []
		self.numY = 0
		self.epoch = 3
		self.num_layer = 0


		## Initialize input, weights, and outputs of the network
		self.X = []
		self.W = []
		self.B = []
		self.Y = []
		self.layers = []

		self.t = 0

	def addLayer(self, layer):
		if(layer.type=='input' and self.num_layer==0):
			self.numX = layer.num_node
		else:
			if(layer.type=='output'):
				self.numY = layer.num_node
			else:
				self.numW.append(layer.num_node)
			if(layer.weight=='xavier'):
				print('weight : Xavier')
				layer.value = np.random.normal(0, 1/(self.numX+self.numY),(self.layers[-1].num_node, layer.num_node))
				layer.B = np.random.normal(0, 1/(self.numX+self.numY),(1, layer.num_node))
			else:
				print('weight : Random')
				layer.value = layer.weight_scale*np.random.rand(self.layers[-1].num_node, layer.num_node)-layer.weight_scale/2.0
				layer.B = layer.weight_scale*np.random.rand(1, layer.num_node)-layer.weight_scale/2.0
			self.W.append(layer.value)
			self.B.append(layer.B)

		self.layers.append(layer)

	'''
	NN forward propagation
	X(float[]) : array of inputs to calculate the forward propagation
	'''
	def forward(self, X):
		if(type(X)==type([])):
			X = np.array([X])
		a = X
		for i in range(len(self.W)):
			z = np.dot(a/np.prod(a.shape), self.W[i]) + self.B[i] # z = a*w + b / np.prod(a.shape)
			if (i==len(self.W)-1):
				return z
			else:
				if(self.layers[i+1].func=='sigmoid'):
					a = self.sigmoid(z) # a = sigma(a*w)
				elif(self.layers[i+1].func=='relu'):
					a = self.relu(z)
				elif(self.layers[i+1].func=='reul2'):
					a = self.relu2(z)
				elif(self.layers[i+1].func=='lrelu'):
					a = self.lrelu(z)
				else:
					a = self.relu(z)
				# if(self.dropout<1.0):
				# 	a = np.multiply(a, np.random.binomial([p for p in np.ones(a.shape)], self.dropout)[0] * (1.0/(1-self.dropout)))
		return a

	def train(self, X, Y, *arg):
		if(type(X)==type([])):
			X = np.array([X])
			Y = np.array([Y])
		if(len(arg)==1):
			n = float(arg[0])
		else:
			n = 1.0

		self.t += 1
		A = [0]*(len(self.W)+1)
		Z = [0]*len(self.W)
		A[0] = X
		for i in range(len(self.W)):
			Z[i] = np.dot(A[i]/np.prod(A[i].shape), self.W[i]) + self.B[i] # z = a*w + b
			if (i==len(self.W)-1):
				A[i+1] = Z[i]
			else:
				if(self.layers[i+1].func=='sigmoid'):
					A[i+1] = self.sigmoid(Z[i]) # a = sigma(a*w)
				elif(self.layers[i+1].func=='relu'):
					A[i+1] = self.relu(Z[i])
				elif(self.layers[i+1].func=='relu2'):
					A[i+1] = self.relu2(Z[i])
				elif(self.layers[i+1].func=='lrelu'):
					A[i+1] = self.lrelu(Z[i])
				else:
					A[i+1] = self.relu(Z[i])
				if(self.layers[i+1].dropout<1.0):
					dropout_list = np.random.binomial([p for p in np.ones(A[i+1].shape)], self.layers[i+1].dropout)
					if(np.sum(dropout_list)==0):
						dropout_list[np.random.randint(len(dropout_list))] = 1
					A[i+1] = np.multiply(A[i+1], dropout_list[0]) #/ np.sum(dropout_list) #* (1.0/(1-self.dropout)))
		
		D = [0]*(len(self.W)+1)
		# D[0] = (A[-1]-Y) * A[-1]*(1-A[-1])
		D[0] = (A[-1]-Y)
		# D[0] = 0.5 * (A[-1]-Y)**2 / np.prod(A[-1].shape) * A[-1]*(1-A[-1])
		np.seterr(all='raise')
		for i in range(len(self.W)):
			if (self.layers[-1-i].func=='sigmoid'):
				D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.multiply(A[-2-i],(1-A[-2-i])))
			elif (self.layers[-1-i].func=='relu'):
				D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.array([1 if element>0 else 0 for element in A[-2-i][0]]))
			elif (self.layers[-1-i].func=='relu2'):
				D[i+1] = np.transpose(np.dot(self.W[-i-1], np.transpose(D[i])))
			elif (self.layers[-1-i].func=='lrelu'):
				D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.array([1 if element>0 else 0.001 for element in A[-2-i][0]]))
			else: # Default is ReLU
				D[i+1] = np.multiply(np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))), np.array([1 if element>0 else 0 for element in A[-2-i][0]]))
			
			# Update the parameters (reference to http://cs231n.github.io/neural-networks-3/)
			dx = (np.dot(np.transpose(A[-2-i]), D[i]))
			if (self.layers[-1-i].optimizer=='Vanilla'):
				self.W[-1-i] = self.W[-1-i] - n * dx

			elif (self.layers[-1-i].optimizer=='Momentum'):
				self.layers[-1-i].v = np.multiply(self.layers[-1-i].v, (self.layers[-1-i].v>10e-100))
				self.layers[-1-i].v_prev = self.layers[-1-i].v
				self.layers[-1-i].v = self.layers[-1-i].mu * self.layers[-1-i].v - n * dx
				self.W[-1-i] = self.W[-1-i] - self.layers[-1-i].mu * self.layers[-1-i].v_prev + (1+self.layers[-1-i].mu) * self.layers[-1-i].v
			
			elif (self.layers[-1-i].optimizer=='Adagrad'):
				self.layers[-1-i].cache = np.multiply(dx, dx)
				self.W[-1-i] = self.W[-1-i] - n * np.divide(dx, (np.sqrt(self.layers[-1-i].cache) + self.layers[-1-1].eps))
			
			elif (self.layers[-1-i].optimizer=='RMSprop'):
				self.layers[-1-i].cache = self.layers[-1-i].decayrate * self.layers[-1-i].cache + (1-self.layers[-1-i].decayrate) * (dx**2)
				# self.W[-1-i] = self.W[-1-i] - n * np.divide(dx, (np.sqrt(self.layers[-1-i].cache) + self.layers[-1-1].eps))
				self.W[-1-i] = self.W[-1-i] - n * dx * self.layers[-1-i].mu * (np.sqrt(self.layers[-1-i].cache) + self.layers[-1-1].eps)
			
			elif (self.layers[-1-i].optimizer=='ADAM'):
				self.layers[-1-i].m = self.layers[-1-i].beta1*self.layers[-1-i].m + (1-self.layers[-1-i].beta1)*dx
				self.layers[-1-i].m = np.multiply(self.layers[-1-i].m, (self.layers[-1-i].m>10e-100))
				mt = self.layers[-1-i].m/(1-self.layers[-1-i].beta1**self.t)
				self.layers[-1-i].v = self.layers[-1-i].beta2*self.layers[-1-i].v + (1-self.layers[-1-i].beta2)*(dx**2)
				self.layers[-1-i].v = np.multiply(self.layers[-1-i].v, (self.layers[-1-i].v>10e-100))
				vt = self.layers[-1-i].v/(1-self.layers[-1-i].beta2**self.t)
				self.W[-1-i] = self.W[-1-i] - n / (np.sqrt(self.layers[-1-i].v+self.layers[-1-i].eps)) * self.layers[-1-i].m
			
			else: # use Vanilla by default
				self.W[-1-i] = self.W[-1-i] - n * dx
			self.B[-1-i] = self.B[-1-i] - n * D[i]

class Layer():
	def __init__(self, type, num_node, **kargs):
		self.type = type
		self.num_node = num_node
		self.value = None
		self.B = None

		# variables for optimization
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.eps = 1e-8
		self.m = 0
		self.v = 0
		self.v_prev = 0
		self.cache = 0
		self.decayrate = 0.99
		self.mu = 0.1
		# if(self.type=='output' or self.type=='hidden'):
		# 	self.value = np.zeros([prev_num_node, self.num_node])


		self.func = 'relu'
		self.dropout = 1.0
		self.weight = 'Random'
		self.weight_scale = 1.0
		self.optimizer = 'Vanilla'

		for name, value in kargs.items():
			if(name=='func'):
				self.func = value
			elif(name=='dropout'):
				self.dropout = float(value)
			elif(name=='weight'):
				self.weight = value
			elif(name=='weight_scale'):
				self.weight_scale = float(value)
			elif(name=='optimizer'):
				self.optimizer = value
