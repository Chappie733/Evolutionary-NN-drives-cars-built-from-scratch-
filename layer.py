import numpy as np
import random

MIN_START_BIAS, MAX_START_BIAS = -10, 10
MIN_START_WEIGHT, MAX_START_WEIGHT = -3, 3

def sigmoid(x):
	return 1/(1+np.exp(-x))

def tanh(x):
	return np.tanh(x)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def relu(x):
	return x if x >= 0 else 0

def none(x):
	return x

'''
I'm not gonna comment this because If you don't know what this does I won't be able to explain it in a comment,
and if you do you most likely won't need them anyways.
This class represents a Dense layer in the neural network(s)
'''
class Layer:

	def __init__(self, n_neurons, MIN=(MIN_START_BIAS, MIN_START_WEIGHT), MAX=(MAX_START_BIAS, MAX_START_WEIGHT)
		, activation=None, dtype=np.float64):
		self.n_neurons = n_neurons
		self.neurons = np.array([0 for _ in range(n_neurons)], dtype=dtype)
		self.biases = np.array([random.uniform(MIN[0], MAX[0]) for _ in range(n_neurons)], dtype=dtype)
		self.weights = None # [ending neuron][starting neuron]
		self.func = activation if activation != None else none # I would use a lambda x: x but then I can't use pickle.dump on the layers for some reason
		self.sw = (MIN[1], MAX[1]) # bounds of the starting random weights
		self.dtype = dtype

	def log(self, layer):
		self.weights = np.array([[random.uniform(self.sw[0], self.sw[1]) for _ in range(self.n_neurons)] for i in range(layer.n_neurons)], dtype=self.dtype)

	def feed(self, prev):
		if isinstance(prev, Layer):
			for i in range(self.n_neurons):
				self.neurons[i] = prev.activation(i, bias=self.biases[i])
		else:
			if len(prev) == self.n_neurons:
				self.neurons = prev
			else:
				raise Exception("Expected " + str(self.n_neurons) + " inputs, but received " + str(len(prev)))

	def activation(self, neuron_index, bias=0):
		return self.func(np.dot(self.neurons, self.weights[neuron_index])+bias)

	def __call__(self, x):
		self.feed(x)