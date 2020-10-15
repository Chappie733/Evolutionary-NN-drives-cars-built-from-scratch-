import numpy as np
import pickle
from layer import *

def save_model(model, path):
	with open(path, "wb") as f:
		pickle.dump(model, f)

def load_model(path):
	with open(path, "rb") as f:
		model = pickle.load(f)
	return model

'''
The name explains it
'''
class NeuralNetwork:

	def __init__(self, layers=None):
		if layers != None:
			for i in range(len(layers)-1):
				layers[i].log(layers[i+1])
		self.layers = layers

	def predict(self, x):
		self.layers[0](x)
		for i in range(1, len(self.layers)):
			self.layers[i](self.layers[i-1])
		return self.layers[-1].neurons

	def __call__(self, x):
		return self.predict(x)

	def get_state(self):
		return [layer.n_neurons for layer in layers] + [np.copy(layer.weights) for layer in self.layers] + [np.copy(layer.biases) for layer in self.layers] + [layer.func for layer in layers]

	def set_state(self, state):
		if state == None:
			return
		self.layers = []
		for i in range(int(len(state)/4)):
			self.layers.append( Layer(state[i], activation=state[int(len(state)*3/4)+i]) )
			self.layers[-1].weights = state[int(len(state)*1/4)+i]
			self.layers[-1].biases = state[int(len(state)*1/2)+i]

	def __str__(self):
		res = ''
		for i,layer in enumerate(self.layers):
			res += "Layer #" + str(i) + ", neurons: " + str(layer.n_neurons) + ", activation: " + str(layer.func) + "\n"
		return res