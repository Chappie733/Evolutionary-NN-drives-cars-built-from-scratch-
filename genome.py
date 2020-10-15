from layer import *
from NN import NeuralNetwork

class Genome:
	DEFAULT_MUTATION_RATE = 0.5
	MAX_FITNESS = 50

	def __init__(self, state=None, net_shape=None, mutation_rate=DEFAULT_MUTATION_RATE):
		self.nn = NeuralNetwork(net_shape) # initialize the actual AI
		self.nn.set_state(state)
		self.mutation_rate = mutation_rate # percentage of maximum random mutation in every generation
		self.fitness = 0 # determines how good the genome has "behaved"
		self.alive = True # whether the car/genome is alive/active
		self.fsr = 100 # frames since (last) reward

	# returns a number of new genomes, random mutations of this one
	def get_mutations(self, mutations=1, mutation_rate=None):
	#	mutation_rate = k/f, f -> fitness, k -> constant (just an idea)
		if mutation_rate == None:
			mutation_rate = self.mutation_rate
		results = []
		for i in range(mutations): # for every mutation
			n_neurons = []
			biases = []
			weights = []
			activations = []
			# apply all the random changes
			for layer in self.nn.layers:
				new_biases = layer.biases * np.random.uniform(low=1-self.mutation_rate, high=1+self.mutation_rate, size=layer.biases.shape)
				try:
					new_weights = layer.weights * np.random.uniform(low=1-self.mutation_rate, high=1+self.mutation_rate, size=layer.weights.shape)
				except:
					new_weights = None	
				n_neurons.append(layer.n_neurons)
				biases.append(new_biases)
				weights.append(new_weights)
				activations.append(layer.func)
			results.append(n_neurons+weights+biases+activations) # store the applied changes as the new genome/AI
		return [Genome(state=state, mutation_rate=self.mutation_rate) for state in results]