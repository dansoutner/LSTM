import Neuron
import math
import numpy as np

class SoftmaxNeuron(Neuron.Neuron):

	def __init__(self):
		pass

	def Activate(self, x):
		e = np.exp(x)
		return (e / np.sum(e))

	def Derivative(self, m):
		m = self.Activate(input)
		return m - np.power(m, 2)

