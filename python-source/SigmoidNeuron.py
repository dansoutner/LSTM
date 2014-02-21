import Neuron
import math

class SigmoidNeuron(Neuron.Neuron):

	def Activate(self, x):
		return 1 / (1 + math.exp(-x))

	def Derivative(self, x):
		act = self.Activate(x)
		return act * (1 - act)
