import Neuron
import math

class TanhNeuron(Neuron.Neuron):

	def __init__(self):
		pass

	def Activate(self, x):
		return math.tanh(x)

	def Derivative(self, x):
		coshx = math.cosh(x)
		denom = (math.cosh(2*x) + 1)
		return 4 * coshx * coshx / (denom * denom)
