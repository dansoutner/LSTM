import Neuron

class IdentityNeuron(Neuron.Neuron):

	def Activate(self, x): 
		return x

	def Derivative(self, x):
		# TODO Auto-generated method stub
		return 1;


