# -*- coding:utf-8 -*-
"""
Simple implementation of LSTM RNN,
proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

dsoutner@kky.zcu.cz, 2013

"""
__version__ = "0.1"


import numpy as np
import Neuron
import random
import math


class IdentityNeuron():
	def Activate(self, x): 
		return x
	def Derivative(self, x):
		return 1;

class SoftmaxNeuron():
	def Activate(self, x):
		e = np.exp(x)
		return (e / np.sum(e))
	def Derivative(self, m):
		m = self.Activate(input)
		return m - np.power(m, 2)

class TanhNeuron():
	def Activate(self, x):
		return math.tanh(x)
	def Derivative(self, x):
		coshx = math.cosh(x)
		denom = (math.cosh(2 * x) + 1)
		return 4 * coshx * coshx / (denom * denom)

class SigmoidNeuron():
	def Activate(self, x):
		return 1 / (1 + math.exp(-x))
	def Derivative(self, x):
		act = self.Activate(x)
		return act * (1 - act)


class LSTM(): 


	def getHiddenState():
		return self.context


	def setHiddenState(new_state):
		self.context = new_state


	def getHiddenDimension():
		return self.cell_blocks


	def __init__(self, input_dimension, output_dimension, cell_blocks, rnd_seed=None):

		init_weight_range = 0.1
		biasInputGate = 2
		biasForgetGate = -2
		biasOutputGate = 2
		self.learningRate = 0.1

		self.output_dimension = output_dimension
		self.cell_blocks = cell_blocks

		self.CEC = []
		self.context = []
		self.peepInputGate = []
		self.peepForgetGate = []
		self.peepOutputGate = []

		random.seed(rnd_seed)

		self.CEC = np.zeros(self.cell_blocks)
		self.context = np.zeros(self.cell_blocks)

		self.full_input_dimension = input_dimension + self.cell_blocks + 1; #+1 for bias
		self.full_hidden_dimension = self.cell_blocks + 1; #+1 for bias

		self.neuronNetInput = TanhNeuron()
		self.neuronInputGate = SigmoidNeuron()
		self.neuronForgetGate = SigmoidNeuron()
		self.neuronOutputGate = SigmoidNeuron()
		self.neuronCECSquash = IdentityNeuron()
		self.neuronNetOutput = SoftmaxNeuron()

		self.weightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))

		self.dSdwWeightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))

		self.weightsOutputGate = np.zeros((self.cell_blocks, self.full_input_dimension))

		for i in range(self.full_input_dimension):
			for j in range(self.cell_blocks):
				self.weightsNetInput[j][i] = (random.random() * 2 - 1) * init_weight_range
				self.weightsInputGate[j][i] = (random.random() * 2 - 1) * init_weight_range
				self.weightsForgetGate[j][i] = (random.random() * 2 - 1) * init_weight_range
				self.weightsOutputGate[j][i] = (random.random() * 2 - 1) * init_weight_range


		for j in range(self.cell_blocks):
			self.weightsInputGate[j][self.full_input_dimension-1] += biasInputGate
			self.weightsForgetGate[j][self.full_input_dimension-1] += biasForgetGate
			self.weightsOutputGate[j][self.full_input_dimension-1] += biasOutputGate


		self.peepInputGate = np.zeros(self.cell_blocks)
		self.peepForgetGate = np.zeros(self.cell_blocks)
		self.peepOutputGate = np.zeros(self.cell_blocks)

		for j in range(cell_blocks):
			self.peepInputGate[j] = (random.random() * 2 - 1) * init_weight_range
			self.peepForgetGate[j] = (random.random() * 2 - 1) * init_weight_range
			self.peepOutputGate[j] = (random.random() * 2 - 1) * init_weight_range

		self.weightsGlobalOutput = np.zeros((self.output_dimension, self.full_hidden_dimension))

		for j in range(self.full_hidden_dimension):
			for k in range(self.output_dimension):
				self.weightsGlobalOutput[k][j] = (random.random() * 2 - 1) * init_weight_range


	def Reset(self):
		#TODO: reset deltas here?
		for c in range(len(self.CEC)):
			self.CEC[c] = 0.0
		for c in range(len(self.context)):
			self.context[c] = 0.0
		#reset accumulated partials
		for c in range(self.cell_blocks):
			for i in range(self.full_input_dimension):
				self.dSdwWeightsForgetGate[c][i] = 0
				self.dSdwWeightsInputGate[c][i] = 0
				self.dSdwWeightsNetInput[c][i] = 0


	def Next(self, input, target_output):
	#public double[] Next(double[] input, double[] target_output) {

		#setup input vector
		full_input = np.zeros((self.full_input_dimension))
		loc = 0
		for i in range(len(input)):
			full_input[loc] = input[i]
			loc += 1
		for c in range(len(self.context)):
			full_input[loc] = self.context[c]
			loc += 1
		full_input[loc] = 1.0 #bias
		loc += 1
		

		#cell block arrays
		NetInputSum = np.zeros((self.cell_blocks))
		InputGateSum = np.zeros((self.cell_blocks))
		ForgetGateSum = np.zeros((self.cell_blocks))
		OutputGateSum = np.zeros((self.cell_blocks))

		NetInputAct = np.zeros((self.cell_blocks))
		InputGateAct = np.zeros((self.cell_blocks))
		ForgetGateAct = np.zeros((self.cell_blocks))
		OutputGateAct = np.zeros((self.cell_blocks))

		CECSquashAct = np.zeros((self.cell_blocks))

		NetOutputAct = np.zeros((self.cell_blocks))

		#inputs to cell blocks
		for i in range(self.full_input_dimension):
			for j in range(self.cell_blocks):
				NetInputSum[j] += self.weightsNetInput[j][i] * full_input[i]
				InputGateSum[j] += self.weightsInputGate[j][i] * full_input[i]
				ForgetGateSum[j] += self.weightsForgetGate[j][i] * full_input[i]
				OutputGateSum[j] += self.weightsOutputGate[j][i] * full_input[i]

		CEC1 = np.zeros((self.cell_blocks))
		CEC2 = np.zeros((self.cell_blocks))
		CEC3 = np.zeros((self.cell_blocks))

		#internals of cell blocks
		for j in range(self.cell_blocks):
			CEC1[j] = self.CEC[j]

			NetInputAct[j] = self.neuronNetInput.Activate(NetInputSum[j])

			ForgetGateSum[j] += self.peepForgetGate[j] * CEC1[j]
			ForgetGateAct[j] = self.neuronForgetGate.Activate(ForgetGateSum[j])

			CEC2[j] = CEC1[j] * ForgetGateAct[j]

			InputGateSum[j] += self.peepInputGate[j] * CEC2[j]
			InputGateAct[j] = self.neuronInputGate.Activate(InputGateSum[j])

			CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]

			OutputGateSum[j] += self.peepOutputGate[j] * CEC3[j] #TODO: this versus squashed?
			OutputGateAct[j] = self.neuronOutputGate.Activate(OutputGateSum[j])

			CECSquashAct[j] = self.neuronCECSquash.Activate(CEC3[j])

			NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j]


		#prepare hidden layer plus bias
		full_hidden = np.zeros((self.full_hidden_dimension))
		loc = 0
		for j in range(self.cell_blocks):
			full_hidden[loc] = NetOutputAct[j]
			loc += 1
		full_hidden[loc] = 1.0		#bias
		loc += 1

		#calculate output
		output = np.zeros((self.output_dimension))
		for k in range(self.output_dimension):
			for j in range(self.full_hidden_dimension):
				output[k] += self.weightsGlobalOutput[k][j] * full_hidden[j]
			#output not squashed
		output = self.neuronNetOutput.Activate(output)
		
		#//BACKPROP
		#scale partials
		for c in range(self.cell_blocks):
			for i in range(self.full_input_dimension):
				self.dSdwWeightsInputGate[c][i] *= ForgetGateAct[c]
				self.dSdwWeightsForgetGate[c][i] *= ForgetGateAct[c]
				self.dSdwWeightsNetInput[c][i] *= ForgetGateAct[c]

				self.dSdwWeightsInputGate[c][i] += full_input[i] * self.neuronInputGate.Derivative(InputGateSum[c]) * NetInputAct[c]
				self.dSdwWeightsForgetGate[c][i] += full_input[i] * self.neuronForgetGate.Derivative(ForgetGateSum[c]) * CEC1[c]
				self.dSdwWeightsNetInput[c][i] += full_input[i] * self.neuronNetInput.Derivative(NetInputSum[c]) * InputGateAct[c]

		if (target_output != None):
			deltaGlobalOutputPre = np.zeros((self.output_dimension))
			for k in range(self.output_dimension):
				deltaGlobalOutputPre[k] = target_output[k] - output[k]

			#output to hidden
			deltaNetOutput = np.zeros((self.cell_blocks))
			for k in range(self.output_dimension):
				#links
				for c in range(self.cell_blocks):
					deltaNetOutput[c] += deltaGlobalOutputPre[k] * self.weightsGlobalOutput[k][c]
					self.weightsGlobalOutput[k][c] += deltaGlobalOutputPre[k] * NetOutputAct[c] * self.learningRate
				#bias
				self.weightsGlobalOutput[k][self.cell_blocks] += deltaGlobalOutputPre[k] * 1.0 * self.learningRate

			for c in range(self.cell_blocks):
				#update output gates
				deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c]
				deltaOutputGatePre = self.neuronOutputGate.Derivative(OutputGateSum[c]) * deltaOutputGatePost
				for i in range(self.full_input_dimension):
					self.weightsOutputGate[c][i] += full_input[i] * deltaOutputGatePre * self.learningRate
				self.peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * self.learningRate
				#before outgate
				deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c] * self.neuronCECSquash.Derivative(CEC3[c])

				#update input gates
				deltaInputGatePost = deltaCEC3 * NetInputAct[c]
				deltaInputGatePre = self.neuronInputGate.Derivative(InputGateSum[c]) * deltaInputGatePost
				for i in range(self.full_input_dimension):
					self.weightsInputGate[c][i] += self.dSdwWeightsInputGate[c][i] * deltaCEC3 * self.learningRate
				self.peepInputGate[c] += CEC2[c] * deltaInputGatePre * self.learningRate

				#before ingate
				deltaCEC2 = deltaCEC3

				#update forget gates
				deltaForgetGatePost = deltaCEC2 * CEC1[c]
				deltaForgetGatePre = self.neuronForgetGate.Derivative(ForgetGateSum[c]) * deltaForgetGatePost
				for i in range(self.full_input_dimension):
					self.weightsForgetGate[c][i] += self.dSdwWeightsForgetGate[c][i] * deltaCEC2 * self.learningRate
				self.peepForgetGate[c] += CEC1[c] * deltaForgetGatePre * self.learningRate

				#update cell inputs
				for i in range(self.full_input_dimension):
					self.weightsNetInput[c][i] += self.dSdwWeightsNetInput[c][i] * deltaCEC3 * self.learningRate
				#no peeps for cell inputs

		#//roll-over context to next time step
		for j in range(self.cell_blocks):
			self.context[j] = NetOutputAct[j]
			self.CEC[j] = CEC3[j]

		#give results
		return output

	def forward(self, input):
		#setup input vector
		full_input = np.zeros((self.full_input_dimension))
		loc = 0
		for i in range(len(input)):
			full_input[loc] = input[i]
			loc += 1
		for c in range(len(self.context)):
			full_input[loc] = self.context[c]
			loc += 1
		full_input[loc] = 1.0 #bias
		loc += 1

		#cell block arrays
		NetInputSum = np.zeros((self.cell_blocks))
		InputGateSum = np.zeros((self.cell_blocks))
		ForgetGateSum = np.zeros((self.cell_blocks))
		OutputGateSum = np.zeros((self.cell_blocks))

		NetInputAct = np.zeros((self.cell_blocks))
		InputGateAct = np.zeros((self.cell_blocks))
		ForgetGateAct = np.zeros((self.cell_blocks))
		OutputGateAct = np.zeros((self.cell_blocks))

		CECSquashAct = np.zeros((self.cell_blocks))

		NetOutputAct = np.zeros((self.cell_blocks))

		#inputs to cell blocks
		for i in range(self.full_input_dimension):
			for j in range(self.cell_blocks):
				NetInputSum[j] += self.weightsNetInput[j][i] * full_input[i]
				InputGateSum[j] += self.weightsInputGate[j][i] * full_input[i]
				ForgetGateSum[j] += self.weightsForgetGate[j][i] * full_input[i]
				OutputGateSum[j] += self.weightsOutputGate[j][i] * full_input[i]

		CEC1 = np.zeros((self.cell_blocks))
		CEC2 = np.zeros((self.cell_blocks))
		CEC3 = np.zeros((self.cell_blocks))

		#internals of cell blocks
		for j in range(self.cell_blocks):
			CEC1[j] = self.CEC[j]
			NetInputAct[j] = self.neuronNetInput.Activate(NetInputSum[j])
			ForgetGateSum[j] += self.peepForgetGate[j] * CEC1[j]
			ForgetGateAct[j] = self.neuronForgetGate.Activate(ForgetGateSum[j])
			CEC2[j] = CEC1[j] * ForgetGateAct[j]
			InputGateSum[j] += self.peepInputGate[j] * CEC2[j]
			InputGateAct[j] = self.neuronInputGate.Activate(InputGateSum[j])
			CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]
			OutputGateSum[j] += self.peepOutputGate[j] * CEC3[j]
			OutputGateAct[j] = self.neuronOutputGate.Activate(OutputGateSum[j])
			CECSquashAct[j] = self.neuronCECSquash.Activate(CEC3[j])
			NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j]

		#prepare hidden layer plus bias
		full_hidden = np.zeros((self.full_hidden_dimension))
		loc = 0
		for j in range(self.cell_blocks):
			full_hidden[loc] = NetOutputAct[j]
			loc += 1
		full_hidden[loc] = 1.0		#bias
		loc += 1

		#calculate output
		output = np.zeros((self.output_dimension))
		for k in range(self.output_dimension):
			for j in range(self.full_hidden_dimension):
				output[k] += self.weightsGlobalOutput[k][j] * full_hidden[j]
			#output not squashed
		output = self.neuronNetOutput.Activate(output)
		#give results
		return output




