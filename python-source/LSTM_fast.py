# -*- coding:utf-8 -*-
"""
Simple implementation of LSTM RNN,
proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

dsoutner@kky.zcu.cz, 2013

"""
__version__ = "0.2"

import numpy as np
import random

import time
t = time.time


def SoftmaxActivate(x):
	e = np.exp(x)
	return (e / np.sum(e))

def SoftmaxDerivative(x):
	m = SoftmaxActivate(x)
	return m - np.power(m, 2)

def TanhActivate(x):
	return np.tanh(x)

def TanhDerivative(x):
	coshx = np.cosh(x)
	denom = (np.cosh(2 * x) + 1)
	return 4 * coshx * coshx / (denom * denom)

def SigmoidActivate(x):
	return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
	act = 1 / (1 + np.exp(-x))
	return act * (1 - act)



class LSTM(): 

	def getHiddenState(self):
		return self.context


	def setHiddenState(self, new_state):
		self.context = new_state


	def getHiddenDimension(self):
		return self.cell_blocks


	def __init__(self, input_dimension, output_dimension, cell_blocks, rnd_seed=None):

		init_weight_range = 0.1
		biasInputGate = 2
		biasForgetGate = -2
		biasOutputGate = 2
		self.learningRate = 0.1

		random.seed(rnd_seed)

		self.output_dimension = output_dimension
		self.cell_blocks = cell_blocks

		self.CEC = np.zeros(self.cell_blocks)
		self.context = np.zeros(self.cell_blocks)

		self.full_input_dimension = input_dimension + self.cell_blocks + 1; #+1 for bias
		self.full_hidden_dimension = self.cell_blocks + 1; #+1 for bias

		self.weightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsOutputGate = np.zeros((self.cell_blocks, self.full_input_dimension))

		self.dSdwWeightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))

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
		self.CEC = np.zeros(self.cell_blocks)
		self.context = np.zeros(self.cell_blocks)
		self.dSdwWeightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))


	def Next(self, input, target_output, learningRate=None):
		
		if not learningRate == None:
			self.learningRate = learningRate
			
		#setup input vector
		full_input = np.zeros((self.full_input_dimension))
		full_input[0:len(input)] = input
		full_input[len(input):len(self.context)+len(input)] = self.context
		full_input[-1] = 1.0
		
		
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
		CEC1 = np.zeros((self.cell_blocks))
		CEC2 = np.zeros((self.cell_blocks))
		CEC3 = np.zeros((self.cell_blocks))

		
		for i in range(self.full_input_dimension):
			NetInputSum += self.weightsNetInput[:, i] * full_input[i]
			InputGateSum += self.weightsInputGate[:, i] * full_input[i]
			ForgetGateSum += self.weightsForgetGate[:, i] * full_input[i]
			OutputGateSum += self.weightsOutputGate[:, i] * full_input[i]
		
		#internals of cell blocks
		CEC1 = self.CEC
		NetInputAct = TanhActivate(NetInputSum)
		for j in range(self.cell_blocks):
			ForgetGateSum[j] += self.peepForgetGate[j] * CEC1[j]

		ForgetGateAct = SigmoidActivate(ForgetGateSum)
		CEC2 = CEC1 * ForgetGateAct

		for j in range(self.cell_blocks):
			InputGateSum[j] += self.peepInputGate[j] * CEC2[j]
		
		InputGateAct = SigmoidActivate(InputGateSum)
		CEC3 = CEC2 + NetInputAct * InputGateAct
		
		for j in range(self.cell_blocks):
			OutputGateSum[j] += self.peepOutputGate[j] * CEC3[j]

		OutputGateAct = SigmoidActivate(OutputGateSum)
		CECSquashAct = CEC3
		NetOutputAct = CECSquashAct * OutputGateAct

		#prepare hidden layer plus bias
		full_hidden = np.zeros((self.full_hidden_dimension))
		full_hidden[0: self.cell_blocks] = NetOutputAct
		full_hidden[-1] = 1.0
		
		#calculate output
		output = np.zeros((self.output_dimension))
		for k in range(self.output_dimension):
			for j in range(self.full_hidden_dimension):
				output[k] += self.weightsGlobalOutput[k][j] * full_hidden[j]
			#output not squashed
		output = SoftmaxActivate(output)
		
		#//BACKPROP
		#scale partials
		for i in range(self.full_input_dimension):
			self.dSdwWeightsInputGate[:, i] *= ForgetGateAct
			self.dSdwWeightsForgetGate[:, i] *= ForgetGateAct
			self.dSdwWeightsNetInput[:, i] *= ForgetGateAct
			self.dSdwWeightsInputGate[:, i] += full_input[i] * SigmoidDerivative(InputGateSum) * NetInputAct
			self.dSdwWeightsForgetGate[:, i] += full_input[i] * SigmoidDerivative(ForgetGateSum) * CEC1
			self.dSdwWeightsNetInput[:, i] += full_input[i] * TanhDerivative(NetInputSum) * InputGateAct
		
		deltaGlobalOutputPre = target_output - output
		
		#output to hidden
		deltaNetOutput = np.zeros((self.cell_blocks))
		for k in range(self.output_dimension):
			deltaNetOutput += deltaGlobalOutputPre[k] * self.weightsGlobalOutput[k][:-1]
			self.weightsGlobalOutput[k, :-1] += deltaGlobalOutputPre[k] * NetOutputAct * self.learningRate
		self.weightsGlobalOutput[:, self.cell_blocks] += deltaGlobalOutputPre * 1.0 * self.learningRate
		
		
		for c in range(self.cell_blocks):
			#update output gates
			deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c]
			deltaOutputGatePre = SigmoidDerivative(OutputGateSum[c]) * deltaOutputGatePost
			self.weightsOutputGate[c] += full_input * deltaOutputGatePre * self.learningRate
			self.peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * self.learningRate
			#before outgate
			deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c]

			#update input gates
			deltaInputGatePost = deltaCEC3 * NetInputAct[c]
			deltaInputGatePre = SigmoidDerivative(InputGateSum[c]) * deltaInputGatePost
			self.weightsInputGate[c] += self.dSdwWeightsInputGate[c] * deltaCEC3 * self.learningRate
			self.peepInputGate[c] += CEC2[c] * deltaInputGatePre * self.learningRate

			#before ingate
			deltaCEC2 = deltaCEC3

			#update forget gates
			deltaForgetGatePost = deltaCEC2 * CEC1[c]
			deltaForgetGatePre = SigmoidDerivative(ForgetGateSum[c]) * deltaForgetGatePost
			self.weightsForgetGate[c] += self.dSdwWeightsForgetGate[c] * deltaCEC2 * self.learningRate
			self.peepForgetGate[c] += CEC1[c] * deltaForgetGatePre * self.learningRate

			#update cell inputs
			self.weightsNetInput[c] += self.dSdwWeightsNetInput[c] * deltaCEC3 * self.learningRate
			#no peeps for cell inputs
		
		#//roll-over context to next time step
		self.context = NetOutputAct
		self.CEC = CEC3

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
			NetInputAct[j] = TanhActivate(NetInputSum[j])
			ForgetGateSum[j] += self.peepForgetGate[j] * CEC1[j]
			ForgetGateAct[j] = SigmoidActivate(ForgetGateSum[j])
			CEC2[j] = CEC1[j] * ForgetGateAct[j]
			InputGateSum[j] += self.peepInputGate[j] * CEC2[j]
			InputGateAct[j] = SigmoidActivate(InputGateSum[j])
			CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]
			OutputGateSum[j] += self.peepOutputGate[j] * CEC3[j]
			OutputGateAct[j] = SigmoidActivate(OutputGateSum[j])
			#CECSquashAct[j] = self.neuronCECSquash.Activate(CEC3[j])
			CECSquashAct[j] = CEC3[j]
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
		output = SoftmaxActivate(output)
		#give results
		return output




