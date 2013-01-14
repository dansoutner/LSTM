# -*- coding:utf-8 -*-
"""
Simple implementation of LSTM RNN,
proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

dsoutner@kky.zcu.cz, 2013

"""
__version__ = "0.2"

import numpy as np
cimport numpy as np
import random
cimport cython
import time
t = time.time

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline np.ndarray SoftmaxActivate(np.ndarray x):
	cdef np.ndarray e = np.exp(x)
	return (e / np.sum(e))

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline np.ndarray SoftmaxDerivative(np.ndarray x):
	cdef np.ndarray m = SoftmaxActivate(x)
	return m - np.power(m, 2)

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline np.ndarray TanhActivate(np.ndarray x):
	return np.tanh(x)

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline float TanhActivate_f(float x):
	return np.tanh(x)

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline np.ndarray TanhDerivative(np.ndarray x):
	cdef np.ndarray coshx = np.cosh(x)
	cdef np.ndarray denom = (np.cosh(2 * x) + 1)
	return 4 * coshx * coshx / (denom * denom)

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline np.ndarray SigmoidActivate(np.ndarray x):
	return 1 / (1 + np.exp(-x))

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline np.ndarray SigmoidDerivative(np.ndarray x):
	cdef np.ndarray act = 1 / (1 + np.exp(-x))
	return act * (1 - act)

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
cdef inline float SigmoidDerivative_f(float x):
	cdef float act = 1 / (1 + np.exp(-x))
	return act * (1 - act)

print "Using Cython module. "

npzeros = np.zeros


class LSTM: 

	def __init__(self, input_dimension, output_dimension, cell_blocks, rnd_seed):

		cdef double init_weight_range = 0.1
		cdef int biasInputGate = 2
		cdef int biasForgetGate = -2
		cdef int biasOutputGate = 2
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

	@cython.boundscheck(False) # turn of bounds-checking for entire function
	@cython.wraparound(False)
	def Next(self, np.ndarray[DTYPE_t, ndim=1] input, np.ndarray[DTYPE_t, ndim=1] target_output, learningRate=None):
		
		if not learningRate == None:
			self.learningRate = learningRate
		
		cdef int cell_blocks = self.cell_blocks
		cdef int full_input_dimension = self.full_input_dimension
		cdef int i, j, k
		
		
		#setup input vector
		cdef np.ndarray full_input = np.zeros((full_input_dimension), dtype=DTYPE)
		full_input[0:len(input)] = input
		full_input[len(input):len(self.context)+len(input)] = self.context
		full_input[-1] = 1.0
		
		
		
		#cell block arrays
		cdef np.ndarray[DTYPE_t, ndim=1] NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] NetInputAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] InputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] ForgetGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] OutputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] CECSquashAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] NetOutputAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] CEC1 = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] CEC2 = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] CEC3 = np.zeros((cell_blocks), dtype=DTYPE)

		
		for i in range(full_input_dimension):
			NetInputSum += self.weightsNetInput[:, i] * full_input[i]
			InputGateSum += self.weightsInputGate[:, i] * full_input[i]
			ForgetGateSum += self.weightsForgetGate[:, i] * full_input[i]
			OutputGateSum += self.weightsOutputGate[:, i] * full_input[i]
		
		#internals of cell blocks
		CEC1 = self.CEC
		NetInputAct = TanhActivate(NetInputSum)
		for j in range(cell_blocks):
			ForgetGateSum[j] += self.peepForgetGate[j] * CEC1[j]

		ForgetGateAct = SigmoidActivate(ForgetGateSum)
		CEC2 = CEC1 * ForgetGateAct

		for j in range(cell_blocks):
			InputGateSum[j] += self.peepInputGate[j] * CEC2[j]
		
		InputGateAct = SigmoidActivate(InputGateSum)
		CEC3 = CEC2 + NetInputAct * InputGateAct
		
		for j in range(cell_blocks):
			OutputGateSum[j] += self.peepOutputGate[j] * CEC3[j]

		OutputGateAct = SigmoidActivate(OutputGateSum)
		CECSquashAct = CEC3
		NetOutputAct = CECSquashAct * OutputGateAct

		#prepare hidden layer plus bias
		full_hidden = np.zeros((self.full_hidden_dimension))
		full_hidden[0: cell_blocks] = NetOutputAct
		full_hidden[-1] = 1.0
		
		#calculate output
		cdef np.ndarray[DTYPE_t, ndim=1] output = np.zeros((self.output_dimension), dtype=DTYPE)
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
		cdef np.ndarray deltaNetOutput = np.zeros((cell_blocks))
		for k in range(self.output_dimension):
			deltaNetOutput += deltaGlobalOutputPre[k] * self.weightsGlobalOutput[k][:-1]
			self.weightsGlobalOutput[k, :-1] += deltaGlobalOutputPre[k] * NetOutputAct * self.learningRate
		self.weightsGlobalOutput[:, cell_blocks] += deltaGlobalOutputPre * 1.0 * self.learningRate
		
		
		for c in range(cell_blocks):
			#update output gates
			deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c]
			deltaOutputGatePre = SigmoidDerivative_f(OutputGateSum[c]) * deltaOutputGatePost
			self.weightsOutputGate[c] += full_input * deltaOutputGatePre * self.learningRate
			self.peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * self.learningRate
			#before outgate
			deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c]

			#update input gates
			deltaInputGatePost = deltaCEC3 * NetInputAct[c]
			deltaInputGatePre = SigmoidDerivative_f(InputGateSum[c]) * deltaInputGatePost
			self.weightsInputGate[c] += self.dSdwWeightsInputGate[c] * deltaCEC3 * self.learningRate
			self.peepInputGate[c] += CEC2[c] * deltaInputGatePre * self.learningRate

			#before ingate
			deltaCEC2 = deltaCEC3

			#update forget gates
			deltaForgetGatePost = deltaCEC2 * CEC1[c]
			deltaForgetGatePre = SigmoidDerivative_f(ForgetGateSum[c]) * deltaForgetGatePost
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
	
	@cython.boundscheck(False) # turn of bounds-checking for entire function
	def forward(self, np.ndarray input):
		#setup input vector
		cdef np.ndarray full_input = np.zeros((self.full_input_dimension))
		full_input[0:len(input)] = input
		full_input[len(input):len(self.context)+len(input)] = self.context
		full_input[-1] = 1.0
		
		
		#cell block arrays
		cdef np.ndarray NetInputSum = np.zeros((self.cell_blocks))
		cdef np.ndarray InputGateSum = np.zeros((self.cell_blocks))
		cdef np.ndarray ForgetGateSum = np.zeros((self.cell_blocks))
		cdef np.ndarray OutputGateSum = np.zeros((self.cell_blocks))
		cdef np.ndarray NetInputAct = np.zeros((self.cell_blocks))
		cdef np.ndarray InputGateAct = np.zeros((self.cell_blocks))
		cdef np.ndarray ForgetGateAct = np.zeros((self.cell_blocks))
		cdef np.ndarray OutputGateAct = np.zeros((self.cell_blocks))
		cdef np.ndarray CECSquashAct = np.zeros((self.cell_blocks))
		cdef np.ndarray NetOutputAct = np.zeros((self.cell_blocks))
		cdef np.ndarray CEC1 = np.zeros((self.cell_blocks))
		cdef np.ndarray CEC2 = np.zeros((self.cell_blocks))
		cdef np.ndarray CEC3 = np.zeros((self.cell_blocks))

		
		for i in xrange(self.full_input_dimension):
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
		cdef np.ndarray output = np.zeros((self.output_dimension))
		for k in range(self.output_dimension):
			for j in range(self.full_hidden_dimension):
				output[k] += self.weightsGlobalOutput[k][j] * full_hidden[j]
			#output not squashed
		output = SoftmaxActivate(output)
		#give results
		return output




