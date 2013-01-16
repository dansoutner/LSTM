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

DTYPE = np.float
ctypedef np.float_t DTYPE_t

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

import LSTM
import numpy as np
import re
import math
import sys
import time


import cProfile


class LSTM: 
	

	def __init__(self):
		pass
	
	def text_file_to_list_of_words(self, input_text_file):
		sText = open(input_text_file).read()
		sText = sText.replace("\n", " </s> ")	# convert eos
		sText = re.subn("[ ]+", " ",sText)[0]		# disable more than one space
		lText = sText.lower().split()			# to lowercase
		return lText
	
	def index_to_vector(self, idx, dic_len):
		o = [0.] * dic_len
		o[idx] = 1.
		return np.array(o)
	
	
	def ppl(self, iText):
		"""
		Computes perplexity of RNN on given text (splitted)
		"""
		ppl = 0.
		count = 0
		#self.Reset()
		for i in xrange(len(iText)-1):
			o = self.forward(self.index_to_vector(iText[i], len(self.dic)))
			ppl += math.log(o[iText[i + 1]], 2)
			count += 1
		return math.pow(2, (-1.0 / count) * ppl)
	
	def Reset(self):
		self.CEC = np.zeros(self.cell_blocks)
		self.context = np.zeros(self.cell_blocks)
		self.dSdwWeightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.dSdwWeightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))
	
	
	def train(self):
		cdef int iHidden
	
		if len(sys.argv) < 3:
			train_file = "nano_train"
			test_file = "nano_test"
			iHidden = 15
		else:
			train_file = sys.argv[1]
			test_file = sys.argv[2]
			iHidden = int(sys.argv[3])
		
		print "Preparing data..."
		# make data
		lText = self.text_file_to_list_of_words(train_file)
		lTest = self.text_file_to_list_of_words(test_file)
		self.dic = sorted(list(set(lText + ["<unk>"])))
		
		
		iText = []
		for word in lText:
			try:
				iText.append(self.dic.index(word))
			except:
				iText.append(self.dic.index("<unk>"))
		
		iTest = []
		for word in lTest:
			try:
				iTest.append(self.dic.index(word))
			except:
				iTest.append(self.dic.index("<unk>"))
		
		print "Dictionary length: %d" % len(self.dic)
		print "Train text %d words" % len(iText)
		print "Test text %d words" % len(iTest)
		print "Hidden layer %d" % iHidden
		
		self.cell_blocks = iHidden
		#self.full_input_dimension
		
		self.runTrain(iText, iTest, 0, iHidden, len(self.dic), len(self.dic))
	
	@cython.boundscheck(False) # turn of bounds-checking for entire function
	@cython.wraparound(False)
	def runTrain(self, iText, iTest, int rnd_seed, int cell_blocks, int input_dimension, int output_dimension):
				
		#lstm = LSTM.LSTM(len(dic), len(dic), iHidden, rnd_seed=2)
		cdef float init_weight_range = 0.1
		cdef int biasInputGate = 2
		cdef int biasForgetGate = -2
		cdef int biasOutputGate = 2
		cdef float learningRate = 0.1
		
		cdef int i, j, k, c 
		
		random.seed(rnd_seed)

		cdef np.ndarray[DTYPE_t, ndim=1] CEC = np.zeros(cell_blocks)
		cdef np.ndarray[DTYPE_t, ndim=1] context = np.zeros(cell_blocks)

		cdef int full_input_dimension = input_dimension + cell_blocks + 1
		cdef int full_hidden_dimension = cell_blocks + 1

		cdef np.ndarray[DTYPE_t, ndim=2] weightsNetInput = np.zeros((cell_blocks, full_input_dimension))
		cdef np.ndarray[DTYPE_t, ndim=2] weightsInputGate = np.zeros((cell_blocks, full_input_dimension))
		cdef np.ndarray[DTYPE_t, ndim=2] weightsForgetGate = np.zeros((cell_blocks, full_input_dimension))
		cdef np.ndarray[DTYPE_t, ndim=2] weightsOutputGate = np.zeros((cell_blocks, full_input_dimension))

		cdef np.ndarray[DTYPE_t, ndim=2] dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension))
		cdef np.ndarray[DTYPE_t, ndim=2] dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension))
		cdef np.ndarray[DTYPE_t, ndim=2] dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension))
		
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
		cdef np.ndarray[DTYPE_t, ndim=1] output = np.zeros((output_dimension), dtype=DTYPE)
		
		cdef np.ndarray[DTYPE_t, ndim=1] deltaNetOutput = np.zeros((cell_blocks))
		cdef np.ndarray[DTYPE_t, ndim=1] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
		
		cdef np.ndarray[DTYPE_t, ndim=1] full_hidden = np.zeros((full_hidden_dimension))
		
		cdef np.ndarray[DTYPE_t, ndim=1] deltaGlobalOutputPre = np.zeros((output_dimension), dtype=DTYPE)
		
		cdef np.ndarray[DTYPE_t, ndim=1] peepInputGate = np.zeros(cell_blocks)
		cdef np.ndarray[DTYPE_t, ndim=1] peepForgetGate = np.zeros(cell_blocks)
		cdef np.ndarray[DTYPE_t, ndim=1] peepOutputGate = np.zeros(cell_blocks)
		
		cdef np.ndarray[DTYPE_t, ndim=2] weightsGlobalOutput = np.zeros((output_dimension, full_hidden_dimension))
		
		# random init
		for i in range(full_input_dimension):
			for j in range(cell_blocks):
				weightsNetInput[j][i] = (random.random() * 2 - 1) * init_weight_range
				weightsInputGate[j][i] = (random.random() * 2 - 1) * init_weight_range
				weightsForgetGate[j][i] = (random.random() * 2 - 1) * init_weight_range
				weightsOutputGate[j][i] = (random.random() * 2 - 1) * init_weight_range

		for j in range(cell_blocks):
			weightsInputGate[j][full_input_dimension-1] += biasInputGate
			weightsForgetGate[j][full_input_dimension-1] += biasForgetGate
			weightsOutputGate[j][full_input_dimension-1] += biasOutputGate

		for j in range(cell_blocks):
			peepInputGate[j] = (random.random() * 2 - 1) * init_weight_range
			peepForgetGate[j] = (random.random() * 2 - 1) * init_weight_range
			peepOutputGate[j] = (random.random() * 2 - 1) * init_weight_range

		for j in range(full_hidden_dimension):
			for k in range(output_dimension):
				weightsGlobalOutput[k][j] = (random.random() * 2 - 1) * init_weight_range
		
		dic = self.dic
		cdef int len_dic = len(dic)
		p = [10000]
		
		cdef int iter = 0
		while True:
			start = time.time()
			for i in range(len(iText) - 1):
				if i % 10 == 1:
					print "speed %.2f words/s" % (i / (time.time() - start))
				input = self.index_to_vector(iText[i], len_dic)
				target_output = self.index_to_vector(iText[i + 1], len_dic)
				
				###############
				full_input[0:len(input)] = input
				full_input[len(input):len(context)+len(input)] = context
				full_input[-1] = 1.0
				
				#cell block arrays
				NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
				InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
				ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
				OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
				NetInputAct = np.zeros((cell_blocks), dtype=DTYPE)
				InputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
				ForgetGateAct = np.zeros((cell_blocks), dtype=DTYPE)
				OutputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
				CECSquashAct = np.zeros((cell_blocks), dtype=DTYPE)
				NetOutputAct = np.zeros((cell_blocks), dtype=DTYPE)
				CEC1 = np.zeros((cell_blocks), dtype=DTYPE)
				CEC2 = np.zeros((cell_blocks), dtype=DTYPE)
				CEC3 = np.zeros((cell_blocks), dtype=DTYPE)
				
				for i in range(full_input_dimension):
					NetInputSum += weightsNetInput[:, i] * full_input[i]
					InputGateSum += weightsInputGate[:, i] * full_input[i]
					ForgetGateSum += weightsForgetGate[:, i] * full_input[i]
					OutputGateSum += weightsOutputGate[:, i] * full_input[i]
				
				#internals of cell blocks
				CEC1 = CEC
				NetInputAct = TanhActivate(NetInputSum)
				for j in range(cell_blocks):
					ForgetGateSum[j] += peepForgetGate[j] * CEC1[j]
				
				ForgetGateAct = SigmoidActivate(ForgetGateSum)
				CEC2 = CEC1 * ForgetGateAct
		
				for j in range(cell_blocks):
					InputGateSum[j] += peepInputGate[j] * CEC2[j]
				
				InputGateAct = SigmoidActivate(InputGateSum)
				CEC3 = CEC2 + NetInputAct * InputGateAct
				
				for j in range(cell_blocks):
					OutputGateSum[j] += peepOutputGate[j] * CEC3[j]
				
				OutputGateAct = SigmoidActivate(OutputGateSum)
				CECSquashAct = CEC3
				NetOutputAct = CECSquashAct * OutputGateAct
				
				#prepare hidden layer plus bias
				#full_hidden = np.zeros((full_hidden_dimension))
				full_hidden[0: cell_blocks] = NetOutputAct
				full_hidden[-1] = 1.0
				
				#calculate output
				#output = np.zeros((output_dimension), dtype=DTYPE)
				for k in range(output_dimension):
					for j in range(full_hidden_dimension):
						output[k] += weightsGlobalOutput[k][j] * full_hidden[j]
					#output not squashed
				output[:] = SoftmaxActivate(output[:])
				
				#//BACKPROP
				#scale partials
				for i in range(full_input_dimension):
					dSdwWeightsInputGate[:, i] *= ForgetGateAct
					dSdwWeightsForgetGate[:, i] *= ForgetGateAct
					dSdwWeightsNetInput[:, i] *= ForgetGateAct
					dSdwWeightsInputGate[:, i] += full_input[i] * SigmoidDerivative(InputGateSum) * NetInputAct
					dSdwWeightsForgetGate[:, i] += full_input[i] * SigmoidDerivative(ForgetGateSum) * CEC1
					dSdwWeightsNetInput[:, i] += full_input[i] * TanhDerivative(NetInputSum) * InputGateAct
				
				deltaGlobalOutputPre = target_output - output
				
				#output to hidden
				deltaNetOutput = np.zeros((cell_blocks))
				for k in range(output_dimension):
					deltaNetOutput += deltaGlobalOutputPre[k] * weightsGlobalOutput[k][:-1]
					weightsGlobalOutput[k, :-1] += deltaGlobalOutputPre[k] * NetOutputAct * learningRate
				weightsGlobalOutput[:, cell_blocks] += deltaGlobalOutputPre * learningRate
				
				
				for c in range(cell_blocks):
					#update output gates
					deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c]
					deltaOutputGatePre = SigmoidDerivative_f(OutputGateSum[c]) * deltaOutputGatePost
					weightsOutputGate[c] += full_input * deltaOutputGatePre * learningRate
					peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * learningRate
					#before outgate
					deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c]
		
					#update input gates
					deltaInputGatePost = deltaCEC3 * NetInputAct[c]
					deltaInputGatePre = SigmoidDerivative_f(InputGateSum[c]) * deltaInputGatePost
					weightsInputGate[c] += dSdwWeightsInputGate[c] * deltaCEC3 * learningRate
					peepInputGate[c] += CEC2[c] * deltaInputGatePre * learningRate
		
					#before ingate
					deltaCEC2 = deltaCEC3
		
					#update forget gates
					deltaForgetGatePost = deltaCEC2 * CEC1[c]
					deltaForgetGatePre = SigmoidDerivative_f(ForgetGateSum[c]) * deltaForgetGatePost
					weightsForgetGate[c] += dSdwWeightsForgetGate[c] * deltaCEC2 * learningRate
					peepForgetGate[c] += CEC1[c] * deltaForgetGatePre * learningRate
		
					#update cell inputs
					weightsNetInput[c] += dSdwWeightsNetInput[c] * deltaCEC3 * learningRate
					#no peeps for cell inputs
				
				#//roll-over context to next time step
				context = NetOutputAct
				CEC = CEC3
		
				########
			
			self.CEC = CEC
			self.context = context

			#self.Reset()
			p.append(self.ppl(iTest))
			print "%d, speed %.2f words/s, ppl: %.2f, alpha %.5f" % (iter, i / (time.time() - start), p[-1], learningRate)
			iter += 1
			
			print "#"
			
			if p[-1]/p[-2] > 0.95:
				learningRate /= 1.8
		
		print p

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

if __name__=="main":
	l = LSTM()
	l.train()

