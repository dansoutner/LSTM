# -*- coding:utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

"""
Simple implementation of LSTM RNN,
proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

dsoutner@kky.zcu.cz, 2013

"""
__version__ = "0.3"

import numpy as np
cimport numpy as np
import random
cimport cython
import time
t = time.time
import re
import math
import sys
import cProfile

np.import_array()

cdef extern from "math.h":
	double exp(double x)
	double tanh(double x)
	double cosh(double x)
	double pow (double base, double exponent )
#from libc.math cimport exp
#from libc.math cimport sum

#DTYPE = np.float
#DTYPE = np.double
DTYPE = np.double
#ctypedef np.double_t DTYPE_t
ctypedef np.double_t DTYPE_t

cdef inline float SumArray(np.ndarray[DTYPE_t, ndim=1] x):
	cdef unsigned int i = 0
	cdef float o = 0
	for i in range(len(x)):
		o += x[i]
	return o

cdef inline np.ndarray[DTYPE_t, ndim=1] SoftmaxActivate(np.ndarray[DTYPE_t, ndim=1] x):
	cdef np.ndarray[DTYPE_t, ndim=1] e = np.exp(x)
	#return (e / np.sum(e))
	return (e / e.sum())
	#return (e / SumArray(e))

cdef inline np.ndarray[DTYPE_t, ndim=1] SoftmaxDerivative(np.ndarray[DTYPE_t, ndim=1] x):
	cdef np.ndarray[DTYPE_t, ndim=1] m = SoftmaxActivate(x)
	return m - np.power(m, 2)

cdef inline np.ndarray[DTYPE_t, ndim=1] TanhActivate(np.ndarray[DTYPE_t, ndim=1] x):
	return np.tanh(x)

cdef inline double TanhActivate_f(double x):
	return tanh(x)

cdef inline np.ndarray[DTYPE_t, ndim=1] TanhDerivative(np.ndarray[DTYPE_t, ndim=1] x):
	cdef np.ndarray[DTYPE_t, ndim=1] coshx = np.cosh(x)
	cdef np.ndarray[DTYPE_t, ndim=1] denom = (np.cosh(2 * x) + 1)
	return 4 * coshx * coshx / (denom * denom)

cdef inline double TanhDerivative_f(double x):
	cdef double coshx = cosh(x)
	cdef double denom = (cosh(2 * x) + 1)
	return 4 * coshx * coshx / (denom * denom)

cdef inline np.ndarray[DTYPE_t, ndim=1] SigmoidActivate(np.ndarray[DTYPE_t, ndim=1] x):
	return 1 / (1 + np.exp(-x))
	
cdef inline double SigmoidActivate_f(double x):
	return 1 / (1 + exp(-x))

cdef inline np.ndarray[DTYPE_t, ndim=1] SigmoidDerivative(np.ndarray[DTYPE_t, ndim=1] x):
	cdef np.ndarray[DTYPE_t, ndim=1] act = 1 / (1 + np.exp(-x))
	return act * (1 - act)

cdef inline double SigmoidDerivative_f(double x):
	cdef double act = 1 / (1 + exp(-x))
	return act * (1 - act)

print "Using Cython module. "




class LSTM: 

	def text_file_to_list_of_words(self, input_text_file):
		sText = open(input_text_file).read()
		sText = sText.replace("\n", " </s> ")	# convert eos
		sText = re.subn("[ ]+", " ",sText)[0]		# disable more than one space
		lText = sText.lower().split()			# to lowercase
		return lText


	def index_to_vector(self, int idx, int dic_len):
		o = np.zeros((dic_len))
		o[idx] = 1.
		return o


	def ppl(self, np.ndarray[long, ndim=1] iText):
		"""
		Computes perplexity of RNN on given text (splitted)
		"""
		cdef float ppl = 0.
		cdef int count = 0
		self.Reset()
		for i in range(len(iText)-1):
			o = self.forward(self.index_to_vector(iText[i], len(self.dic)))
			ppl += math.log(o[iText[i + 1]], 2)
			count += 1
		return math.pow(2, (-1.0 / count) * ppl)


	def Reset(self):
		self.CEC = np.zeros(self.cell_blocks)
		self.context = np.zeros(self.cell_blocks)
		#self.dSdwWeightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		#self.dSdwWeightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		#self.dSdwWeightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))


	def __init__(self):
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
		self.input_dimension = len(self.dic)
		self.output_dimension = len(self.dic)
		self.hidden_dimension = iHidden
		self.full_hidden_dimension = self.cell_blocks + 1
		self.full_input_dimension = self.input_dimension + self.cell_blocks + 1
		
		
		self.CEC = np.zeros(self.cell_blocks)
		self.context = np.zeros(self.cell_blocks)
		
		self.peepInputGate = np.zeros(self.cell_blocks)
		self.peepForgetGate = np.zeros(self.cell_blocks)
		self.peepOutputGate = np.zeros(self.cell_blocks)
		
		self.weightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsOutputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		self.weightsGlobalOutput = np.zeros((self.output_dimension, self.full_hidden_dimension))
		
		iText = np.array(iText)
		iTest = np.array(iTest)

		self.runTrain(iText, iTest, 0, iHidden, len(self.dic), len(self.dic))


	def runTrain(self, np.ndarray[long, ndim=1] iText, np.ndarray[long, ndim=1] iTest, int rnd_seed, int cell_blocks, int input_dimension, int output_dimension):
		
		random.seed(rnd_seed)
		
		# CONST
		cdef float init_weight_range = 0.1
		cdef int biasInputGate = 2
		cdef int biasForgetGate = -2
		cdef int biasOutputGate = 2
		cdef float learningRate = 0.1
		
		# indexes
		cdef unsigned int i, j, k, c, loc 

		cdef np.ndarray[DTYPE_t, ndim=1] CEC = np.zeros(cell_blocks)
		cdef np.ndarray[DTYPE_t, ndim=1] context = np.zeros(cell_blocks)

		cdef int full_input_dimension = input_dimension + cell_blocks + 1
		cdef int full_hidden_dimension = cell_blocks + 1

		cdef np.ndarray[DTYPE_t, ndim=2] weightsNetInput = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] weightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] weightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] weightsOutputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] weightsGlobalOutput = np.zeros((output_dimension, full_hidden_dimension), dtype=DTYPE)
		
		cdef np.ndarray[DTYPE_t, ndim=2] dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		
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
		
		cdef np.ndarray[DTYPE_t, ndim=1] deltaNetOutput = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] deltaGlobalOutputPre = np.zeros((output_dimension), dtype=DTYPE)
		
		cdef np.ndarray[DTYPE_t, ndim=1] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] input = np.zeros((input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] target_output = np.zeros((output_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] full_hidden = np.zeros((full_hidden_dimension), dtype=DTYPE)
		
		cdef double deltaOutputGatePost = 0
		cdef double deltaOutputGatePre = 0
		cdef double deltaInputGatePost = 0
		cdef double deltaInputGatePre = 0
		cdef double deltaForgetGatePost = 0
		cdef double deltaForgetGatePre = 0
		cdef double deltaCEC2 = 0
		cdef double deltaCEC3 = 0
		
		cdef np.ndarray[DTYPE_t, ndim=1] peepInputGate = np.zeros(cell_blocks)
		cdef np.ndarray[DTYPE_t, ndim=1] peepForgetGate = np.zeros(cell_blocks)
		cdef np.ndarray[DTYPE_t, ndim=1] peepOutputGate = np.zeros(cell_blocks)
		
		# random init
		for i in range(full_input_dimension):
			for j in range(cell_blocks):
				weightsNetInput[j,i] = (random.random() * 2 - 1) * init_weight_range
				weightsInputGate[j,i] = (random.random() * 2 - 1) * init_weight_range
				weightsForgetGate[j,i] = (random.random() * 2 - 1) * init_weight_range
				weightsOutputGate[j,i] = (random.random() * 2 - 1) * init_weight_range

		for j in range(cell_blocks):
			weightsInputGate[j,full_input_dimension-1] += biasInputGate
			weightsForgetGate[j,full_input_dimension-1] += biasForgetGate
			weightsOutputGate[j,full_input_dimension-1] += biasOutputGate

		for j in range(cell_blocks):
			peepInputGate[j] = (random.random() * 2 - 1) * init_weight_range
			peepForgetGate[j] = (random.random() * 2 - 1) * init_weight_range
			peepOutputGate[j] = (random.random() * 2 - 1) * init_weight_range

		for j in range(full_hidden_dimension):
			for k in range(output_dimension):
				weightsGlobalOutput[k][j] = (random.random() * 2 - 1) * init_weight_range
		
		cdef int len_dic = len(self.dic)
		p = [self.ppl(iTest)]
		
		cdef int iter = 0
		while True:
			start = time.time()
			self.Reset()
			dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
			dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
			dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
			for i in range(len(iText) - 1):
				if i % 1000 == 1:
					speed = i / (time.time() - start)
					time_to_end = time.strftime('%H:%M:%S', time.gmtime((len(iText) - i) * (1/speed))) 
					print "speed %.2f words/s, %s remaining, alpha: %.5f\r" % (speed, time_to_end, learningRate),
					sys.stdout.flush()
				input = np.zeros((len_dic), dtype=DTYPE)
				input[iText[i]] = 1.
				#input = self.index_to_vector(iText[i], len_dic)
				target_output = np.zeros((len_dic), dtype=DTYPE)
				target_output[iText[i + 1]] = 1.
				#target_output = self.index_to_vector(iText[i + 1], len_dic)
				
				###############
				#setup input vector
				loc = 0
				for i in range(input_dimension):
					full_input[loc] = input[i]
					loc += 1
				for c in range(cell_blocks):
					full_input[loc] = context[c]
					loc += 1
				full_input[loc] = 1.0 #bias
				
				#cell block arrays
				NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
				InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
				ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
				OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
				
				#inputs to cell blocks
				for i in range(full_input_dimension):
					for j in range(cell_blocks):
						NetInputSum[j] += weightsNetInput[j,i] * full_input[i]
						InputGateSum[j] += weightsInputGate[j,i] * full_input[i]
						ForgetGateSum[j] += weightsForgetGate[j,i] * full_input[i]
						OutputGateSum[j] += weightsOutputGate[j,i] * full_input[i]
		
				#internals of cell blocks
				for j in range(cell_blocks):
					CEC1[j] = CEC[j]
					NetInputAct[j] = TanhActivate_f(NetInputSum[j])
					ForgetGateSum[j] += peepForgetGate[j] * CEC1[j]
					ForgetGateAct[j] = SigmoidActivate_f(ForgetGateSum[j])
					CEC2[j] = CEC1[j] * ForgetGateAct[j]
					InputGateSum[j] += peepInputGate[j] * CEC2[j]
					InputGateAct[j] = SigmoidActivate_f(InputGateSum[j])
					CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]
					OutputGateSum[j] += peepOutputGate[j] * CEC3[j] #TODO: this versus squashed?
					OutputGateAct[j] = SigmoidActivate_f(OutputGateSum[j])
					CECSquashAct[j] = CEC3[j]
					NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j]
				
				#prepare hidden layer plus bias
				loc = 0
				for j in range(cell_blocks):
					full_hidden[loc] = NetOutputAct[j]
					loc += 1
				full_hidden[loc] = 1.0		#bias
				
				#calculate output
				output = np.zeros((output_dimension))
				for k in range(output_dimension):
					for j in range(full_hidden_dimension):
						output[k] += weightsGlobalOutput[k, j] * full_hidden[j]
					#output not squashed
				output[:] = SoftmaxActivate(output[:])
				
				#//BACKPROP
				#scale partials
				#dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
				#dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
				#dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
				for c in range(cell_blocks):
					for i in range(full_input_dimension):
						dSdwWeightsInputGate[c, i] *= ForgetGateAct[c]
						dSdwWeightsForgetGate[c, i] *= ForgetGateAct[c]
						dSdwWeightsNetInput[c, i] *= ForgetGateAct[c]
		
						dSdwWeightsInputGate[c, i] += full_input[i] * SigmoidDerivative_f(InputGateSum[c]) * NetInputAct[c]
						dSdwWeightsForgetGate[c, i] += full_input[i] * SigmoidDerivative_f(ForgetGateSum[c]) * CEC1[c]
						dSdwWeightsNetInput[c, i] += full_input[i] * TanhDerivative_f(NetInputSum[c]) * InputGateAct[c]
		
				for k in range(output_dimension):
					deltaGlobalOutputPre[k] = target_output[k] - output[k]
				#deltaGlobalOutputPre = target_output - output
				
				#output to hidden
				deltaNetOutput = np.zeros((cell_blocks))
				for k in range(output_dimension):
					#links
					for c in range(cell_blocks):
						deltaNetOutput[c] += deltaGlobalOutputPre[k] * weightsGlobalOutput[k,c]
						weightsGlobalOutput[k, c] += deltaGlobalOutputPre[k] * NetOutputAct[c] * learningRate
					#bias
					weightsGlobalOutput[k, cell_blocks] += deltaGlobalOutputPre[k] * learningRate
				
				for c in range(cell_blocks):
					#update output gates
					deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c]
					deltaOutputGatePre = SigmoidDerivative_f(OutputGateSum[c]) * deltaOutputGatePost
					for i in range(full_input_dimension):
						weightsOutputGate[c, i] += full_input[i] * deltaOutputGatePre * learningRate
					peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * learningRate
					#before outgate
					deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c] * CEC3[c]
		
					#update input gates
					deltaInputGatePost = deltaCEC3 * NetInputAct[c]
					deltaInputGatePre = SigmoidDerivative_f(InputGateSum[c]) * deltaInputGatePost
					for i in range(full_input_dimension):
						weightsInputGate[c, i] += dSdwWeightsInputGate[c, i] * deltaCEC3 * learningRate
					peepInputGate[c] += CEC2[c] * deltaInputGatePre * learningRate
		
					#before ingate
					deltaCEC2 = deltaCEC3
		
					#update forget gates
					deltaForgetGatePost = deltaCEC2 * CEC1[c]
					deltaForgetGatePre = SigmoidDerivative_f(ForgetGateSum[c]) * deltaForgetGatePost
					for i in range(full_input_dimension):
						weightsForgetGate[c, i] += dSdwWeightsForgetGate[c, i] * deltaCEC2 * learningRate
					peepForgetGate[c] += CEC1[c] * deltaForgetGatePre * learningRate
		
					#update cell inputs
					for i in range(full_input_dimension):
						weightsNetInput[c, i] += dSdwWeightsNetInput[c, i] * deltaCEC3 * learningRate
					#no peeps for cell inputs
		
				#//roll-over context to next time step
				for j in range(cell_blocks):
					context[j] = NetOutputAct[j]
					CEC[j] = CEC3[j]
				
				#print CEC
				########
			
			# save to net
			self.CEC = CEC
			self.context = context
			
			self.peepForgetGate = peepForgetGate
			self.peepInputGate = peepInputGate
			self.peepOutputGate = peepOutputGate
			
			self.weightsForgetGate = weightsForgetGate
			self.weightsInputGate = weightsInputGate
			self.weightsOutputGate = weightsOutputGate
			self.weightsGlobalOutput = weightsGlobalOutput
			self.weightsNetInput = weightsNetInput
			
			p.append(self.ppl(iTest))
			print "                                                                                                              \r",
			print "%d, speed %.2f words/s, ppl: %.2f, alpha %.5f" % (iter, i / (time.time() - start), p[len(p)-1], learningRate)
			iter += 1
			if p[len(p)-1]/p[len(p)-2] > 0.92:
				learningRate /= 1.5
		print p


	def forward(self, np.ndarray input):
	
		# INIT
		cdef np.ndarray[DTYPE_t, ndim=1] CEC = self.CEC
		cdef np.ndarray[DTYPE_t, ndim=1] context = self.context
		cdef int cell_blocks = self.cell_blocks
		cdef int input_dimension = self.input_dimension
		cdef int output_dimension = self.output_dimension
		
		cdef int full_input_dimension = input_dimension + cell_blocks + 1
		cdef int full_hidden_dimension = cell_blocks + 1

		cdef np.ndarray[DTYPE_t, ndim=2] weightsNetInput = self.weightsNetInput
		cdef np.ndarray[DTYPE_t, ndim=2] weightsInputGate = self.weightsInputGate
		cdef np.ndarray[DTYPE_t, ndim=2] weightsForgetGate = self.weightsForgetGate
		cdef np.ndarray[DTYPE_t, ndim=2] weightsOutputGate = self.weightsOutputGate
		
		cdef np.ndarray[DTYPE_t, ndim=1] peepInputGate = self.peepInputGate
		cdef np.ndarray[DTYPE_t, ndim=1] peepForgetGate = self.peepForgetGate
		cdef np.ndarray[DTYPE_t, ndim=1] peepOutputGate = self.peepOutputGate
		
		cdef np.ndarray[DTYPE_t, ndim=2] weightsGlobalOutput = self.weightsGlobalOutput

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
		
		cdef np.ndarray[DTYPE_t, ndim=1] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] full_hidden = np.zeros((full_hidden_dimension))
		

		
		cdef int len_dic = len(self.dic)
		cdef unsigned int i, j, k, c, loc 
		
		#setup input vector
		loc = 0
		for i in range(input_dimension):
			full_input[loc] = input[i]
			loc += 1
		for c in range(cell_blocks):
			full_input[loc] = context[c]
			loc += 1
		full_input[loc] = 1.0 #bias
		
		#inputs to cell blocks
		for i in range(full_input_dimension):
			for j in range(cell_blocks):
				NetInputSum[j] += weightsNetInput[j,i] * full_input[i]
				InputGateSum[j] += weightsInputGate[j,i] * full_input[i]
				ForgetGateSum[j] += weightsForgetGate[j,i] * full_input[i]
				OutputGateSum[j] += weightsOutputGate[j,i] * full_input[i]

		#internals of cell blocks
		for j in range(cell_blocks):
			CEC1[j] = CEC[j]
			NetInputAct[j] = TanhActivate_f(NetInputSum[j])
			ForgetGateSum[j] += peepForgetGate[j] * CEC1[j]
			ForgetGateAct[j] = SigmoidActivate_f(ForgetGateSum[j])
			CEC2[j] = CEC1[j] * ForgetGateAct[j]
			InputGateSum[j] += peepInputGate[j] * CEC2[j]
			InputGateAct[j] = SigmoidActivate_f(InputGateSum[j])
			CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]
			OutputGateSum[j] += peepOutputGate[j] * CEC3[j] #TODO: this versus squashed?
			OutputGateAct[j] = SigmoidActivate_f(OutputGateSum[j])
			CECSquashAct[j] = CEC3[j]
			NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j]
		
		#prepare hidden layer plus bias
		loc = 0
		for j in range(cell_blocks):
			full_hidden[loc] = NetOutputAct[j]
			loc += 1
		full_hidden[loc] = 1.0		#bias
		
		#calculate output
		for k in range(output_dimension):
			for j in range(full_hidden_dimension):
				output[k] += weightsGlobalOutput[k, j] * full_hidden[j]
			#output not squashed
		output[:] = SoftmaxActivate(output[:])
		
		#//roll-over context to next time step
		for j in range(cell_blocks):
			context[j] = NetOutputAct[j]
			CEC[j] = CEC3[j]
		
		self.context = context
		self.CEC = CEC
		
		return output
		
