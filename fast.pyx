#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#coding: utf-8

"""
Copyright (c) 2014, Daniel Soutner All rights reserved. Licensed under the 3-clause BSD.
"""

import cython
import numpy as np
cimport numpy as cnp
import math
import sys
import time

DTYPE = np.double
ctypedef cnp.double_t DTYPE_t

from cython.parallel cimport prange
cimport openmp

cdef extern from "math.h":
	double exp(double x) nogil
	double tanh(double x) nogil
	double cosh(double x) nogil
	double pow(double base, double exponent) nogil
	double log10(double x) nogil

IF UNAME_SYSNAME == "Windows":
	cdef inline double exp10(double x) nogil:
		return pow(10, x)
ELSE:
	cdef extern from "math.h":
		double exp10(double x) nogil


cdef inline double TanhActivate_f(double x) nogil:
	return tanh(x)

cdef inline double TanhDerivative_f(double x) nogil:
	cdef double coshx = cosh(x)
	cdef double denom = (cosh(2 * x) + 1)
	return 4 * coshx * coshx / (denom * denom)

cdef inline double SigmoidActivate_f(double x) nogil:
	return 1. / (1. + exp(-x))

cdef inline double SigmoidDerivative_f(double x) nogil:
	cdef double act = 1 / (1 + exp(-x))
	return act * (1 - act)

cpdef inline double exp10_f(double x) nogil:
	return exp10(x)




def FastForward(model, int word_idx, int next_word_idx, cache):

	# INIT
	cdef int cell_blocks = model.cell_blocks
	cdef int input_dimension = model.input_dimension
	cdef int output_dimension = model.output_dimension
	cdef int full_input_dimension = model.input_dimension + cell_blocks + 1
	cdef int full_hidden_dimension = cell_blocks + 1

	cdef DTYPE_t *CEC = <DTYPE_t *>(cnp.PyArray_DATA(model.CEC))
	cdef DTYPE_t *context = <DTYPE_t *>(cnp.PyArray_DATA(model.context))

	cdef DTYPE_t *weightsNetInput = <DTYPE_t *>(cnp.PyArray_DATA(model.weightsNetInput))
	cdef DTYPE_t *weightsInputGate = <DTYPE_t *>(cnp.PyArray_DATA(model.weightsInputGate))
	cdef DTYPE_t *weightsForgetGate = <DTYPE_t *>(cnp.PyArray_DATA(model.weightsForgetGate))
	cdef DTYPE_t *weightsOutputGate = <DTYPE_t *>(cnp.PyArray_DATA(model.weightsOutputGate))
	cdef DTYPE_t *weightsGlobalOutput = <DTYPE_t *>(cnp.PyArray_DATA(model.weightsGlobalOutput))

	cdef DTYPE_t *peepInputGate = <DTYPE_t *>(cnp.PyArray_DATA(model.peepInputGate))
	cdef DTYPE_t *peepForgetGate = <DTYPE_t *>(cnp.PyArray_DATA(model.peepForgetGate))
	cdef DTYPE_t *peepOutputGate = <DTYPE_t *>(cnp.PyArray_DATA(model.peepOutputGate))

	cdef double[::1] NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] NetInputAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] InputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] ForgetGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] OutputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CECSquashAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] NetOutputAct = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] CEC1 = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CEC2 = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CEC3 = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] output = np.zeros((output_dimension), dtype=DTYPE)

	cdef double[::1] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
	cdef double[::1] full_hidden = np.zeros((full_hidden_dimension), dtype=DTYPE)

	cdef unsigned int i, j, k, c, loc
	cdef double softmax_suma, softmax_val = 0.
	cdef int len_dic = model.len_dic

	num_threads = model.num_threads

	if model.output_classes:
		len_class_cn = len(model.class_cn)
		out_cl = model.vocab[next_word_idx].class_index
		out_wrd = model.vocab[next_word_idx].in_class

	input = model.index_to_vector(word_idx, cache)

	# setup input vector
	loc = 0
	for i in range(input_dimension):
		full_input[loc] = input[i]
		loc += 1
	for c in range(cell_blocks):
		full_input[loc] = context[c]
		loc += 1
	full_input[loc] = 1.        # bias

	# speed-up
	#nonzero_input_dimension = (range(word_idx, word_idx + 1) + range(len_dic, full_input_dimension))
	nonzero_input_dimension = [i for i in xrange(0, full_input_dimension) if (i==word_idx or i >= len_dic)]
	#zero_input_dimension = (range(0, word_idx) + range(word_idx + 1, len_dic))

	# Tanh layer
	#for i in prange(full_input_dimension, nogil=True, num_threads=num_threads, schedule='guided'):
	#for i in range(full_input_dimension):
	for i in nonzero_input_dimension:
		full_input[i] = TanhActivate_f(full_input[i])

	with nogil:
		# cell block arrays
		for i in range(cell_blocks):
			NetInputSum[i] = 0.
			InputGateSum[i] = 0.
			ForgetGateSum[i] = 0.
			OutputGateSum[i] = 0.

	#for i in range(full_input_dimension):
	for i in nonzero_input_dimension:
		for j in range(cell_blocks):
			#print cell_blocks * i + j
			#NetInputSum[j] += weightsNetInput[j, i] * full_input[i]
			NetInputSum[j] += weightsNetInput[full_input_dimension * j + i] * full_input[i]
			#InputGateSum[j] += weightsInputGate[j, i] * full_input[i]
			InputGateSum[j] += weightsInputGate[full_input_dimension * j + i] * full_input[i]
			#ForgetGateSum[j] += weightsForgetGate[j, i] * full_input[i]
			ForgetGateSum[j] += weightsForgetGate[full_input_dimension * j + i] * full_input[i]
			#OutputGateSum[j] += weightsOutputGate[j, i] * full_input[i]
			OutputGateSum[j] += weightsOutputGate[full_input_dimension * j + i] * full_input[i]

	with nogil:
		# internals of cell blocks
		for j in range(cell_blocks):
			CEC1[j] = CEC[j]
			NetInputAct[j] = SigmoidActivate_f(NetInputSum[j])
			ForgetGateSum[j] += peepForgetGate[j] * CEC1[j]
			ForgetGateAct[j] = SigmoidActivate_f(ForgetGateSum[j])
			CEC2[j] = CEC1[j] * ForgetGateAct[j]
			InputGateSum[j] += peepInputGate[j] * CEC2[j]
			InputGateAct[j] = SigmoidActivate_f(InputGateSum[j])
			CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]
			OutputGateSum[j] += peepOutputGate[j] * CEC3[j]
			OutputGateAct[j] = SigmoidActivate_f(OutputGateSum[j])
			CECSquashAct[j] = CEC3[j]
			NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j]

		# prepare hidden layer plus bias
		for j in range(cell_blocks):
			full_hidden[j] = NetOutputAct[j]
		full_hidden[cell_blocks] = 1.

	# calculate output
	if not model.output_classes:
		for k in range(output_dimension):
			output[k] = 0.
		for k in range(output_dimension):
			for j in range(full_hidden_dimension):
				#output[k] += weightsGlobalOutput[k, j] * full_hidden[j]
				output[k] += weightsGlobalOutput[full_hidden_dimension * k + j] * full_hidden[j]

		# SoftMax
		softmax_suma = 0.
		for k in range(output_dimension):
			softmax_val = exp(output[k])
			softmax_suma += softmax_val
			output[k] = softmax_val

		for k in range(output_dimension):
			output[k] /= softmax_suma

	else:
		# factorization
		#output = np.zeros((output_dimension))
		for k in range(output_dimension):
			output[k] = 0
		#out_cl = self.vocab[next_word_idx].class_index
		#out_wrd = self.vocab[next_word_idx].in_class

		# length of output
		M = model.class_cn[out_cl] + len_class_cn

		for k in range(len_class_cn):
			for j in range(full_hidden_dimension):
				#output[k] += weightsGlobalOutput[k, j] * full_hidden[j]
				output[k] += weightsGlobalOutput[k, j] * full_hidden[j]
		for k in range(len_class_cn, M):
			for j in range(full_hidden_dimension):
				#output[k] += weightsGlobalOutput[k, j] * full_hidden[j]
				output[k] += weightsGlobalOutput[k, j] * full_hidden[j]

		# SoftMax over classes
		softmax_suma = 0.
		for k in range(len_class_cn):
			softmax_val = exp(output[k])
			softmax_suma += softmax_val
			output[k] = softmax_val
		for k in range(len_class_cn):
			output[k] /= softmax_suma

		# SoftMax over word
		softmax_suma = 0.
		for k in range(len_class_cn, M):
			softmax_val = exp(output[k])
			softmax_suma += softmax_val
			output[k] = softmax_val
		for k in range(len_class_cn, M):
			output[k] /= softmax_suma

	# roll-over context to next time step
	for j in range(cell_blocks):
		context[j] = NetOutputAct[j]
		CEC[j] = CEC3[j]

	return output


def FastRunTrain(model, long[::1] iText, lText):
	#Main training method

	cdef double learningRate = model.alpha
	cdef int cell_blocks = model.cell_blocks
	cdef int input_dimension = model.input_dimension
	cdef int output_dimension = model.output_dimension

	# indexes
	cdef unsigned int i, j, k, c, z, loc, word = 0

	cdef double softmax_suma, softmax_val = 0.
	cdef double logp = 0.0

	cdef double deltaOutputGatePost = 0
	cdef double deltaOutputGatePre = 0
	cdef double deltaInputGatePost = 0
	cdef double deltaInputGatePre = 0
	cdef double deltaForgetGatePost = 0
	cdef double deltaForgetGatePre = 0
	cdef double deltaCEC2 = 0
	cdef double deltaCEC3 = 0

	cdef int LAST_N = model.len_cache

	cdef unsigned int len_dic = model.len_dic
	cdef unsigned int full_input_dimension = input_dimension + cell_blocks + 1
	cdef unsigned int full_hidden_dimension = cell_blocks + 1

	cdef double[::1] CEC = model.CEC
	cdef double[::1] context = model.context

	cdef double[:,::1] weightsNetInput = model.weightsNetInput
	cdef double[:,::1] weightsInputGate = model.weightsInputGate
	cdef double[:,::1] weightsForgetGate = model.weightsForgetGate
	cdef double[:,::1] weightsOutputGate = model.weightsOutputGate
	cdef double[:,::1] weightsGlobalOutput = model.weightsGlobalOutput

	cdef double[::1] peepInputGate = model.peepInputGate
	cdef double[::1] peepForgetGate = model.peepForgetGate
	cdef double[::1] peepOutputGate = model.peepOutputGate

	cdef double[::1] NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] NetInputAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] InputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] ForgetGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] OutputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CECSquashAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] NetOutputAct = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] CEC1 = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CEC2 = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CEC3 = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] deltaNetOutput = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] deltaGlobalOutputPre = np.zeros((output_dimension), dtype=DTYPE)

	cdef double[::1] output = np.zeros((output_dimension), dtype=DTYPE)

	cdef double[::1] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
	cdef double[::1] full_hidden = np.zeros((full_hidden_dimension), dtype=DTYPE)

	cdef double[:,::1] dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
	cdef double[:,::1] dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
	cdef double[:,::1] dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)

	cdef double[::1] input = np.zeros((input_dimension), dtype=DTYPE)
	cdef double[::1] target_output = np.zeros((output_dimension), dtype=DTYPE)

	cdef int num_threads = model.num_threads
	cdef long word_idx = 0
	cdef long next_word_idx = 0

	# zero-out buffers
	#for c in prange(cell_blocks, nogil=True):
	for c in range(cell_blocks):
		CEC[c] = 0.
		context[c] = 1.
	#for c in prange(cell_blocks, nogil=True):
	for c in range(cell_blocks):
		for i in range(full_input_dimension):
			dSdwWeightsNetInput[c, i] = 0.
			dSdwWeightsInputGate[c, i] = 0.
			dSdwWeightsForgetGate[c, i] = 0.

	cdef double start = time.time()
	for word in xrange(len(iText) - 1):
		# case of OOV
		word_idx = iText[word]
		next_word_idx = iText[word + 1]
		if next_word_idx < 0: # == -1
			continue

		if word % 1000 == 999:
			speed = word / (time.time() - start)
			time_to_end = time.strftime('%H:%M:%S', time.gmtime((len(iText) - word) * (1 / speed)))
			print "speed %.2f words/s, %s remaining, train ppl: %.2f, alpha: %.4f\r" % (
			speed, time_to_end, exp10(-logp / np.double(word)), learningRate),
			sys.stdout.flush()

			# if we went a wrong way
			if math.isnan(logp):
				model.Reset()
				break

		# making input vector
		cache = lText[max(0, word-LAST_N): (word + 1)]
		input = model.index_to_vector(word_idx, cache)
		# making target vector
		target_output[next_word_idx] = 1.

		#release gil
		with nogil:

			# setup input vector
			loc = 0
			for i in xrange(input_dimension):
				full_input[loc] = input[i]
				loc += 1
			for c in xrange(cell_blocks):
				full_input[loc] = context[c]
				loc += 1
			full_input[loc] = 1.        # bias

			# Tanh layer
			for i in xrange(full_input_dimension):
			#for i in prange(full_input_dimension, nogil=True, num_threads=num_threads, schedule='runtime'):
			#for i in range(full_input_dimension):
				full_input[i] = TanhActivate_f(full_input[i])

			# cell block arrays
			#for j in prange(cell_blocks, schedule='runtime', num_threads=num_threads):
			for i in xrange(cell_blocks):
				NetInputSum[i] = 0.
				InputGateSum[i] = 0.
				ForgetGateSum[i] = 0.
				OutputGateSum[i] = 0.

			#for i in range(full_input_dimension):
			for i in xrange(full_input_dimension):
				#for j in prange(cell_blocks, schedule='runtime', num_threads=num_threads):
				for j in xrange(cell_blocks):
					NetInputSum[j] += weightsNetInput[j, i] * full_input[i]
					InputGateSum[j] += weightsInputGate[j, i] * full_input[i]
					ForgetGateSum[j] += weightsForgetGate[j, i] * full_input[i]
					OutputGateSum[j] += weightsOutputGate[j, i] * full_input[i]

			# internals of cell blocks
			#for j in prange(cell_blocks, schedule='runtime', num_threads=num_threads):
			for j in xrange(cell_blocks):
				CEC1[j] = CEC[j]
				NetInputAct[j] = SigmoidActivate_f(NetInputSum[j])
				ForgetGateSum[j] += peepForgetGate[j] * CEC1[j]
				ForgetGateAct[j] = SigmoidActivate_f(ForgetGateSum[j])
				CEC2[j] = CEC1[j] * ForgetGateAct[j]
				InputGateSum[j] += peepInputGate[j] * CEC2[j]
				InputGateAct[j] = SigmoidActivate_f(InputGateSum[j])
				CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]
				OutputGateSum[j] += peepOutputGate[j] * CEC3[j]
				OutputGateAct[j] = SigmoidActivate_f(OutputGateSum[j])
				#CECSquashAct[j] = CEC3[j]
				#NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j]
				NetOutputAct[j] = CEC3[j] * OutputGateAct[j]

			# prepare hidden layer plus bias
			for j in xrange(cell_blocks):
			#for j in prange(cell_blocks, schedule='runtime', num_threads=num_threads):
				full_hidden[j] = NetOutputAct[j]
			full_hidden[cell_blocks] = 1.

			# calculate output
			for k in xrange(output_dimension):
				output[k] = 0.
			for k in xrange(output_dimension):
				for j in xrange(full_hidden_dimension):
					output[k] += weightsGlobalOutput[k, j] * full_hidden[j]

			# SoftMax
			softmax_suma = 0.
			#for k in prange(output_dimension, schedule='runtime', num_threads=num_threads):
			for k in xrange(output_dimension):
				softmax_val = exp(output[k])
				softmax_suma += softmax_val
				output[k] = softmax_val

			for k in xrange(output_dimension):
				output[k] /= softmax_suma

			logp += log10(output[next_word_idx])


			# BACKPROPAGATION PART
			# scale partials
			for c in xrange(cell_blocks):
				for i in xrange(full_input_dimension):
					dSdwWeightsInputGate[c, i] *= ForgetGateAct[c]
					dSdwWeightsForgetGate[c, i] *= ForgetGateAct[c]
					dSdwWeightsNetInput[c, i] *= ForgetGateAct[c]
					dSdwWeightsInputGate[c, i] += full_input[i] * SigmoidDerivative_f(InputGateSum[c]) * NetInputAct[c]
					dSdwWeightsForgetGate[c, i] += full_input[i] * SigmoidDerivative_f(ForgetGateSum[c]) * CEC1[c]
					dSdwWeightsNetInput[c, i] += full_input[i] * SigmoidDerivative_f(NetInputSum[c]) * InputGateAct[c]

			for k in xrange(output_dimension):
				deltaGlobalOutputPre[k] = target_output[k] - output[k]

			#output --> hidden
			for c in xrange(cell_blocks):
				deltaNetOutput[c] = 0.

			for k in xrange(output_dimension):
				for c in xrange(cell_blocks):
					deltaNetOutput[c] += deltaGlobalOutputPre[k] * weightsGlobalOutput[k, c]
					weightsGlobalOutput[k, c] += deltaGlobalOutputPre[k] * NetOutputAct[c] * learningRate
				#bias
				weightsGlobalOutput[k, cell_blocks] += deltaGlobalOutputPre[k] * learningRate

			#hidden
			for c in xrange(cell_blocks):
				#update output gates
				#deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c]
				deltaOutputGatePost = deltaNetOutput[c] * CEC3[c]
				deltaOutputGatePre = SigmoidDerivative_f(OutputGateSum[c]) * deltaOutputGatePost

				for i in xrange(full_input_dimension):
				#for i in range(full_input_dimension):
					weightsOutputGate[c, i] += full_input[i] * deltaOutputGatePre * learningRate

				peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * learningRate
				#before outgate
				deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c] * CEC3[c]

				#update input gates
				deltaInputGatePost = deltaCEC3 * NetInputAct[c]
				deltaInputGatePre = SigmoidDerivative_f(InputGateSum[c]) * deltaInputGatePost
				for i in xrange(full_input_dimension):
					weightsInputGate[c, i] += dSdwWeightsInputGate[c, i] * deltaCEC3 * learningRate
				peepInputGate[c] += CEC2[c] * deltaInputGatePre * learningRate

				#before ingate
				deltaCEC2 = deltaCEC3

				#update forget gates
				deltaForgetGatePost = deltaCEC2 * CEC1[c]
				deltaForgetGatePre = SigmoidDerivative_f(ForgetGateSum[c]) * deltaForgetGatePost
				for i in xrange(full_input_dimension):
					weightsForgetGate[c, i] += dSdwWeightsForgetGate[c, i] * deltaCEC2 * learningRate
				peepForgetGate[c] += CEC1[c] * deltaForgetGatePre * learningRate

				#update cell inputs
				for i in xrange(full_input_dimension):
					weightsNetInput[c, i] += dSdwWeightsNetInput[c, i] * deltaCEC3 * learningRate

			#roll-over context to next time step
			#for j in prange(cell_blocks, schedule='runtime', num_threads=num_threads):
			for j in xrange(cell_blocks):
				context[j] = NetOutputAct[j]
				CEC[j] = CEC3[j]

		if next_word_idx == 0 and model.independent: # if reached end of sentence </s>
			#for c in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
			for c in xrange(cell_blocks):
				CEC[c] = 0.
				context[c] = 1.

		# to zero out target vector
		target_output[next_word_idx] = 0.

	model.CEC = np.asarray(CEC)
	model.context = np.asarray(context)

	model.weightsNetInput = np.asarray(weightsNetInput)
	model.weightsInputGate = np.asarray(weightsInputGate)
	model.weightsForgetGate = np.asarray(weightsForgetGate)
	model.weightsOutputGate = np.asarray(weightsOutputGate)
	model.weightsGlobalOutput = np.asarray(weightsGlobalOutput)

	model.peepInputGate = np.asarray(peepInputGate)
	model.peepForgetGate = np.asarray(peepForgetGate)
	model.peepOutputGate = np.asarray(peepOutputGate)


def FastRunTrain_N(model, long[::1] iText):
	#Main training method

	cdef double learningRate = model.alpha
	cdef int cell_blocks = model.cell_blocks
	cdef int input_dimension = model.input_dimension
	cdef int output_dimension = model.output_dimension
	cdef int independent = model.independent

	# indexes
	cdef unsigned int i, j, k, c, z, loc, word = 0

	cdef double softmax_suma, softmax_val = 0.
	cdef double logp = 0.0

	cdef double deltaOutputGatePost = 0
	cdef double deltaOutputGatePre = 0
	cdef double deltaInputGatePost = 0
	cdef double deltaInputGatePre = 0
	cdef double deltaForgetGatePost = 0
	cdef double deltaForgetGatePre = 0
	cdef double deltaCEC2 = 0
	cdef double deltaCEC3 = 0

	cdef unsigned int len_dic = model.len_dic
	cdef unsigned int full_input_dimension = input_dimension + cell_blocks + 1
	cdef unsigned int full_hidden_dimension = cell_blocks + 1

	cdef double[::1] CEC = model.CEC
	cdef double[::1] context = model.context

	cdef double[:,::1] weightsNetInput = model.weightsNetInput
	cdef double[:,::1] weightsInputGate = model.weightsInputGate
	cdef double[:,::1] weightsForgetGate = model.weightsForgetGate
	cdef double[:,::1] weightsOutputGate = model.weightsOutputGate
	cdef double[:,::1] weightsGlobalOutput = model.weightsGlobalOutput

	cdef double[::1] peepInputGate = model.peepInputGate
	cdef double[::1] peepForgetGate = model.peepForgetGate
	cdef double[::1] peepOutputGate = model.peepOutputGate

	cdef double[::1] NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] NetInputAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] InputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] ForgetGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] OutputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CECSquashAct = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] NetOutputAct = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] CEC1 = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CEC2 = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] CEC3 = np.zeros((cell_blocks), dtype=DTYPE)

	cdef double[::1] deltaNetOutput = np.zeros((cell_blocks), dtype=DTYPE)
	cdef double[::1] deltaGlobalOutputPre = np.zeros((output_dimension), dtype=DTYPE)

	cdef double[::1] output = np.zeros((output_dimension), dtype=DTYPE)

	cdef double[::1] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
	cdef double[::1] full_hidden = np.zeros((full_hidden_dimension), dtype=DTYPE)

	cdef double[:,::1] dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
	cdef double[:,::1] dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
	cdef double[:,::1] dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)

	cdef double[::1] input = np.zeros((input_dimension), dtype=DTYPE)
	cdef double[::1] target_output = np.zeros((output_dimension), dtype=DTYPE)

	cdef int num_threads = model.num_threads
	cdef long word_idx = 0
	cdef long next_word_idx = 0

	# zero-out buffers
	with nogil:#for c in prange(cell_blocks, nogil=True):
		for c in range(cell_blocks):
			CEC[c] = 0.
			context[c] = 1.
		#for c in prange(cell_blocks, nogil=True):
		for c in range(cell_blocks):
			for i in range(full_input_dimension):
				dSdwWeightsNetInput[c, i] = 0.
				dSdwWeightsInputGate[c, i] = 0.
				dSdwWeightsForgetGate[c, i] = 0.

	cdef double start = time.time()
	for word in xrange(len(iText) - 1):
		# case of OOV
		word_idx = iText[word]
		next_word_idx = iText[word + 1]
		if next_word_idx < 0: # == -1
			continue

		if word % 1000 == 999:
			speed = word / (time.time() - start)
			time_to_end = time.strftime('%H:%M:%S', time.gmtime((len(iText) - word) * (1 / speed)))
			print "speed %.2f words/s, %s remaining, train ppl: %.2f, alpha: %.4f\r" % (
			speed, time_to_end, exp10(-logp / np.double(word)), learningRate),
			sys.stdout.flush()

			# if we went a wrong way
			if math.isnan(logp):
				model.Reset()
				break

		# making input vector
		if word_idx > -1:
			input[word_idx] = 1.
		# making target vector
		target_output[next_word_idx] = 1.

		# setup input vector
		loc = 0
		for i in xrange(input_dimension):
			full_input[loc] = input[i]
			loc += 1
		for c in xrange(cell_blocks):
			full_input[loc] = context[c]
			loc += 1
		full_input[loc] = 1.        # bias

		# speed-up
		#nonzero_input_dimension = (range(word_idx, word_idx + 1) + range(len_dic, full_input_dimension))
		#nonzero_input_dimension = ([word_idx] + range(len_dic, full_input_dimension))
		#TODO: why this makes worse results? generator vs array
		#nonzero_input_dimension = (i for i in xrange(0, full_input_dimension) if (i == word_idx or i >= len_dic))
		#print "*"
		#print list(nonzero_input_dimension)
		#print [i for i in xrange(0, full_input_dimension) if (i==word_idx or i >= len_dic)]
		#print [word_idx] + range(len_dic, full_input_dimension)
		nonzero_input_dimension = [i for i in xrange(0, full_input_dimension) if (i==word_idx or i >= len_dic)]
		#zero_input_dimension = (range(0, word_idx) + range(word_idx + 1, len_dic))
		#zero_input_dimension = (long(i) for i in xrange(0, len_dic) if (i != word_idx))
		zero_input_dimension = [i for i in xrange(0, len_dic) if (i!=word_idx)]

		# Tanh layer
		for i in nonzero_input_dimension:
		#for i in prange(full_input_dimension, nogil=True, num_threads=num_threads, schedule='runtime'):
		#for i in range(full_input_dimension):
			full_input[i] = TanhActivate_f(full_input[i])

		# cell block arrays
		#for j in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
		for i in xrange(cell_blocks):
			NetInputSum[i] = 0.
			InputGateSum[i] = 0.
			ForgetGateSum[i] = 0.
			OutputGateSum[i] = 0.

		#for i in range(full_input_dimension):
		for i in nonzero_input_dimension:
		#for i in range(len_dic, full_input_dimension):
			#for j in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
			for j in xrange(cell_blocks):
				NetInputSum[j] += weightsNetInput[j, i] * full_input[i]
				InputGateSum[j] += weightsInputGate[j, i] * full_input[i]
				ForgetGateSum[j] += weightsForgetGate[j, i] * full_input[i]
				OutputGateSum[j] += weightsOutputGate[j, i] * full_input[i]
		"""
		i = word_idx
		#for j in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
		for j in xrange(cell_blocks):
			NetInputSum[j] += weightsNetInput[j, i] * full_input[i]
			InputGateSum[j] += weightsInputGate[j, i] * full_input[i]
			ForgetGateSum[j] += weightsForgetGate[j, i] * full_input[i]
			OutputGateSum[j] += weightsOutputGate[j, i] * full_input[i]
		"""
		with nogil:
			# internals of cell blocks
			#for j in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
			for j in xrange(cell_blocks):
				CEC1[j] = CEC[j]
				NetInputAct[j] = SigmoidActivate_f(NetInputSum[j])
				ForgetGateSum[j] += peepForgetGate[j] * CEC1[j]
				ForgetGateAct[j] = SigmoidActivate_f(ForgetGateSum[j])
				CEC2[j] = CEC1[j] * ForgetGateAct[j]
				InputGateSum[j] += peepInputGate[j] * CEC2[j]
				InputGateAct[j] = SigmoidActivate_f(InputGateSum[j])
				CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j]
				OutputGateSum[j] += peepOutputGate[j] * CEC3[j]
				OutputGateAct[j] = SigmoidActivate_f(OutputGateSum[j])
				#CECSquashAct[j] = CEC3[j]
				#NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j]
				NetOutputAct[j] = CEC3[j] * OutputGateAct[j]

			# prepare hidden layer plus bias
			for j in xrange(cell_blocks):
			#for j in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
				full_hidden[j] = NetOutputAct[j]
			full_hidden[cell_blocks] = 1.

			# calculate output
			for k in xrange(output_dimension):
				output[k] = 0.
			for k in xrange(output_dimension):
				for j in xrange(full_hidden_dimension):
					output[k] += weightsGlobalOutput[k, j] * full_hidden[j]

			# SoftMax
			softmax_suma = 0.
			#for k in prange(output_dimension, nogil=True, schedule='runtime', num_threads=num_threads):
			for k in xrange(output_dimension):
				softmax_val = exp(output[k])
				softmax_suma += softmax_val
				output[k] = softmax_val

			for k in xrange(output_dimension):
				output[k] /= softmax_suma

			logp += log10(output[next_word_idx])


		# BACKPROPAGATION PART
		# scale partials
		for c in xrange(cell_blocks):
			#for i in range(full_input_dimension):
			for i in nonzero_input_dimension:
				dSdwWeightsInputGate[c, i] *= ForgetGateAct[c]
				dSdwWeightsForgetGate[c, i] *= ForgetGateAct[c]
				dSdwWeightsNetInput[c, i] *= ForgetGateAct[c]
				dSdwWeightsInputGate[c, i] += full_input[i] * SigmoidDerivative_f(InputGateSum[c]) * NetInputAct[c]
				dSdwWeightsForgetGate[c, i] += full_input[i] * SigmoidDerivative_f(ForgetGateSum[c]) * CEC1[c]
				dSdwWeightsNetInput[c, i] += full_input[i] * SigmoidDerivative_f(NetInputSum[c]) * InputGateAct[c]
			for i in zero_input_dimension:
				dSdwWeightsInputGate[c, i] *= ForgetGateAct[c]
				dSdwWeightsForgetGate[c, i] *= ForgetGateAct[c]
				dSdwWeightsNetInput[c, i] *= ForgetGateAct[c]

		for k in xrange(output_dimension):
			deltaGlobalOutputPre[k] = target_output[k] - output[k]

		#output --> hidden
		for c in xrange(cell_blocks):
			deltaNetOutput[c] = 0.

		for k in xrange(output_dimension):
			for c in xrange(cell_blocks):
				deltaNetOutput[c] += deltaGlobalOutputPre[k] * weightsGlobalOutput[k, c]
				weightsGlobalOutput[k, c] += deltaGlobalOutputPre[k] * NetOutputAct[c] * learningRate
			#bias
			weightsGlobalOutput[k, cell_blocks] += deltaGlobalOutputPre[k] * learningRate

		#hidden
		for c in xrange(cell_blocks):
			#update output gates
			#deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c]
			deltaOutputGatePost = deltaNetOutput[c] * CEC3[c]
			deltaOutputGatePre = SigmoidDerivative_f(OutputGateSum[c]) * deltaOutputGatePost

			for i in nonzero_input_dimension:
			#for i in range(full_input_dimension):
				weightsOutputGate[c, i] += full_input[i] * deltaOutputGatePre * learningRate

			peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * learningRate
			#before outgate
			deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c] * CEC3[c]

			#update input gates
			deltaInputGatePost = deltaCEC3 * NetInputAct[c]
			deltaInputGatePre = SigmoidDerivative_f(InputGateSum[c]) * deltaInputGatePost
			for i in xrange(full_input_dimension):
				weightsInputGate[c, i] += dSdwWeightsInputGate[c, i] * deltaCEC3 * learningRate
			peepInputGate[c] += CEC2[c] * deltaInputGatePre * learningRate

			#before ingate
			deltaCEC2 = deltaCEC3

			#update forget gates
			deltaForgetGatePost = deltaCEC2 * CEC1[c]
			deltaForgetGatePre = SigmoidDerivative_f(ForgetGateSum[c]) * deltaForgetGatePost
			for i in xrange(full_input_dimension):
				weightsForgetGate[c, i] += dSdwWeightsForgetGate[c, i] * deltaCEC2 * learningRate
			peepForgetGate[c] += CEC1[c] * deltaForgetGatePre * learningRate

			#update cell inputs
			for i in xrange(full_input_dimension):
				weightsNetInput[c, i] += dSdwWeightsNetInput[c, i] * deltaCEC3 * learningRate

		#roll-over context to next time step
		#for j in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
		for j in xrange(cell_blocks):
			context[j] = NetOutputAct[j]
			CEC[j] = CEC3[j]

		if next_word_idx == 0 and independent: # if reached end of sentence </s>
			#for c in prange(cell_blocks, nogil=True, schedule='runtime', num_threads=num_threads):
			for c in xrange(cell_blocks):
				CEC[c] = 0.
				context[c] = 1.

		# to zero out target vector
		target_output[next_word_idx] = 0.
		if word_idx > -1:
			input[word_idx] = 0.

	model.CEC = np.asarray(CEC)
	model.context = np.asarray(context)

	model.weightsNetInput = np.asarray(weightsNetInput)
	model.weightsInputGate = np.asarray(weightsInputGate)
	model.weightsForgetGate = np.asarray(weightsForgetGate)
	model.weightsOutputGate = np.asarray(weightsOutputGate)
	model.weightsGlobalOutput = np.asarray(weightsGlobalOutput)

	model.peepInputGate = np.asarray(peepInputGate)
	model.peepForgetGate = np.asarray(peepForgetGate)
	model.peepOutputGate = np.asarray(peepOutputGate)
