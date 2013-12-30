# -*- coding:utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True

"""
Recurrent neural network based statistical language modelling toolkit

Based on LSTM RNN, model proposed by Jürgen Schmidhuber
http://www.idsia.ch/~juergen/

Implemented by Daniel Soutner,
Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
dsoutner@kky.zcu.cz, 2013

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

__version__ = "0.5.0"

# cython libs
import cython
cimport numpy as np
cimport cython

from cython.parallel import prange, parallel

# usual python libs
import numpy as np
import random
import time
#import re
import math
import sys
import cPickle
from operator import itemgetter
import codecs

# ARPA module
from ArpaLM import *


# numpy settings
np.set_printoptions(edgeitems=2, infstr='Inf', linewidth=75, nanstr='NaN',	precision=8, suppress=True, threshold=1000000)


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


#ctypedef np.double_t DTYPE_t
#DTYPE = cython.double
DTYPE = np.double
ctypedef cython.double DTYPE_t


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

print "LSTM LM version %s" % __version__

# CONST
UNK = "<unk>"
ENCODING = "utf-8"

class LSTM(object):
	"""Main LSTM object"""

	def __str__(self):
		o = ""
		try:
			o += "Train text %d words from %s\n" % (len(self.lText), self.train_file)
			o += "Test text %d words from %s\n" % (len(self.Test), self.test_file)
			o += "Valid text %d words from %s\n" % (len(self.lValid), self.valid_file)
		except (AttributeError, ValueError):
			pass

		o += "Input layer %d\n" % self.input_dimension
		o += "Hidden layer %d\n" % self.hidden_dimension
		o += "Output layer %d\n" % self.output_dimension

		o += "Dictionary length: %d\n" % len(self.dic)

		return o

	def __repr__(self):
		return str(self)

	def __init__(self, args):
		"""
		Parse args and initialize net
		"""

		self.len_cache = 50
		self.version = __version__
		self.debug = args.debug

		if not args.save_net:
			try:
				self.net_save_file = args.train[0] + "_" + str(args.iHidden) + "_" + "_" + str(time.time()).split(".")[0]
			except:
				self.net_save_file = args.save_net
		else:
			self.net_save_file = args.save_net

		self.net_load_file = args.load_net
		self.net_nbest_file = args.nbest_rescore

		self.independent = args.independent

		# if we are loading from file
		if self.net_load_file:
			print "Loading net from %s..." % self.net_load_file
			self.load(self.net_load_file)

		if args.train:
			self.train = True
		else:
			self.train = False

		if self.debug:
			print args

		cdef int iHidden = 0
		if args.iHidden:
			iHidden = args.iHidden

		rnd_seed = args.rnd_seed

		# set train, test and valid files
		if args.train:
			train_file = args.train[0]
			test_file = args.train[1]
			valid_file = args.train[2]

		if self.debug:
			print "Preparing data..."

		if args.train:
			# make data
			lText = self.text_file_to_list_of_words(train_file)
			lTest = self.text_file_to_list_of_words(test_file)
			lValid = self.text_file_to_list_of_words(valid_file)

			if args.vocabulary_file:
				self.dic = self.create_dic_from_file(args.vocabulary_file)
			else:
				self.dic = self.create_dic(lText)

			self.dic.remove("</s>")    # it is good to have a </s> as 0 in dic
			self.dic.insert(0, "</s>")

			iText = []
			for word in lText:
				try:
					iText.append(self.dic.index(word))
				except (IndexError, ValueError, KeyError):
					iText.append(-1)    # unk

			iTest = []
			for word in lTest:
				try:
					iTest.append(self.dic.index(word))
				except (IndexError, ValueError, KeyError):
					iTest.append(-1)    # unk

			iValid = []
			for word in lValid:
				try:
					iValid.append(self.dic.index(word))
				except (IndexError, ValueError, KeyError):
					iValid.append(-1)    # unk

			self.iText = np.array(iText)
			self.iTest = np.array(iTest)
			self.iValid = np.array(iValid)
			self.lText = lText
			self.lTest = lTest
			self.lValid = lValid

		self.len_dic = len(self.dic)

		self.srilm_file = args.srilm_file
		self.srilm_lambda = args.srilm_lambda
		self.LMW = args.lmw
		self.WIP = args.wip

		# load ARPA LM to memory
		if self.srilm_file:
			print "Loading Arpa LM from %s..." % self.srilm_file
			self.arpaLM = ArpaLM(path=self.srilm_file)
		else:
			self.arpaLM = None

		# compute input length - depends on vector type
		iInputs = len(self.dic)
		iOutputs = len(self.dic)
		self.output_classes = False

		if args.train:
			# save
			self.cell_blocks = iHidden
			self.input_dimension = iInputs
			self.output_dimension = iOutputs
			self.hidden_dimension = iHidden
			self.full_hidden_dimension = self.cell_blocks + 1
			self.full_input_dimension = self.input_dimension + self.cell_blocks + 1

		try:        # just to be sure if we loaded all varibles
			self.CEC
			self.context

			self.peepInputGate
			self.peepForgetGate
			self.peepOutputGate

			self.weightsNetInput
			self.weightsInputGate
			self.weightsForgetGate
			self.weightsOutputGate

			self.weightsGlobalOutput
			#self.weightsGlobalInput

		except (AttributeError, NameError) as e:        # if not, init net matrixes
			#print e

			self.CEC = np.zeros((self.cell_blocks), dtype=DTYPE)
			self.context = np.zeros((self.cell_blocks), dtype=DTYPE)

			self.peepInputGate = np.zeros((self.cell_blocks), dtype=DTYPE)
			self.peepForgetGate = np.zeros((self.cell_blocks), dtype=DTYPE)
			self.peepOutputGate = np.zeros((self.cell_blocks), dtype=DTYPE)

			self.weightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension), dtype=DTYPE)
			self.weightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension), dtype=DTYPE)
			self.weightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension), dtype=DTYPE)
			self.weightsOutputGate = np.zeros((self.cell_blocks, self.full_input_dimension), dtype=DTYPE)

			self.weightsGlobalOutput = np.zeros((self.output_dimension, self.full_hidden_dimension), dtype=DTYPE)

		print self

		# train, rescore or ppl?
		if args.train:
			self.alpha = args.alpha
			self.runTrain(self.iText, self.iTest, self.iValid, rnd_seed, iHidden, iInputs, iOutputs)

		if args.nbest_rescore:
			self.net_nbest_file = args.nbest_rescore
			self.nbest_rescore(self.net_nbest_file)

		# compute PPL?
		if args.ppl_file:
			iTextPPL = []
			lTextPPL = self.text_file_to_list_of_words(args.ppl_file)
			for word in lTextPPL:
				try:
					iTextPPL.append(self.dic.index(word))
				except (IndexError, ValueError, KeyError):
					iTextPPL.append(-1)    # unk
			iTextPPL = np.array(iTextPPL, dtype=long)
			if not self.arpaLM:
				print "File: %s, PPL:%.2f" % (args.ppl_file, self.ppl(iTextPPL, lTextPPL)[0])
			else:
				print "File: %s, PPL:%.2f" % (args.ppl_file, self.ppl_combine(iTextPPL, lTextPPL))


	def create_dic(self, lText):
		vocab = {}
		vocab["</s>"] = True
		cdef int i
		for i in range(len(lText)):
			vocab[lText[i]] = True
		return sorted(vocab.keys())


	def create_dic_from_file(self, vocab_file):
		vocab = {}
		vocab["</s>"] = True
		with open(vocab_file) as f:
			for line in f:
				w = line.strip()
				if len(w) > 0:
					vocab[w] = True
		return sorted(vocab.keys())


	def text_file_to_list_of_words(self, input_text_file):
		"""
		Makes from text in input_text_file a list of words, appends </s>
		"""
		lText = []
		try:
			with codecs.open(input_text_file, "r", ENCODING) as f:
				for line in f:
					sText = line.replace("\n", " </s> ")        # convert eos
					#sText = re.subn("[ ]+", " ", sText)[0]        # disable more than one space
					#lText += sText.lower().split()                # to lowercase
					lText += sText.split()                # to lowercase
		except IOError:
			print "File %s not found." % input_text_file
		return lText


	def index_to_vector(self, int idx, cache):
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] o = np.zeros((self.len_dic), dtype=DTYPE)
		if idx > -1:
			o[idx] = 1.
		return o


	#def ppl(self, np.ndarray[long, ndim=1] iText, lText):
	def ppl(self, iText, lText):
		"""
		Computes perplexity of RNN on given text (splitted)
		"""
		cdef double logp = 0.
		cdef int count = 0
		cdef int oov_net = 0

		self.Reset()
		for word_idx in range(len(iText) - 1):
			if iText[word_idx + 1] > -1:
				cache = lText[word_idx - self.len_cache: word_idx + 1]
				vector = self.index_to_vector(iText[word_idx], cache)
				o = self.forward(iText[word_idx], iText[word_idx + 1], cache)
				logp += log10(o[iText[word_idx + 1]])
				count += 1
			else:
				oov_net += 1
				pass

			if iText[word_idx + 1] == 0 and self.independent:
				self.CEC = np.zeros(self.cell_blocks)
				self.context = np.ones(self.cell_blocks)

		if self.debug:
			print "Net OOVs %d" % oov_net

		return exp10(-logp / count), logp


	def ppl_combine(self, iText, lText):
		"""
		Computes perplexity of RNN on given text (splitted)
		"""

		cdef double logp_net = 0.
		cdef double logp_sri = 0.
		cdef double logp_combine = 0.
		cdef int oov_combine = 0
		cdef int oov_sri = 0
		cdef int oov_net = 0
		cdef int count = 0
		cdef double srilm_lambda = self.srilm_lambda
		cdef double LOG10TOLOG = np.log(10)
		cdef double LOGTOLOG10 = 1. / LOG10TOLOG
		cdef double oov_penalty = -0           #some ad hoc penalty - when mixing different vocabularies, single model score is not real PPL
		cdef double p_sri = 0.
		cdef double p_net = 0.
		self.Reset()

		for word_idx in range(len(iText) - 1):

			ctx = lText[max(0, word_idx - self.len_cache): word_idx + 2] # last xx words
			ctx = ctx[::-1]

			# ARPA takes only the last sentence
			if "</s>" in ctx:
				idx = ctx.index("</s>")
				if idx == 0:
					try:
						idx = ctx[1:].index("</s>")
					except ValueError:
						idx = len(ctx) - 1
				ctx = ctx[:idx + 1]
			if len(ctx) == 2 and ctx[len(ctx) - 1] == "</s>":
				ctx.insert(1, "<s>")

			l_sri = LOGTOLOG10 * self.arpaLM.prob(*ctx)
			p_sri = exp10(l_sri)

			if (iText[word_idx + 1] > -1) or (l_sri > -98):   # not NetOOV or not SriOOV
				if iText[word_idx + 1] == -1:                 # if NetOOV
					logp_net += oov_penalty
					#logp_combine += log10(0 * (1. - srilm_lambda) + p_sri * srilm_lambda)
					logp_combine += log10(p_sri * srilm_lambda)
				else:                                   # if not NetOOV
					cache = lText[word_idx - self.len_cache: word_idx + 1]
					o = self.forward(iText[word_idx], iText[word_idx + 1], cache)
					p_net = o[iText[word_idx + 1]]
					logp_net += log10(p_net)
					logp_combine += log10((p_net * (1. - srilm_lambda)) + (p_sri * (srilm_lambda)))

				if l_sri > -98:                         # if not SriOOV
					logp_sri += l_sri

				count += 1

			else:
				oov_combine += 1

			# count OOVs
			if iText[word_idx + 1] == -1:
				oov_net += 1
			if l_sri < -98:
				oov_sri += 1

			#!
			if iText[word_idx] == -1:
				print "*"
				print lText[word_idx]
				print lText[word_idx + 1]
				print o[np.argmax(o)]
				print self.dic[np.argmax(o)]
				print p_net
				print p_sri
			#!

			if iText[word_idx + 1] == 0 and self.independent:
				self.CEC = np.zeros(self.cell_blocks)
				self.context = np.ones(self.cell_blocks)

		print "PPL Net: %.2f" % exp10(-logp_net / count)
		print "OOV Net: %d" % oov_net
		print "PPL SRI: %.2f" % exp10(-logp_sri / count)
		print "OOV SRI: %d" % oov_sri
		print "PPL: %.2f" % exp10(-logp_combine / count)
		print "OOV: %d" % oov_combine

		return exp10(-logp_combine / count)


	def Reset(self):
		"""
		Reset net state
		"""
		self.CEC = np.zeros(self.cell_blocks)
		self.context = np.ones(self.cell_blocks)
		#self.dSdwWeightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension))
		#self.dSdwWeightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension))
		#self.dSdwWeightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension))


	def save(self, filename):
		"""
		cPickle net to filename
		"""
		# attributes that we want to save
		to_save = ['CEC', 'cell_blocks',
		           'context', 'dic', 'full_hidden_dimension',
		           'full_input_dimension', 'hidden_dimension',
		           'independent', 'input_dimension', 'output_dimension',
		           'peepForgetGate', 'peepInputGate', 'peepOutputGate',
		           'version', 'weightsForgetGate', 'weightsGlobalOutput']

		lstm_container = {}

		for attr in dir(self):
			if attr in to_save:
				lstm_container[attr] = getattr(self, attr)

		try:
			cPickle.dump(lstm_container, open(filename+".lstm", "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
		except IOError:
			raise

	def load(self, filename):
		"""
		Loads net from file
		"""
		try:
			lstm_container = cPickle.load(open(filename, "rb"))
		except IOError:
			raise

		for attr in lstm_container.keys():
			try:
				setattr(self, attr, lstm_container[attr])
			except:
				raise

		if not self.version == __version__:
			print "Warning: Loadad RNN is version %s, this is version %s." % (self.version, __version__)

	def nbest_rescore(self, input_file):
		"""Rescores the input nbest file"""
		# nbset file format:
		# hypothesis_number acoustic_model_score language_model_score count_of_words <s> words of hypothesis </s>
		if self.net_load_file:
			N = "_".join([self.net_load_file, input_file, str(self.srilm_lambda), str(self.srilm_file), str(self.LMW),
			              str(self.WIP), "rescored.nbest"])
		else:
			N = "_".join([input_file, str(self.srilm_lambda), str(self.srilm_file), str(self.LMW), str(self.WIP),
			              "rescored.nbest"])
		hypothesis = []

		LMW = self.LMW
		WIP = self.WIP

		cache = []
		no_current = 0
		with codecs.open(N, "w", ENCODING) as fout:
			for line in codecs.open(input_file, "r", ENCODING):
				l = line.strip().split()

				try:
					# read line
					no = int(l[0])
					am = float(l[1])
					lm = float(l[2])
					wrd_cnt = int(l[3])
					words = l[5:] # without <s>
				except IndexError:
					continue

				if no > no_current:
					# results from last block
					unfiltered = [C for C in hypothesis if C[0] == no_current]
					s = sorted(unfiltered, key=itemgetter(6), reverse=True)[0]        # sort by score
					fout.write("%d %f %f %d %s %f %f\n" % (s[0], s[1], s[2], s[3], " ".join(s[4]), s[5], s[6]))

					cache += s[4]
					no_current = no

				new_logp = self.getLogP(words, cache)
				score = (LMW * new_logp) + am + (WIP * wrd_cnt)
				hypothesis.append((no, am, lm, wrd_cnt, words, new_logp, score))


	def getLogP(self, words, cache):
		"""Get logratimic probability of given text (words) with cache text before
		@param words: list of String
		@param cache: list of String
		@return: float
		"""
		#convert to idxs
		cdef double logp = 0.
		cdef double logp_net = 0.
		cdef double logp_sri = 0.
		cdef double LOG10TOLOG = np.log(10)
		cdef double LOGTOLOG10 = 1. / LOG10TOLOG
		cdef double oov_log_penalty = -5            #log penalty
		cdef double srilm_lambda = self.srilm_lambda

		iText = [0]
		for word in words:
			try:
				iText.append(self.dic.index(word))
			except (IndexError, ValueError, KeyError):
				iText.append(-1)    # unk

		# only RNN
		if not self.srilm_file:
			for word_idx in range(len(iText) - 1):
				if iText[word_idx + 1] > -1:
					o = self.forward(iText[word_idx], iText[word_idx + 1], cache)
					logp += log10(o[iText[word_idx + 1]])
				else:
					logp += oov_log_penalty

		# combine with SRILM
		else:
			self.Reset()
			for word_idx in range(len(iText) - 1):
				ctx = words[: word_idx + 1] # last xx words
				ctx.insert(0, "<s>")
				ctx = ctx[::-1]
				l_sri = LOGTOLOG10 * self.arpaLM.prob(*ctx)
				p_sri = exp10(l_sri)
				# if in RNN dic
				if iText[word_idx + 1] > -1 and l_sri > -98:
					vector = self.index_to_vector(iText[word_idx], cache)
					if iText[word_idx] > -1:
						nonzeros = [iText[word_idx]] + range(len(self.dic), self.input_dimension + self.cell_blocks + 1)
					else:
						nonzeros = range(len(self.dic), self.input_dimension + self.cell_blocks + 1)
					#o = self.forward(vector, nonzeros)

					o = self.forward(iText[word_idx], iText[word_idx + 1], cache)
					p_net = o[iText[word_idx + 1]]
					logp += log10(p_net * (1 - srilm_lambda) + p_sri * (srilm_lambda))

				# if not
				else: # give a penalty or back-off with SRI
					if l_sri > -98:
						logp += l_sri
					else:
						logp += oov_log_penalty
		return logp


	def runTrain(self, np.ndarray[long, ndim=1] iText, np.ndarray[long, ndim=1] iTest,
	             np.ndarray[long, ndim=1] iValid,
	             rnd_seed,
	             int cell_blocks,
	             int input_dimension,
	             int output_dimension):
		"""Main training method"""

		print "Initialize training..."

		# init random module
		random.seed(rnd_seed)

		# CONST
		cdef double init_weight_range = 0.1
		cdef int biasInputGate = 2 #orig. 2
		cdef int biasForgetGate = -2 #orig. -2
		cdef int biasOutputGate = 2 #orig. 2

		cdef double learningRate = self.alpha

		# indexes
		cdef unsigned int i, j, k, c, z, loc, word = 0
		#cdef Py_ssize_t i, j, k, c, z, loc, word = 0

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

		cdef int LAST_N = self.len_cache
		cdef unsigned int len_dic = self.len_dic
		cdef unsigned int full_input_dimension = input_dimension + cell_blocks + 1
		cdef unsigned int full_hidden_dimension = cell_blocks + 1

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC = np.zeros((cell_blocks), dtype=DTYPE)
		#CEC = np.zeros((cell_blocks), dtype=DTYPE)
		#cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC
		#cdef DTYPE_t *CEC = <DTYPE_t *>(np.PyArray_DATA(np.zeros((cell_blocks), dtype=DTYPE)))

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] context = np.ones((cell_blocks), dtype=DTYPE)
		#cdef DTYPE_t *context = <DTYPE_t *>(np.PyArray_DATA(np.ones((cell_blocks), dtype=DTYPE)))


		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsNetInput = np.zeros((cell_blocks, full_input_dimension),
																				dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsInputGate = np.zeros((cell_blocks, full_input_dimension),
																				dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsForgetGate = np.zeros((cell_blocks, full_input_dimension),
																				dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsOutputGate = np.zeros((cell_blocks, full_input_dimension),
																				dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsGlobalOutput = np.zeros((output_dimension, full_hidden_dimension), dtype=DTYPE)
		#cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsGlobalInput = np.zeros((input_dimension, full_hidden_dimension), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension),dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] NetInputAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] InputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] ForgetGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] OutputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CECSquashAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] NetOutputAct = np.zeros((cell_blocks), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC1 = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC2 = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC3 = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] output = np.zeros((output_dimension), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] deltaNetOutput = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] deltaGlobalOutputPre = np.zeros((output_dimension), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] input = np.zeros((input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] target_output = np.zeros((output_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] full_hidden = np.zeros((full_hidden_dimension), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] peepInputGate = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] peepForgetGate = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] peepOutputGate = np.zeros((cell_blocks), dtype=DTYPE)

		cdef int word_idx = 0
		cdef int next_word_idx = 0

		"""
		DEF EXP_TABLE_SIZE = 1000
		DEF MAX_EXP = 6

		cdef DTYPE_t[EXP_TABLE_SIZE] EXP_TABLE
		# build the sigmoid table
		for i in range(EXP_TABLE_SIZE):
			EXP_TABLE[i] = <DTYPE_t>exp((i / <DTYPE_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
			EXP_TABLE[i] = <DTYPE_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
		"""

		if self.net_load_file:
			print "Loading net..."
			# init from weights
			weightsNetInput = self.weightsNetInput
			weightsInputGate = self.weightsInputGate
			weightsForgetGate = self.weightsForgetGate
			weightsOutputGate = self.weightsOutputGate

			peepInputGate = self.peepInputGate
			peepForgetGate = self.peepForgetGate
			peepOutputGate = self.peepOutputGate

			weightsGlobalOutput = self.weightsGlobalOutput

		else:
			# random init
			for i in xrange(full_input_dimension):
				for j in xrange(cell_blocks):
					weightsNetInput[j, i] = (random.random() * 2 - 1) * init_weight_range
					weightsInputGate[j, i] = (random.random() * 2 - 1) * init_weight_range
					weightsForgetGate[j, i] = (random.random() * 2 - 1) * init_weight_range
					weightsOutputGate[j, i] = (random.random() * 2 - 1) * init_weight_range

			for j in xrange(cell_blocks):
				weightsInputGate[j, full_input_dimension - 1] += biasInputGate
				weightsForgetGate[j, full_input_dimension - 1] += biasForgetGate
				weightsOutputGate[j, full_input_dimension - 1] += biasOutputGate

			for j in xrange(cell_blocks):
				peepInputGate[j] = (random.random() * 2 - 1) * init_weight_range
				peepForgetGate[j] = (random.random() * 2 - 1) * init_weight_range
				peepOutputGate[j] = (random.random() * 2 - 1) * init_weight_range

			for j in xrange(full_hidden_dimension):
				for k in xrange(output_dimension):
					weightsGlobalOutput[k][j] = (random.random() * 2 - 1) * init_weight_range

		# compute ppl of raw net
		p = [(self.len_dic, self.len_dic*-1000)]    # if only zeros in RNN, should be ppl like this

		cdef int iter = 0

		while True:
			start = time.time()

			#self.Reset()
			#CEC = np.zeros((cell_blocks), dtype=DTYPE)
			#context = np.ones((cell_blocks), dtype=DTYPE)
			for c in range(cell_blocks):
				CEC[c] = 0.
				context[c] = 1.

			dSdwWeightsNetInput = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
			dSdwWeightsInputGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)
			dSdwWeightsForgetGate = np.zeros((cell_blocks, full_input_dimension), dtype=DTYPE)

			logp = 0.
			word_idx = 0

			for word in xrange(len(iText) - 1):

				# case of OOV
				word_idx = iText[word]
				next_word_idx = iText[word + 1]
				if next_word_idx < 0: # == -1
					continue

				if word % 1000 == 999:
					speed = word / (time.time() - start)
					time_to_end = time.strftime('%d:%H:%M:%S', time.gmtime((len(iText) - word) * (1 / speed)))
					print "speed %.2f words/s, %s remaining, train ppl: %.2f, alpha: %.4f\r" % (
					speed, time_to_end, exp10(-logp / np.double(word)), learningRate),
					sys.stdout.flush()

					# if we went a wrong way
					if math.isnan(logp):
						self.Reset()
						break

				# making input vector
				cache = self.lText[word - LAST_N: word + 1]
				input = self.index_to_vector(word_idx, cache)

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
				nonzero_input_dimension = range(word_idx, word_idx + 1) + range(len_dic, full_input_dimension)
				zero_input_dimension = range(0, word_idx) + range(word_idx + 1, len_dic)


				# Tanh layer
				for i in nonzero_input_dimension:
					full_input[i] = TanhActivate_f(full_input[i])

				with nogil, parallel():
					# cell block arrays
					for i in xrange(cell_blocks):
						NetInputSum[i] = 0.
						InputGateSum[i] = 0.
						ForgetGateSum[i] = 0.
						OutputGateSum[i] = 0.

				for i in nonzero_input_dimension:
				#for i in xrange(word_idx, word_idx + 1): # + xrange(len_dic, full_input_dimension):
					#with nogil:
					for j in xrange(cell_blocks):
						NetInputSum[j] += weightsNetInput[j, i] * full_input[i]
						InputGateSum[j] += weightsInputGate[j, i] * full_input[i]
						ForgetGateSum[j] += weightsForgetGate[j, i] * full_input[i]
						OutputGateSum[j] += weightsOutputGate[j, i] * full_input[i]

				# internals of cell blocks
				with nogil:
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

				with nogil:
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
					with nogil:
						deltaOutputGatePost = deltaNetOutput[c] * CEC3[c]
						deltaOutputGatePre = SigmoidDerivative_f(OutputGateSum[c]) * deltaOutputGatePost

					for i in nonzero_input_dimension:
					#for i in range(full_input_dimension):
						weightsOutputGate[c, i] += full_input[i] * deltaOutputGatePre * learningRate

					with nogil:
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

				with nogil:
					#roll-over context to next time step
					for j in xrange(cell_blocks):
						context[j] = NetOutputAct[j]
						CEC[j] = CEC3[j]

				if next_word_idx == 0 and self.independent: # if reached end of sentence </s>
					#CEC = np.zeros(cell_blocks)
					#context = np.ones(cell_blocks)
					for c in xrange(cell_blocks):
						CEC[c] = 0.
						context[c] = 1.

				# to zero out target vector
				target_output[next_word_idx] = 0.

			# save results
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

			p.append(self.ppl(iTest, self.lTest))

			print "                                                                                                              \r",
			print "%d, speed %.2f words/s, ppl: %.2f, alpha %.4f" % (
			iter, word / (time.time() - start), p[len(p) - 1][0], learningRate)
			iter += 1

			#logp*min_improvement<llogp
			if p[len(p) - 1][1] * 1.003 < p[len(p) - 2][1]:
				learningRate /= 2

			# save iteration - in case we will crash
			self.save("%s_%02d" % (self.net_save_file, iter))

			# when to stop
			if learningRate < 0.0005:
				break

			# maximal number of iterations allowed
			if iter > 20:
				break

		# print combine test PPL if srilm model presented
		if self.srilm_file:
			self.ppl_combine(iValid, self.lValid)
		else:
			print self.ppl(iValid, self.lValid)

		# save final net
		self.save(self.net_save_file)


	def forward(self, int word_idx, int next_word_idx, cache):

		# INIT
		cdef int cell_blocks = self.cell_blocks
		cdef int input_dimension = self.input_dimension
		cdef int output_dimension = self.output_dimension
		cdef int full_input_dimension = input_dimension + cell_blocks + 1
		cdef int full_hidden_dimension = cell_blocks + 1

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC = self.CEC
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] context = self.context

		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsNetInput = self.weightsNetInput
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsInputGate = self.weightsInputGate
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsForgetGate = self.weightsForgetGate
		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsOutputGate = self.weightsOutputGate

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] peepInputGate = self.peepInputGate
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] peepForgetGate = self.peepForgetGate
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] peepOutputGate = self.peepOutputGate

		cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] weightsGlobalOutput = self.weightsGlobalOutput

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] NetInputSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] InputGateSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] ForgetGateSum = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] OutputGateSum = np.zeros((cell_blocks), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] NetInputAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] InputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] ForgetGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] OutputGateAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CECSquashAct = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] NetOutputAct = np.zeros((cell_blocks), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC1 = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC2 = np.zeros((cell_blocks), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] CEC3 = np.zeros((cell_blocks), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] output = np.zeros((output_dimension), dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] full_input = np.zeros((full_input_dimension), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] full_hidden = np.zeros((full_hidden_dimension), dtype=DTYPE)

		cdef unsigned int i, j, k, c, loc
		cdef double softmax_suma, softmax_val = 0.
		cdef int len_dic = len(self.dic)

		input = self.index_to_vector(word_idx, cache)

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
		if word_idx > -1:
			nonzero_input_dimension = range(word_idx, word_idx + 1) + range(len_dic, full_input_dimension)
			zero_input_dimension = range(0, word_idx) + range(word_idx + 1, len_dic)
		else:
			nonzero_input_dimension = range(len_dic, full_input_dimension)
			zero_input_dimension = range(0, len_dic)

		# Tanh layer
		for i in nonzero_input_dimension:
			full_input[i] = TanhActivate_f(full_input[i])

		with nogil:
			# cell block arrays
			for i in range(cell_blocks):
				NetInputSum[i] = 0.
				InputGateSum[i] = 0.
				ForgetGateSum[i] = 0.
				OutputGateSum[i] = 0.

		for i in nonzero_input_dimension:
			for j in range(cell_blocks):
				NetInputSum[j] += weightsNetInput[j, i] * full_input[i]
				InputGateSum[j] += weightsInputGate[j, i] * full_input[i]
				ForgetGateSum[j] += weightsForgetGate[j, i] * full_input[i]
				OutputGateSum[j] += weightsOutputGate[j, i] * full_input[i]

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
			for k in range(output_dimension):
				output[k] = 0.
			for k in range(output_dimension):
				for j in range(full_hidden_dimension):
					output[k] += weightsGlobalOutput[k, j] * full_hidden[j]

			# SoftMax
			softmax_suma = 0.
			for k in range(output_dimension):
				softmax_val = exp(output[k])
				softmax_suma += softmax_val
				output[k] = softmax_val

			for k in range(output_dimension):
				output[k] /= softmax_suma

		# roll-over context to next time step
		for j in range(cell_blocks):
			context[j] = NetOutputAct[j]
			CEC[j] = CEC3[j]

		self.context = context
		self.CEC = CEC

		return output
