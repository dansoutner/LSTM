# -*- coding:utf-8 -*-

"""
Recurrent neural network based statistical language modelling toolkit

Based on LSTM RNN, model proposed by Jürgen Schmidhuber
http://www.idsia.ch/~juergen/

Implemented by Daniel Soutner,
Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
dsoutner@kky.zcu.cz, 2014; Licensed under the 3-clause BSD.

"""

__version__ = "0.6.2"

# usual python libs
import numpy as np
import random
import time
#import math
#import sys
import cPickle
from operator import itemgetter
import codecs

# LDA module
from lda import *
# ARPA module
from ArpaLM import *
# fast training
from fast import FastRunTrain as FastRunTrain
from fast import FastRunTrain_outputclasses as FastRunTrain_outputclasses
from fast import FastRunTrain_N as FastRunTrain_N
from fast import FastRunTrain_N_outputclasses as FastRunTrain_N_outputclasses
from fast import FastForward as FastForward
from fast import FastRunTrain_NLDA_outputclasses as FastRunTrain_NLDA_outputclasses
from fast import exp10_f as exp10
# other imports
from numpy import log10 as log10


# numpy settings
np.set_printoptions(edgeitems=2, infstr='Inf', linewidth=75, nanstr='NaN',	precision=8, suppress=True, threshold=1000000)

DTYPE = np.float32

print "LSTM LM version %s" % __version__

# CONST
UNK = "<unk>"
ENCODING = "utf-8"


class LSTM(object):
	"""Main LSTM LM object"""

	def __str__(self):
		o = ""
		try:
			o += "Train text %d words from %s\n" % (len(self.lText), self.train_file)
			o += "Test text %d words from %s\n" % (len(self.lTest), self.test_file)
			o += "Valid text %d words from %s\n" % (len(self.lValid), self.valid_file)
		except (AttributeError, ValueError, TypeError):
			pass

		o += "Type %s\n" % self.input_type
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

		self.num_threads=args.num_threads
		self.version = __version__
		self.debug = args.debug
		self.rnd_seed = args.rnd_seed
		self.independent = args.independent

		if not args.save_net:
			try:
				self.net_save_file = args.train[0] + "_" + str(args.iHidden) + "_" + args.input_type + "_" + str(time.time()).split(".")[0]
			except:
				self.net_save_file = args.save_net
		else:
			self.net_save_file = args.save_net
		self.net_load_file = args.load_net
		self.net_nbest_file = args.nbest_rescore
		self.input_type = args.input_type

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

		iHidden = 0
		if args.iHidden:
			iHidden = args.iHidden

		# set train, test and valid files
		if args.train:
			self.train_file = args.train[0]
			self.test_file = args.train[1]
			self.valid_file = args.train[2]

		# load support models
		# CSLM
		if args.projections_file:
			self.projections = cPickle.load(open(args.projections_file))
			self.len_projections = len(self.projections.values()[0])
		# LDA
		if args.stopwords_file:
			self.stopwords = set(codecs.open(args.stopwords_file, "r", ENCODING).read().lower().split())
		else:
			self.stopwords = []

		self.len_cache = args.len_cache        #default = 50

		if args.lda_dict and args.lda_model:
			self.lda = LDA(self.stopwords)
			self.lda.load(args.lda_model, args.lda_dict)
			self.len_lda = len(self.lda.cache_to_fv(["."]))
			#print self.lda.getTopics()

		# classes
		if args.class_file:
			self.classes = cPickle.load(open(args.class_file))
			all_classes = []
			for k in self.classes.keys():
				all_classes.append(self.classes[k])
			self.len_classes = max(all_classes) + 1

		if self.debug:
			print "Preparing data..."

		if args.train:
			# make data
			lText = self.text_file_to_list_of_words(self.train_file)
			lTest = self.text_file_to_list_of_words(self.test_file)
			lValid = self.text_file_to_list_of_words(self.valid_file)

			# create vocabulary
			if args.vocabulary_file:
				self.dic = self.create_dic_from_file(args.vocabulary_file)
			else:
				self.dic = self.create_dic(self.train_file)
			self.dic.remove("</s>")    # it is good to have a </s> as 0 in dic
			self.dic.insert(0, "</s>")

			# create word hashes
			self.create_hash()

			if args.projections_file:
				prj_oov = []
				for k in self.dic:
					try:
						self.projections[k.encode(ENCODING)]
					except KeyError:
						prj_oov.append(k)
				if len(prj_oov) > 0:
					print "%d words not found in projections \n (%s)" % (len(prj_oov), ", ".join(prj_oov).encode(ENCODING))

			#try:
			#	self.dic.remove(UNK)    # unk word remove from dic
			#except ValueError:
			#	pass

			self.iText = np.array(self.word_list2idx(lText))
			self.iTest = np.array(self.word_list2idx(lTest))
			self.iValid = np.array(self.word_list2idx(lValid))
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
		try:
			self.dic
			self.input_type
		except AttributeError:
			print "Error: No type of input vector."
			return
		else:
			if self.input_type == "N":
				iInputs = len(self.dic)
			elif self.input_type == "FV":
				iInputs = self.len_projections
			elif self.input_type == "FV+":
				iInputs = self.len_projections
			elif self.input_type == "FV+LDA":
				iInputs = self.len_projections + self.len_lda
			elif self.input_type == "N+LDA":
				iInputs = len(self.dic) + self.len_lda
			elif self.input_type == "N+CLASS":
				iInputs = len(self.dic) + self.len_classes
			elif self.input_type == "N+LDA+CLASS":
				iInputs = len(self.dic) + self.len_classes + self.len_lda

		if args.output_classes:
			self.create_classes(args.output_classes)
			iOutputs = len(self.class_cn) + len(self.dic)
			self.output_classes = True
		else:
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

		try:        # just to be sure if we loaded all variables
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

		except (AttributeError, NameError) as e:        # if not, init net matrixes
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
			#self.run_train(self.iText, self.iTest, self.iValid, rnd_seed, iHidden, iInputs, iOutputs)
			self.run_train()

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


	def word_list2idx(self, lText):
		"""
		Converts list of words to list of indexes
		"""
		iText = [0]*len(lText)
		for i in range(len(lText)):
			try:
				iText[i] = self.word2idx_hash[lText[i]]
			except (KeyError):
				iText.append(-1)    # unk
		return iText


	def create_hash(self):
		"""
		Creates word-to-index and index-to-word hashes
		"""
		self.idx2word_hash = {}
		self.word2idx_hash = {}
		for idx in range(len(self.dic)):
			word = self.dic[idx]
			self.idx2word_hash[idx] = word
			self.word2idx_hash[word] = idx


	def create_dic(self, text_file):
		"""
		Creates vocabulary form train file
		"""
		sentences = (line.split() for line in codecs.open(text_file, "r", ENCODING))
		sentence_no, vocab = -1, {}
		vocab["</s>"] = 1
		for sentence_no, sentence in enumerate(sentences):
			for word in sentence:
				try:
					vocab[word] += 1
				except KeyError:
					vocab[word] = 1
		return sorted(vocab.keys())


	def create_dic_from_file(self, vocab_file):
		"""
		Creates vocabulary from input vocabulary file
		"""
		vocab = {}
		vocab["</s>"] = True
		with codecs.open(vocab_file, "r", ENCODING) as f:
			for line in f:
				w = line.strip()
				if len(w) > 0:
					vocab[w] = True
		return sorted(vocab.keys())


	def create_classes(self, number_of_classes):
		"""
		Creates output classes
		"""
		# create classes
		df = 0.
		dd = 0.
		a = 0
		b = 0

		class vocab_word():
			def __init__(self, word):
				self.cn = 0
				self.word = word
				self.prob = 0.
				self.class_index = 0
				self.in_class = 0


		def make_vocab(text_file):
			vocab = {}
			vocab[0] = vocab_word("</s>")
			sentences = (line.split() for line in codecs.open(text_file, "r", ENCODING))
			for sentence in sentences:
				for word in sentence:
					try:
						vocab[self.word2idx_hash[word]].cn += 1
					except KeyError:
						vocab[self.word2idx_hash[word]] = vocab_word(word)
						vocab[self.word2idx_hash[word]].cn = 1
			return vocab

		self.vocab = make_vocab(self.train_file)
		self.class_size = min(number_of_classes, self.len_dic)

		for i in self.vocab.keys():
			b += self.vocab[i].cn
		for i in self.vocab.keys():
			dd += np.sqrt(self.vocab[i].cn/float(b))
		for i in self.vocab.keys():
			df += np.sqrt(self.vocab[i].cn/float(b))/dd
			if (df > 1):
				df = 1
			if (df > (float(a + 1)/self.class_size)):
				self.vocab[i].class_index = a
				if (a < self.class_size - 1):
					a += 1
			else:
				self.vocab[i].class_index = a

		self.class_words = [[]] * (self.class_size)
		self.class_cn = np.zeros(self.class_size, dtype=int)
		self.class_max_cn = np.zeros(self.class_size)

		for i in range(self.class_size):
			self.class_cn[i] = 0
			self.class_words[i] = [[]]*self.class_max_cn[i]

		for i in self.vocab.keys():
			cl = self.vocab[i].class_index
			self.class_words[cl].append(i)
			self.class_cn[cl] += 1

		self.word2class_hash = {}
		self.idx2class_hash = {}
		for i in self.vocab.keys():
			self.vocab[i].in_class = self.class_words[self.vocab[i].class_index].index(i)
			self.word2class_hash[self.idx2word_hash[i]] = self.vocab[i].class_index
			self.idx2class_hash[i] = self.vocab[i].class_index

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


	def index_to_vector(self, idx, cache):
		"""
		Different kinds of input vectors
		"""

		if self.input_type == "N":
			o = np.zeros((self.len_dic), dtype=DTYPE)
			if idx > -1:
				o[idx] = 1.
			return o

		elif self.input_type == "FV+":
			try:
				return np.array((self.projections[cache[len(cache) - 1].encode(ENCODING)]), dtype=DTYPE)
			except (KeyError, IndexError, ValueError):
				return np.zeros((self.len_projections), dtype=DTYPE)

		elif self.input_type == "FV":
			if idx > -1:
				try:
					return np.array((self.projections[self.idx2word_hash[idx].encode(ENCODING)]), dtype=DTYPE)
				except (KeyError, IndexError, ValueError):
					return np.zeros((self.len_projections), dtype=DTYPE)
			else:
				return np.zeros((self.len_projections))

		elif self.input_type == "FV+LDA":
			o = np.zeros((self.len_projections + self.len_lda), dtype=DTYPE)
			try:
				o[:self.len_projections] = np.array((self.projections[cache[len(cache) - 1].encode(ENCODING)]), dtype=DTYPE)
			except (KeyError, IndexError, ValueError):
				pass        # zeros
			cache = [word for word in cache if word not in self.stopwords]
			lda_vector = np.array(self.lda.cache_to_fv(cache))
			o[self.len_projections:] = lda_vector
			return o

		elif self.input_type == "N+LDA":
			o = np.zeros((self.len_dic + self.len_lda), dtype=DTYPE)
			if idx > -1:
				o[idx] = 1.
				cache = [word for word in cache if word not in self.stopwords]
				lda_vector = np.array(self.lda.cache_to_fv(cache))
				o[self.len_dic:] = lda_vector
			return o

		elif self.input_type == "N+CLASS":
			o = np.zeros((self.len_dic + self.len_classes), dtype=DTYPE)
			if idx > -1:
				o[idx] = 1.
				try:
					cl = self.classes[self.dic[idx]]
					o[self.len_dic + cl] = 1.
				except (KeyError, IndexError, ValueError):
					pass
			return o

		elif self.input_type == "N+LDA+CLASS":
			o = np.zeros((self.len_dic + self.len_classes + self.len_lda), dtype=DTYPE)
			if idx > -1:
				# N
				o[idx] = 1.
				# CLASS
				try:
					cl = self.classes[self.dic[idx]]
					o[self.len_dic + cl] = 1.
				except (KeyError, IndexError, ValueError):
					pass
				#LDA
				cache = [word for word in cache if word not in self.stopwords]
				lda_vector = np.array(self.lda.cache_to_fv(cache))
				o[self.len_dic + self.len_classes:] = lda_vector
			return o



	def Reset(self):
		"""
		Reset net state
		"""
		self.CEC = np.zeros((self.cell_blocks), dtype=DTYPE)
		self.context = np.ones((self.cell_blocks), dtype=DTYPE)
		self.dSdwWeightsNetInput = np.zeros((self.cell_blocks, self.full_input_dimension), dtype=DTYPE)
		self.dSdwWeightsInputGate = np.zeros((self.cell_blocks, self.full_input_dimension), dtype=DTYPE)
		self.dSdwWeightsForgetGate = np.zeros((self.cell_blocks, self.full_input_dimension), dtype=DTYPE)


	def save(self, filename):
		"""
		cPickle net to filename
		"""
		# attributes that we want to save
		to_save = set(['CEC', 'cell_blocks',
		           'context', 'dic', 'full_hidden_dimension',
		           'full_input_dimension', 'hidden_dimension',
		           'independent', 'input_dimension', 'output_dimension',
		           'peepForgetGate', 'peepInputGate', 'peepOutputGate',
		           'version', 'weightsForgetGate', 'weightsGlobalOutput',
		           'weightsInputGate', 'weightsNetInput', 'weightsOutputGate',
		           'lda', 'lda_len', 'out_word_to_class', 'out_ppst_to_class',
		           'out_class', 'projections', 'len_projections', 'lda', 'len_lda',
		           'classes', 'len_classes', "input_type",
		           'stopwords', 'len_cache',
		           'class_size', 'class_max_cn', 'class_cn', 'class_words',
		           'idx2class_hash', 'word2class_hash', 'idx2word_hash', 'word2idx_hash'])

		# need to convert memoryviews to array
		convert_and_save = set(['CEC', 'context',
						'peepForgetGate', 'peepInputGate', 'peepOutputGate',
						'weightsForgetGate', 'weightsGlobalOutput',
						'weightsInputGate', 'weightsNetInput', 'weightsOutputGate',])

		# this is rest which we do not convert
		only_save = to_save - convert_and_save

		lstm_container = {}

		for attr in dir(self):
			if attr in convert_and_save:
				lstm_container[attr] = np.asarray(getattr(self,attr))
			if attr in only_save:
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
		"""
		Rescores the input nbest file
		# nbset file format:
		# hypothesis_number acoustic_model_score language_model_score count_of_words <s> words of hypothesis </s>
		"""
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


	def ppl(self, iText, lText):
		"""
		Computes perplexity of RNN on given text (splitted)
		"""
		logp = 0.
		count = 0
		oov_net = 0

		self.Reset()
		for word_idx in range(len(iText) - 1):
			if iText[word_idx + 1] > -1:
				#!cache = lText[word_idx - self.len_cache: word_idx]
				cache = lText[word_idx - self.len_cache: word_idx + 1]
				#vector = self.index_to_vector(iText[word_idx], cache)
				output = FastForward(self, iText[word_idx], iText[word_idx + 1], cache)
				if self.output_classes:
					next_word_idx = iText[word_idx + 1]
					logp += log10(output[self.len_dic + self.idx2class_hash[next_word_idx]] * output[next_word_idx])
				else:
					logp += log10(output[iText[word_idx + 1]])
				count += 1
			else:
				#l += oov_log_penalty
				#print "LSTM OOV at %d" % ii
				oov_net += 1
				pass

			if iText[word_idx + 1] == 0 and self.independent:
				self.CEC = np.zeros((self.cell_blocks), dtype=DTYPE)
				self.context = np.ones((self.cell_blocks), dtype=DTYPE)

		if self.debug:
			print "Net OOVs %d" % oov_net

		return exp10(-logp / count), logp


	def ppl_combine(self, iText, lText):
		"""
		Computes perplexity of RNN on given text (splitted)
		"""

		logp_net = 0.
		logp_sri = 0.
		logp_combine = 0.

		oov_combine = 0
		oov_sri = 0
		oov_net = 0
		count = 0

		LOG10TOLOG = np.log(10)
		LOGTOLOG10 = 1. / LOG10TOLOG

		oov_penalty = -8           #some ad hoc penalty - when mixing different vocabularies, single model score is not real PPL
		p_sri = 0.
		p_net = 0.
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
					#logp_combine += log10(0 * (1. - self.srilm_lambda) + p_sri * srilm_lambda)
					logp_combine += log10(p_sri * self.srilm_lambda)
				else:                                   # if not NetOOV
					cache = lText[word_idx - self.len_cache: word_idx + 1]
					next_word_idx = iText[word_idx + 1]
					output = FastForward(self, iText[word_idx], next_word_idx, cache)
					if self.output_classes:
						p_net = output[self.len_dic + self.idx2class_hash[next_word_idx]] * output[next_word_idx]
					else:
						p_net = output[next_word_idx]
					logp_net += log10(p_net)
					logp_combine += log10((p_net * (1. - self.srilm_lambda)) + (p_sri * self.srilm_lambda))

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

			if iText[word_idx + 1] == 0 and self.independent:
				self.CEC = np.zeros((self.cell_blocks), dtype=DTYPE)
				self.context = np.ones((self.cell_blocks), dtype=DTYPE)

		print "PPL Net: %.2f" % exp10(-logp_net / count)
		print "OOV Net: %d" % oov_net
		print "PPL SRI: %.2f" % exp10(-logp_sri / count)
		print "OOV SRI: %d" % oov_sri
		print "PPL: %.2f" % exp10(-logp_combine / count)
		print "OOV: %d" % oov_combine

		return exp10(-logp_combine / count)


	def getLogP(self, words, cache):
		#Gets logratimic probability of given text (words) with cache text before
		#@param words: list of String
		#@param cache: list of String
		#@return: float

		#convert to idxs
		logp = 0.
		logp_net = 0.
		logp_sri = 0.
		LOG10TOLOG = np.log(10)
		LOGTOLOG10 = 1. / LOG10TOLOG
		oov_log_penalty = -5            # log penalty
		srilm_lambda = self.srilm_lambda

		iText = [0]
		for word in words:
			try:
				iText.append(self.word2idx_hash[word])
			except KeyError:
				iText.append(-1)    # unk

		# only RNN
		if not self.srilm_file:
			for word_idx in range(len(iText) - 1):
				if iText[word_idx + 1] > -1:
					cache = words[word_idx - self.len_cache: word_idx + 1]
					next_word_idx = iText[word_idx + 1]
					output = FastForward(self, iText[word_idx], next_word_idx, cache)
					if self.output_classes:
						logp += log10(output[self.len_dic + self.idx2class_hash[next_word_idx]] * output[next_word_idx])
					else:
						logp += log10(output[next_word_idx])
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
					cache = words[word_idx - self.len_cache: word_idx + 1]
					next_word_idx = iText[word_idx + 1]
					output = FastForward(self, iText[word_idx], next_word_idx, cache)
					if self.output_classes:
						p_net = output[self.len_dic + self.idx2class_hash[next_word_idx]] * output[next_word_idx]
					else:
						p_net = output[next_word_idx]
					logp += log10(p_net * (1 - srilm_lambda) + p_sri * (srilm_lambda))

				# if not
				else: # give a penalty or back-off with SRI
					if l_sri > -98:
						logp += l_sri
					else:
						logp += oov_log_penalty
		return logp


	def init_net(self):
		"""
		Initialize the net weights with random numbers,
		"""
		init_weight_range = 0.1
		biasInputGate = 2 #orig. 2
		biasForgetGate = -2 #orig. -2
		biasOutputGate = 2 #orig. 2

		# init random module
		random.seed(self.rnd_seed)

		# random init
		for i in xrange(self.full_input_dimension):
			for j in xrange(self.cell_blocks):
				self.weightsNetInput[j, i] = (random.random() * 2 - 1) * init_weight_range
				self.weightsInputGate[j, i] = (random.random() * 2 - 1) * init_weight_range
				self.weightsForgetGate[j, i] = (random.random() * 2 - 1) * init_weight_range
				self.weightsOutputGate[j, i] = (random.random() * 2 - 1) * init_weight_range

		for j in xrange(self.cell_blocks):
			self.weightsInputGate[j, self.full_input_dimension - 1] += biasInputGate
			self.weightsForgetGate[j, self.full_input_dimension - 1] += biasForgetGate
			self.weightsOutputGate[j, self.full_input_dimension - 1] += biasOutputGate

		for j in xrange(self.cell_blocks):
			self.peepInputGate[j] = (random.random() * 2 - 1) * init_weight_range
			self.peepForgetGate[j] = (random.random() * 2 - 1) * init_weight_range
			self.peepOutputGate[j] = (random.random() * 2 - 1) * init_weight_range

		for j in xrange(self.full_hidden_dimension):
			for k in xrange(self.output_dimension):
				self.weightsGlobalOutput[k][j] = (random.random() * 2 - 1) * init_weight_range


	def run_train(self):
		"""
		Init and train net
		"""

		self.init_net()
		# compute ppl of raw net
		p = [(self.len_dic, self.len_dic*-1000)]    # if only zeros in RNN, should be ppl like this
		iter = 0

		while True:
			start = time.time()
			# train one iteration
			if self.input_type == "N":
				if self.output_classes:
					FastRunTrain_N_outputclasses(self, self.iText)
				else:
					FastRunTrain_N(self, self.iText)
			elif self.input_type == "N+LDA":
				if self.output_classes:
					FastRunTrain_NLDA_outputclasses(self, self.iText, self.lText)
				else:
					FastRunTrain(self, self.iText, self.lText)
			else:
				if self.output_classes:
					FastRunTrain_outputclasses(self, self.iText, self.lText)
				else:
					FastRunTrain(self, self.iText, self.lText)
			p.append(self.ppl(self.iTest, self.lTest))

			print "                                                                                                           \r",
			print "%d, speed %.2f words/s, ppl: %.2f, alpha %.4f" % (
			iter, len(self.lText) / (time.time() - start), p[len(p) - 1][0], self.alpha)
			iter += 1

			#logp*min_improvement<llogp
			if p[len(p) - 1][1] * 1.003 < p[len(p) - 2][1]:
				self.alpha /= 2

			# save iteration - in case we will crash
			self.save("%s_%02d" % (self.net_save_file, iter))

			# when to stop
			if self.alpha < 0.0005:
				break

			# maximal number of iterations allowed
			if iter > 20:
				break

		# print combine test PPL if srilm model presented
		if self.srilm_file:
			self.ppl_combine(self.iValid, self.lValid)
		else:
			print self.ppl(self.iValid, self.lValid)

		# save final net
		self.save(self.net_save_file)
