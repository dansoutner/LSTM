# -*- coding:utf-8 -*-

# to be included from LSTM.pyx

import gensim
import codecs
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LDA():

	def __init__(self, stopwords):
		"""Init with text stopwords file"""
		#self.STOPWORDS = set(open(stopwords_file, "r").read().split())
		self.STOPWORDS = set(stopwords)

	def train_BH(self, file_corpus, file_dictionary, TOPICS):
		"""
		Train LDA model and save it
		"""
		#prepare corpus
		text = []
		for line in open(file_corpus):
			text.append(line.lower().strip().split())

		#remove stopwords
		text = [[word for word in line if word not in self.STOPWORDS] for line in text]

		splitted_coprus = []
		foo = []
		for i in range(len(text)):
			#if (i % self.SENTENCES) == (self.SENTENCES - 1):
			try:
				text[i][0]
			except:
				continue

			if text[i][0] == "_____":
				splitted_coprus.append(foo)
				foo = []
			foo += text[i]

		#print splitted_corpus

		self.dictionary = gensim.corpora.Dictionary(splitted_coprus)
		self.dictionary.save(file_dictionary) # store the dictionary, for future reference

		corpus = [self.dictionary.doc2bow(text) for text in splitted_coprus]
		gensim.corpora.MmCorpus.serialize('.mm', corpus) # store to disk, for later use

		# go LDA
		self.lda = gensim.models.LdaModel(corpus, id2word=self.dictionary, num_topics=TOPICS, passes=10, chunksize=100) # initialize an LSI transformation


	def train(self, file_corpus, file_dictionary, TOPICS, SENTENCES=10):
		"""
		Train LDA model and save it
		"""
		#prepare corpus
		text = []
		for line in open(file_corpus):
			text.append(line.lower().strip().split())

		#remove stopwords
		text = [[word for word in line if word not in self.STOPWORDS] for line in text]

		splitted_corpus = []
		foo = []
		for i in range(len(text)):
			if (i % SENTENCES) == (SENTENCES - 1):
				splitted_corpus.append(foo)
				foo = []
			foo += text[i]

		self.dictionary = gensim.corpora.Dictionary(splitted_corpus)
		self.dictionary.save(file_dictionary) # store the dictionary, for future reference

		corpus = [self.dictionary.doc2bow(text) for text in splitted_corpus]
		gensim.corpora.MmCorpus.serialize('.mm', corpus) # store to disk, for later use

		# go LDA
		self.lda = gensim.models.LdaModel(corpus, id2word=self.dictionary, num_topics=TOPICS, passes=10, chunksize=100) # initialize an LSI transformation


	def save(self, file_model, file_dictionary):
		"""Saves the LDA model"""
		self.lda.save(file_model)
		self.dictionary.save(file_dictionary)
		print self.dictionary


	def load(self, file_model, file_dictionary):
		"""Loads the LDA model"""
		self.lda = gensim.models.LdaModel.load(file_model)
		self.dictionary = gensim.corpora.Dictionary.load(file_dictionary)
		print self.dictionary


	def cache_to_fv(self, cache):
		"""Converts cahce (list of lowercased words) to FV"""
		vec_bow = self.dictionary.doc2bow(cache)
		vec_lda = self.lda.__getitem__(vec_bow, eps=0) # convert the query to LDA space
		out = []
		for x in vec_lda:
			out.append(x[1])
		return out


if __name__ == "__main__":
	#l = LDA("cz_stopwords.txt")
	stopwords_file = "cz_stopwords.txt"
	STOPWORDS = set(open(stopwords_file, "r").read().split())
	l = LDA(STOPWORDS)
	import sys
	train_file = sys.argv[1]
	test_file = sys.argv[2]

	#TOPICS = 50
	#LAST_N = 50 # how many words count as a history?
	for TOPICS in [20, 30, 40, 50, 70, 100]:

		l.train_BH(train_file, train_file+".dict", TOPICS)
		l.save(train_file + str(TOPICS) + ".lda", train_file + str(TOPICS) + ".dict")
		l.load(train_file + str(TOPICS) +".lda", train_file + str(TOPICS) + ".dict")
			
		"""
		stopwords_file = "cz_stopwords.txt"
		STOPWORDS = set(open(stopwords_file, "r").read().split())
		#print l.cache_to_fv("this is a very big test of japan")
		#t = open(test_file).read().lower().split()

		for i in range(len(t)):
			# last N words
			cache = t[i - LAST_N : i]
			# remove stopwords
			cache = [word for word in cache if word not in STOPWORDS]
			print cache
			print l.cache_to_fv(cache)
		"""
