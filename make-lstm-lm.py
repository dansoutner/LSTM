# -*- coding : utf-8 -*-
"""
Simple example of LSTM RNN

dsoutner@kky.zcu.cz, 2013

"""
__version__ = "0.2"


import LSTM
import numpy as np
import re
import math

import time


import cProfile

def text_file_to_list_of_words(input_text_file):
	sText = open(input_text_file).read()
	sText = sText.replace("\n", " </s> ")	# convert eos
	sText = re.subn("[ ]+", " ",sText)[0]		# disable more than one space
	lText = sText.lower().split()			# to lowercase
	return lText

def index_to_vector(idx, dic_len):
	o = [0.] * dic_len
	o[idx] = 1.
	return np.array(o)


def ppl(iText, net):
	"""
	Computes perplexity of RNN on given text (splitted)
	"""
	ppl = 0.
	count = 0
	net.Reset()
	for i in xrange(len(iText)-1):
		o = net.forward(index_to_vector(iText[i], len(dic)))
		ppl += math.log(o[iText[i + 1]], 2)
		count += 1
	return math.pow(2, (-1.0 / count) * ppl)


if __name__ == "__main__":
	
	import sys
	
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
	lText = text_file_to_list_of_words(train_file)
	lTest = text_file_to_list_of_words(test_file)
	dic = sorted(list(set(lText + ["<unk>"])))
	
	
	iText = []
	for word in lText:
		try:
			iText.append(dic.index(word))
		except:
			iText.append(dic.index("<unk>"))
	
	iTest = []
	for word in lTest:
		try:
			iTest.append(dic.index(word))
		except:
			iTest.append(dic.index("<unk>"))
	
	print "Dictionary length: %d" % len(dic)
	print "Train text %d words" % len(iText)
	print "Test text %d words" % len(iTest)
	print "Hidden layer %d" % iHidden
	
	lstm = LSTM.LSTM(len(dic), len(dic), iHidden, rnd_seed=2)
	ALPHA = 0.1
	
	#p = [ppl(iTest, lstm)]
	p = [100000]
	iter = 0
	while True:
		start = time.time()
		for i in range(len(iText) - 1):
			inx = index_to_vector(iText[i],len(dic))
			outx = index_to_vector(iText[i + 1],len(dic))
			lstm.Next(inx, outx, learningRate=ALPHA)
			if i % 100 == 1:
				print i / (time.time() - start)
		lstm.Reset()
		p.append(ppl(iTest, lstm))
		print "%d, speed %.2f words/s, ppl: %.2f, alpha %.5f" % (iter, i / (time.time() - start), p[-1], ALPHA)
		iter += 1
	
		if p[-1]/p[-2] > 0.95:
			ALPHA = ALPHA/1.8
	
	print p

