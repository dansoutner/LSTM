#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Simple implementation of LSTM RNN,
model proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

Implemented by Daniel Soutner,
Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
dsoutner@kky.zcu.cz, 2014; Licensed under the 3-clause BSD.
"""

import argparse
import LSTM
import sys

__version__ = LSTM.__version__

if __name__ == "__main__":
	DESCRIPTION = """
		Recurrent neural network based statistical language modelling toolkit
		(based on LSTM algorithm)
		Implemented by Daniel Soutner,
		Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
		dsoutner@kky.zcu.cz, 2013
		"""

	parser = argparse.ArgumentParser(description=DESCRIPTION, version=__version__)

	# arguments
	parser.add_argument('--train', nargs=3, action="store", metavar="FILE",
						help='Input training, test and validation files and train a RNN on them!')

	parser.add_argument('--hidden', action="store", dest='iHidden',
						help='Number of hidden neurons', type=int)

	parser.add_argument('--ppl', action="store", dest='ppl_file', metavar="FILE",
						help='Computes PPL of net on text file (if we train, do that after training)')

	parser.add_argument('--load-net', action="store", dest="load_net", default=None, metavar="FILE",
						help="Load RNN from file")

	parser.add_argument('--nbest-rescore', action="store", dest="nbest_rescore", default=None, metavar="FILE",
						help="Rescore with RNN the file of n-best hypothesis (with acoustic score)")

	parser.add_argument('--type', action="store", dest='input_type',
						help='Type of input vector', default='N',
						choices=("N", "FV", "FV+", "N+LDA", "FV+LDA"))

	parser.add_argument('--random-seed', action="store", dest='rnd_seed',
						help='Random seed used to init net', type=int, default=None)

	parser.add_argument('--wip', action="store", dest='wip',
						help='Word insertion penalty for nbest rescore (default is 0)', type=float, default=0)

	parser.add_argument('--lmw', action="store", dest='lmw',
						help='Language model weight for nbest rescore (default is 11)', type=float, default=11)

	parser.add_argument('--independent', action="store_true",
						help='Whether sentences should be independent', default=False)

	parser.add_argument('--debug', action="store_true",
						help='Whether to print debug output', default=False)

	parser.add_argument('--lda-dict', action="store", dest="lda_dict",
						help='LDA dictionary file (when LDA provided)')

	parser.add_argument('--lda-model', action="store", dest="lda_model",
						help='LDA model file (when LDA provided)')

	parser.add_argument('--class-file', action="store", dest="class_file",
						help='Class file (when class provided)')

	parser.add_argument('--srilm-file', action="store", dest="srilm_file",
						help='Srilm LM file (to computing combine model)')

	parser.add_argument('--pos-file', action="store", dest="pos_file",
						help='POS tags file [train test valid] (when POS provided)')

	parser.add_argument('--vocabulary', action="store", dest="vocabulary_file",
						help='file with vocabulary to use (a word per line), if not set will be created from training text')

	parser.add_argument('--lambda', action="store", dest='srilm_lambda',
						help='From 0 to 1, weight of RNN model in combination', type=float, default=0.5)

	parser.add_argument('--projections-file', action="store", dest="projections_file",
						help='Feature vectors file (when FV provided)')

	parser.add_argument('--classes', action="store", dest="output_classes", default=None, type=int,
						help='Learn output layer only to classes not words (should be faster)')

	parser.add_argument('--stopwords', action="store", dest="stopwords_file", default=None,
						help="Loads the text file with stopwords (used by LDA)")

	parser.add_argument('--cache', action='store', dest="len_cache", type=int, default=50,
						help="Length of word cache, default 50 words (used by LDA)")

	parser.add_argument('--cores', action="store", dest="num_threads", type=int, default=1,
						help="Number of cores used for computation (default is 1)")

	parser.add_argument('--alpha', action="store", dest="alpha", type=float, default=0.16,
						help="Learning coeficient (default is 0.16)")

	parser.add_argument('--save-net', action="store", dest="save_net", default=None, metavar="FILE",
						help="Save RNN to file")

	args = parser.parse_args()

	# if no args are passed
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()

	l = LSTM.LSTM(args)


