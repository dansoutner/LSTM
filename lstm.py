#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Simple implementation of LSTM RNN,
model proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

Implemented by Daniel Soutner,
Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
dsoutner@kky.zcu.cz, 2013

Copyright (c) 2013, Daniel Soutner
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import LSTM
import sys
import time

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

	parser.add_argument('--srilm-file', action="store", dest="srilm_file",
						help='Srilm LM file (to computing combine model)')

	parser.add_argument('--vocabulary', action="store", dest="vocabulary_file",
						help='file with vocabulary to use (a word per line), if not set will be created from training text')

	parser.add_argument('--lambda', action="store", dest='srilm_lambda',
						help='From 0 to 1, weight of RNN model in combination', type=float, default=0.5)

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


