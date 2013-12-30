#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Simple implementation of LSTM RNN,
model proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

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


