LSTM
====

LSTM Neural Network in Python and Cython, used for language modelling

Based on LSTM RNN, model proposed by Jürgen Schmidhuber
http://www.idsia.ch/~juergen/
Inspired by RNN LM toolkit by Tomas Mikolov
http://www.fit.vutbr.cz/~imikolov/rnnlm/

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


LICENCE:

	- lstm
		GPL

	- ArpaLM
		Copyright (c) 1999-2001 Carnegie Mellon University.  All rights reserved.

		Redistribution and use in source and binary forms, with or without
		modification, are permitted provided that the following conditions
		are met:

		1. Redistributions of source code must retain the above copyright
		notice, this list of conditions and the following disclaimer.

		2. Redistributions in binary form must reproduce the above copyright
		notice, this list of conditions and the following disclaimer in
		the documentation and/or other materials provided with the
		distribution.


INSTALLATION:

You will need:
		- python >= 2.6

		- cython >= 0.19
		(Win users see: http://www.lfd.uci.edu/~gohlke/pythonlibs/#cython)

		- c++ compiler
		(Win users:
		Visual Studio or MinGW:
		http://stackoverflow.com/questions/6034390/compiling-with-cython-and-mingw-produces-gcc-error-unrecognized-command-line-o
		http://wiki.cython.org/InstallingOnWindows
		gcc compile with -Ofast or -O3)

	Python libs:
		- numpy
		- argparse (is in python 2.7 and higher)

	Files:
		- LSTM.pyx
		- ArpaLM.py
		- setup.py
		- lstm.py

run
python setup.py build_ext --inplace --force


CAHNGE LOG:

0.5.0
- algorithm validation
- input args improvement

0.4.4
- revision of algorithm

0.4.1
- speed up (about 4x - not bad) due not multiplying zeros in input vector
- nbest recsore make more secure (flush file); ability of splitting nbest file

0.4.0
- saving and loading in new format (cPickle version and different attributes)
- first experiments made and published with this version :)

0.3.6
- numerical stability provided (underflow in some places in computation)
- computation of PPL corrected (!!)

0.3.5
- added nbest rescoring function

0.3.4
- saving and loading of RNN

0.3.3
- output reduced to classes

0.3.2
- Softmax changed (should be faster)
- trying to handle numerical stability

0.2
- rewritten to Cython
- first really working release
- support of more input types of vector

0.1
- early not fully working build
- in Python, rewritten from Java

TODO (just ideas - probably will not be implemented :) ):
	- make training faster
	{C like arrays ?}
	{}
	- compile to one file
	- clean the code
	- open text with codecs.utf-8
	- rescoring/ppl: more than one srilm model, more than one LSTM net
	- better handle IO errors and exceptions
