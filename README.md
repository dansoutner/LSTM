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

USAGE:

train LSTM LM on text and save

python lstm.py --train train.txt test.txt valid.txt --hidden 100 --save-net example.lstm-lm

sentences are processed independently (net is reset after every sentence), vocabulary limited to example.vocab (word-per-line)

python lstm.py --train train.txt test.txt valid.txt --hidden 100 --save-net example.lstm-lm --independent --vocabulary example.vocab

load net and evaluate on perplexity

python lstm.py --load-net example.lstm-lm --ppl valid2.txt

load net, combine with ARPA LM and evaluate

python lstm.py --load-net example.lstm-lm --ppl valid2.txt --srilm-file ngram.model.arpa --lambda 0.2

load net and rescore nbest list

python lstm.py --load-net example.lstm-lm --nbest-rescore nbest.list --wip 0 --lmw 11
