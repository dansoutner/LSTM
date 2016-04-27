LSTM
====

**Project currently abandoned, similar toolkit for computing on GPU in LSTMLM repo.**


LSTM Neural Network in Python and Cython, used for language modelling

Based on LSTM RNN, model proposed by Jürgen Schmidhuber
http://www.idsia.ch/~juergen/
Inspired by RNN LM toolkit by Tomas Mikolov
http://www.fit.vutbr.cz/~imikolov/rnnlm/

Implemented by Daniel Soutner,
Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
dsoutner@kky.zcu.cz, 2013

Licensed under the 3-clause BSD.


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
- gensim (for LDA extension)

Files:
- LSTM.py
- ArpaLM.py
- setup.py
- lstm.py
- fast.pyx
- lda.py
- fastonebigheader.h


run
python setup.py build_ext --inplace --force

USAGE:

train LSTM LM on text and save
```
python lstm.py --train train.txt dev.txt test.txt --hidden 100 --save-net example.lstm-lm
```

sentences are processed independently (net is reset after every sentence), vocabulary limited to example.vocab (word-per-line)
```
python lstm.py --train train.txt dev.txt test.txt --hidden 100 --save-net example.lstm-lm --independent --vocabulary example.vocab
```

load net and evaluate on perplexity
```
python lstm.py --load-net example.lstm-lm --ppl valid2.txt
```

load net, combine with ARPA LM and evaluate
```
python lstm.py --load-net example.lstm-lm --ppl valid2.txt --srilm-file ngram.model.arpa --lambda 0.2
```

load net and rescore nbest list
```
python lstm.py --load-net example.lstm-lm --nbest-rescore nbest.list --wip 0 --lmw 11
```
