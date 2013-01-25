# -*- coding:utf-8 -*-
"""
Simple implementation of LSTM RNN,
proposed by Jürgen Schmidhuber, http://www.idsia.ch/~juergen/

dsoutner@kky.zcu.cz, 2012

"""
__version__ = "0.1"


#imports
import numpy
import random
import math
import time


# speed up hacks and shortcuts
now = time.time
mathlog = math.log
rndnormal = random.normalvariate

npdot = numpy.dot
npexp = numpy.exp
npsum = numpy.sum
nparray = numpy.array
npzeros = numpy.zeros


## __func__
def msigmoid(input):
	return (numpy.exp(input) + 1.0 ) / 2.0

def msigmoid_f(input):
	return (1.0 / (1 + numpy.exp(-input)))

def msigmoid_f_d(input):
	r = msigmoid_f(input)
	return r * (1 - r)

def msigmoid_g(input):
	return ((4.0 / (1 + numpy.exp(-input))) - 2.0)

def msigmoid_g_d(input):
	r = msigmoid_g(input)
	return r * (1 - r)

def msigmoid_h(input):
	return ((2.0 / (1 + numpy.exp(-input))) - 1.0)

def msigmoid_h_d(input):
	r = msigmoid_h(input)
	return r * (1 - r)

def msoftmax(input):
	output = nparray(npzeros(input.shape))
	output[:] = npexp(input)
	output /= sum(output)
	return output

def msoftmax_d(input):
	m = msoftmax(input)
	return m - numpy.power(m, 2)


def rnd_array(shape, MU=0, SIGMA=0.1):
	o = numpy.zeros(shape)
	for x in range(o.shape[0]):
		for y in range(o.shape[1]):
			o[x, y] = rndnormal(MU, SIGMA)
	return o


""" LSTM net dimensions """
iCells = 2 # iCells = no. of cells in block
iBlocks = 3 # iBlocks = no. of LSTM block
iInputs = 5 # iOutputs = output net layer
iOutputs = 7 # iInputs = input net layer


class MemoryBlock():

	def __init__(self):
		""" LSTM inner states """
		self.scX = npzeros((iBlocks, iCells))		# last state of LSTM
		self.sc = npzeros((iBlocks, iCells))		# currnet state of LSTM
		self.y_cX = npzeros((iCells*iBlocks, 1))	# last output of LSTM
		self.y_c = npzeros((iCells*iBlocks, 1))		# current output of LSTM


		""" LSTM randomize """
		self.Vc = rnd_array((iBlocks, iCells))
		self.Vin = rnd_array((iBlocks, iCells))
		self.Vout = rnd_array((iBlocks, iCells))
		self.Vfi = rnd_array((iBlocks, iCells))

		self.Uc = rnd_array((iBlocks, iInputs))
		self.Uin = rnd_array((iBlocks, iInputs))
		self.Uout = rnd_array((iBlocks, iInputs))
		self.Ufi = rnd_array((iBlocks, iInputs))

		self.Wfi = rnd_array((iBlocks, iCells))
		self.Win = rnd_array((iBlocks, iCells))
		self.Wout = rnd_array((iBlocks, iCells))

		self.Y = rnd_array((iOutputs, iCells*iBlocks))

		# gradients
		self.dS_c = npzeros((iBlocks, iCells))
		self.dS_in = npzeros((iBlocks, iCells))
		self.dS_fi = npzeros((iBlocks, iCells))
		self.dS_in_c = npzeros((iBlocks, iCells))
		self.dS_fi_c = npzeros((iBlocks, iCells))

	def forward(self, y):

		self.y_cX = self.y_c # save last output
		self.scX = self.sc # save last output

		# net_fi
		sum_Vfi = 0
		for j in range(iBlocks):
			sum_Vfi += npdot(self.Vfi[j].reshape((1, -1)), self.y_cX[iCells*j:iCells*(j+1)])

		self.net_fi = npzeros((iBlocks, 1))
		self.y_fi = npzeros((iBlocks, 1))
		for j in range(iBlocks):
			self.net_fi[j] = npdot(self.Ufi[j], y) + sum_Vfi + sum(self.Wfi[j] * self.sc[j])
			self.y_fi[j] = msigmoid_f(self.net_fi[j])

		# net_in
		sum_Vin = 0
		for j in range(iBlocks):
			sum_Vin += npdot(self.Vin[j].reshape((1, -1)), self.y_cX[iCells*j:iCells*(j+1)])

		self.net_in = npzeros((iBlocks, 1))
		self.y_in = npzeros((iBlocks, 1))
		for j in range(iBlocks):
			self.net_in[j] = npdot(self.Uin[j], y) + sum_Vin + sum(self.Win[j] * self.sc[j])
			self.y_in[j] = msigmoid_f(self.net_in[j])

		# net_c
		sum_Vc = 0
		for j in range(iBlocks):
			sum_Vc += npdot(self.Vc[j], self.y_cX[iCells*j:iCells*(j+1)])

		self.net_c = npzeros((iBlocks, iCells))
		for j in range(iBlocks):
			for v in range(iCells):
				self.net_c[j, v] = npdot(self.Uc[j], y) + sum_Vc

		# s_c
		s = npzeros((iBlocks, iCells))
		for j in range(iBlocks):
			for v in range(iCells):
				s[j][v] = self.y_fi[j] * self.sc[j, v] + self.y_in[j] * msigmoid_g(self.net_c[j][v])
		self.sc = s

		# net_out
		sum_Vout = 0
		for j in range(iBlocks):
			sum_Vout += npdot(self.Vout[j], self.y_cX[iCells*j:iCells*(j+1)])

		self.net_out = npzeros((iBlocks, 1))
		self.y_out = npzeros((iBlocks, 1))
		for j in range(iBlocks):
			self.net_out[j] = npdot(self.Uout[j], y) + sum(self.Wout[j] * self.sc[j])
			self.y_out[j] = msigmoid_f(self.net_out[j])

		# y_c
		for j in range(iBlocks):
			for v in range(iCells):
				self.y_c[j*iCells + v] = self.y_out[j] * self.sc[j,v]

		# output of net
		self.net_k = npdot(self.Y, self.y_c)
		self.y_k = msoftmax(self.net_k)

		return self.y_k


	def backward(self, input, output, alpha=0.1):
		"""Learns NET supervised - for (input) required (output) with leraning rate alpha"""

		# forward pass
		o = self.forward(input)

		# compute derivatives
		for j in range(iBlocks):
			for v in range(iCells):
				self.dS_c[j,v] = self.dS_c[j,v] * self.y_fi[j] + msigmoid_g_d(self.net_c[j,v]) * self.y_in[j] * self.y_cX[iCells*j + v]
				self.dS_in[j,v] = self.dS_in[j,v] * self.y_fi[j] + msigmoid_g(self.net_c[j,v]) * msigmoid_f_d(self.net_in[j]) * self.y_cX[iCells*j + v]
				self.dS_fi[j,v] = self.dS_fi[j,v] * self.y_fi[j] + self.sc[j][v] * msigmoid_f_d(self.net_fi[j]) * self.y_cX[iCells*j + v]
				for v_ in range(iCells):
					self.dS_in_c[j][v] = self.dS_in_c[j][v] * self.y_fi[j] + msigmoid_g(self.net_c[j][v]) * msigmoid_f_d(self.net_in[j]) * self.scX[j][v_]
					self.dS_fi_c[j][v] = self.dS_fi_c[j][v] * self.y_fi[j] + self.scX[j][v] * msigmoid_f_d(self.net_fi[j]) * self.scX[j][v_]

		# injection error
		e_k = output - o

		print "e_k",e_k

		# delta_s of output units
		
		#d_k = npdot(msoftmax_d(self.net_out), e_k.T)
		d_k = (msoftmax_d(self.net_k) * e_k)
		
		print npdot(e_k.T, self.Y)
		print self.net_fi.T.shape
		print npdot(e_k, self.net_fi.T)
		d_fi = npdot(msigmoid_f_d(self.net_fi.T), e_k)
		d_out = npdot(msigmoid_f_d(self.net_out), e_k)
		d_in = npdot(msigmoid_f_d(self.net_in).T, e_k)
		
		self.Y += alpha * npdot(d_k, self.y_c.T)

		print "d_k", d_k

		d_out = npzeros((iBlocks, 1))
		e_sc = npzeros((iBlocks, iCells))
		for j in range(iBlocks):
			s = 0
			for v in range(iCells):
				print self.Wfi[j,v]
				print d_fi
				s += self.sc[j, v] * (self.Wfi[j,v] * d_fi + self.Wout[j,v] * d_out + self.Win[j,v] * d_in)
			
			print s.shape
			
			d_out[j] = msigmoid_f_d(self.net_out[j]) * s
			for v in range(iCells):
				e_sc[j][v] = self.y_out[j] * msigmoid_h(self.sc[j][v]) * (self.Wfi[j,v] * d_fi + self.Wout[j,v] * d_out + self.Win[j,v] * d_in)

		# update weights
		for j in range(iBlocks):

			# output gates
			self.Vout[j] += alpha * d_out[j]
			self.Uout[j] += alpha * d_out[j]
			for v in range(iCells):
				self.Wout[j,v] += alpha * d_out[j] * self.sc[j][v]

			# input gates
			s = 0
			for v in range(iCells):
				s += e_sc[j][v] * self.dS_in[j][v]

			self.Uin[j] += alpha * s
			self.Vin[j] += alpha * s

			for v_ in range(iCells):
				s = 0
				for v in range(iCells):
					s += e_sc[j][v] * self.dS_in_c[j][v_]

				self.Win[j][v_] += alpha * s

			# forget gates
			s = 0
			for v in range(iCells):
				s += e_sc[j][v] * self.dS_fi_c[j][v]

			self.Ufi[j] += alpha * s
			self.Wfi[j] += alpha * s

			for v_ in range(iCells):
				s = 0
				for v in range(iCells):
					s += e_sc[j][v] * self.dS_c[j][v_]
				self.Vfi = self.Vfi + alpha * s

			# cells
			for v in range(iCells):
				self.Uc[j][v] += alpha * e_sc[j][v] * self.dS_c[j][v]
				self.Vc[j][v] += alpha * e_sc[j][v] * self.dS_c[j][v]



if __name__ == "__main__":


	y = nparray([1, 0, 0, 0, 0]).reshape(iInputs, 1)
	o = nparray([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)

	m = MemoryBlock()
	#print m.forward(y)
	print m.backward(y, o, alpha=0.5)
	print m.backward(y, o, alpha=0.5)
	print m.backward(y, o, alpha=0.5)
	print m.backward(y, o, alpha=0.5)
	print m.backward(y, o, alpha=0.5)
	print m.forward(y)
	"""
	print m.forward(y)
	print m.forward(y)
	print m.forward(y)
	print m.forward(y)
	print m.forward(y)
	print m.forward(y)
	"""
