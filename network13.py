
import theano
import theano.tensor as T

import lasagne
from lasagne.layers.shape import * 
from lasagne.layers import *
from lasagne.nonlinearities import *

import sys
sys.path.append('/usr/lib/python2.7/dist-packages/')

import os
#import h5py
import math
import time
import random
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

# to include output_size in trenasposedconv2Dlayer
class CustomTransposedConv2DLayer(lasagne.layers.TransposedConv2DLayer):
	def __init__(self, *args, **kwargs):
		output_size = kwargs.pop('output_size', None)
		super(CustomTransposedConv2DLayer, self).__init__(*args, **kwargs)
		self.output_size = output_size

	def get_output_shape_for(self, input_shape):
		shape = super(CustomTransposedConv2DLayer, self).get_output_shape_for(input_shape)		
		if self.output_size is not None:
		    shape = shape[:2] + self.output_size
		#print shape
		return shape
global new_scale_factor
class My_Upscale2DLayer(lasagne.layers.Upscale2DLayer):
	
    	def __init__(self, *args, **kwargs):
		output_size = kwargs.pop('output_size', None)
		super(My_Upscale2DLayer, self).__init__(*args, **kwargs)
		self.output_size = output_size
		
	def get_output_shape_for(self, input_shape):
		output_shape = list(input_shape)  # copy / convert to mutable list
		in_shape =  output_shape[3]
		if self.output_size is not None:
		    output_shape[2] = output_shape[3] = self.output_size  # it is a Python int
		else:
		    output_shape[2] = output_shape[3] = None  # it is symbolic, so we don't know
		global new_scale_factor 
		new_scale_factor = output_shape[2] / in_shape
		return tuple(output_shape)
	def get_output_for(self, input, **kwargs):			
        	self.scale_factor = new_scale_factor
		a = self.scale_factor
		upscaled = input
		upscaled = T.extra_ops.repeat(upscaled, a, 3)
		upscaled = T.extra_ops.repeat(upscaled, a, 2)
		return upscaled
B =10
C=10
W=227
H=227


def get_net():
	net = {}
	net['input'] = InputLayer(shape=(None, C, W, H)) 
	print "input dimesion",net['input'].output_shape

	net['conv1'] =  Conv2DLayer(net['input'], 256, 11, stride=4, W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh) 
	print "conv1 dimension",net['conv1'].output_shape

	net['mpool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], 2)
	print "mpool1 dimension",net['mpool1'].output_shape

	net['conv2'] =  Conv2DLayer(net['mpool1'], 128, 5, pad=2, W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh) 
	print "conv2 dimension",net['conv2'].output_shape

	net['mpool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], 2)
	print "mpool2 dimension",net['mpool2'].output_shape

	net['conv3'] =  Conv2DLayer(net['mpool2'], 64, 3, pad=1, W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh) 
	print "conv2 dimension",net['conv3'].output_shape

	#################-------------------

	net['deconv1'] =  CustomTransposedConv2DLayer(net['conv3'], 128, 3, stride=1, crop=1, W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh,output_size=(13,13)) 
	print "deconv1 dimension",net['deconv1'].output_shape


	net['uppool1'] = My_Upscale2DLayer(net['deconv1'], 2,output_size=27)
	print "uppool1 dimension",net['uppool1'].output_shape


	net['deconv2'] =  CustomTransposedConv2DLayer(net['uppool1'], 256, 5, stride=1, crop=2, W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh,output_size=(27,27)) 
	print "deconv2 dimension",net['deconv2'].output_shape

	net['uppool2'] = My_Upscale2DLayer(net['deconv2'], 2,output_size=55)
	print "uppool2 dimension",net['uppool2'].output_shape

	net['deconv3'] =  CustomTransposedConv2DLayer(net['uppool2'], 10, 11,stride=4, W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh, output_size=(227,227)) 
	print "deconv3 dimension",net['deconv3'].output_shape

	
	net_output = lasagne.layers.get_output(net['deconv3'] )
	target_values=lasagne.layers.get_output(net['input'])
	loss = lasagne.objectives.squared_error(net_output,target_values).mean()
	weights = lasagne.regularization.regularize_network_params(net['deconv3'], lasagne.regularization.l2)
	loss += weights*1e-5
	all_params = lasagne.layers.get_all_params( net['deconv3'] ,trainable=True)
	updates = lasagne.updates.adagrad(loss, all_params, learning_rate=.0001, epsilon=1e-06)	
	train_fn =theano.function([net['input'].input_var], [net_output,loss], updates=updates)
	pred = lasagne.layers.get_output(net['deconv3'], deterministic=True)
	pred_fn = theano.function([net['input'].input_var], pred) 
	test_fn = theano.function([net['input'].input_var], loss) 	

	return net,train_fn,pred_fn,test_fn


s = time.time()
net,train_fn,pred_fn,test_fn = get_net()
print('Network created in {:.2f}s'.format(time.time() - s))
