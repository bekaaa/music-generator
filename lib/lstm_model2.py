#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import _pickle as pickle
from loss import sequence_loss

class Input(object):
	def __init__(self,num_prev, batch_size, num_steps, element_size, raw_data):
		# expecting raw_data to be in shape [-1, num_prev, element_size ]
		assert raw_data.shape == (raw_data.shape[0], num_prev, element_size)
		# storing given parameters
		self.num_prev = num_prev
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.element_size = element_size
		self.data = raw_data
		# some other variable
		self.index = 0
		self.data_len = self.data.shape[0]
		self.num_batches = self.data_len // self.batch_size
		self.num_loops_in_epoch = (self.num_batches - 1) // self.num_steps
		self.hidden_size = self.element_size * self.num_prev
		self.num_all_elem = self.batch_size * self.num_steps
		self.tensor_input_shape = [self.batch_size, self.num_steps, self.hidden_size]
		self.tensor_output_shape = [self.num_all_elem, self.element_size]
		#------------------------------------
	def next_batch(self):
		self.x = np.zeros((self.batch_size, self.num_steps, self.num_prev, self.element_size))
		self.y = np.zeros((self.batch_size, self.num_steps, self.element_size))
		for i in range(self.batch_size):
			for j in range(self.num_steps):
				self.x[i,j] = self.data[self.index]
				self.index += 1
				self.y[i,j] = self.data[self.index, 0]
		# reshape X to 3 dimensions
		self.x = self.x.reshape(self.batch_size, self.num_steps, self.hidden_size)
		# reshape Y to 2 dimesions
		self.y = self.y.reshape(self.num_all_elem, self.element_size)

		return self.x, self.y
		#----------------------------------------
	def current_batch(self):
		try :
			return(self.x, self.y)
		except :
			raise('You need to run new_batch() for the first time.')
		#----------------------------------
#****************************************************
def Main(input_):
	# defining some param.
	# note that we will use many variables from input object
	droprate = .3
	init_scale = .05
	num_layers = 1
	#----------------------
	# define Graph
	graph = tf.Graph()
	with graph.as_default():
		# input placeholders
		tf_x = tf.placeholder(tf.float32, shape=input_.tensor_input_shape, name='training_input')
		tf_y = tf.placeholder(tf.float32, shape=input_.tensor_output_shape, name='training_output')

		# define LSTM layer
		# LSTM cell
		cell = tf.contrib.rnn.LSTMCell(input_.hidden_size)
		# initial state for C and H
		#initial_state = cell.zero_state(input_.batch_size, dtype=tf.float32)
		# run LSTM
		output, state = tf.nn.dynamic_rnn(cell, tf_x, dtype=tf.float32)#, initial_state=initial_state)
		# output is in shape [batch_size, num_steps, hidden_size]
		# state is in shape [batch_size, cell_state_size]

		# reshape output to 2 dimesions
		output = tf.reshape(output, [input_.num_all_elem, input_.hidden_size])

		# softmax layer
		# define weights and biases
		W = tf.Variable(tf.random_uniform([input_.hidden_size, input_.element_size], -init_scale,init_scale))
		b = tf.Variable(tf.zeros(input_.element_size))
		logits = tf.nn.xw_plus_b(output, W, b)
		# logits are in shape [num_all_elem, element_size] ie.(in the same shape as tf_y)

		# compute cost
		cost = sequence_loss(logits, tf_y)

		# add optimizer
		optimizer = tf.train.AdamOptimizer().minimize(cost)
	#---------------------------------------------------------------------------------------------
	# -*-*-* DEFINE AND RUN SESSION -*-*-*-**-*-*
	epochs = 1
	sess = tf.InteractiveSession(graph=graph)
	print('initializing')
	sess.run(tf.global_variables_initializer())
	print('initialized')
	#for i in range(input_.num_loops_in_epoch):
	for i in range(5):
		x, y = input_.next_batch()
		feed_dict = { tf_x : x, tf_y : y }
		loss, _, current_state = sess.run([cost, optimizer, state], feed_dict=feed_dict)
		print('Epoch {}, step {}, loss {:.3f}'.format(0, i, loss))
	#---------------------------------------------------------------------------



#*************************************************************
if __name__ == '__main__' :
	from preprocessing import load_piece, rearrange_data
	num_prev = 10
	batch_size = 2
	num_steps = 2
	element_size = 19*3
	raw_data = load_piece('../data/Mozart/', 2)
	raw_data = rearrange_data(raw_data, num_prev)
	print(raw_data.shape)

	data_obj = Input(num_prev, batch_size, num_steps, element_size, raw_data)

	Main(data_obj)
