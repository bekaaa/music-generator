#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import _pickle as pickle
from loss import sequence_loss
import datetime as dt

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
		self.num_loops_in_epoch = self.num_batches // self.num_steps
		self.hidden_size = self.element_size * self.num_prev
		self.num_all_elem = self.batch_size * self.num_steps
		self.tensor_input_shape = [self.batch_size, self.num_steps, self.hidden_size]
		self.tensor_output_shape = [self.num_all_elem, self.element_size]
		#------------------------------------
	def next_batch(self):
		self.x = np.zeros((self.batch_size, self.num_steps, self.num_prev, self.element_size))
		self.y = np.zeros((self.batch_size, self.num_steps, self.element_size))
		for i in range(self.batch_size):
			for j in range(self.num_steps-1):
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
def Main(input_, epochs, num_layers=1, droprate=.3, init_scale=.05):
	#----------------------
	# define Graph
	graph = tf.Graph()
	with graph.as_default():
		# input placeholders
		tf_x = tf.placeholder(tf.float32, shape=input_.tensor_input_shape, name='training_input')
		tf_y = tf.placeholder(tf.float32, shape=input_.tensor_output_shape, name='training_output')
		tf_test = tf.placeholder(tf.float32, shape=input_.tensor_input_shape, name='test_input')

		# softmax layer variables
		# define weights and biases
		W = tf.Variable(tf.random_uniform([input_.hidden_size, input_.element_size], -init_scale,init_scale))
		b = tf.Variable(tf.zeros(input_.element_size))

		# set True for training
		is_training = tf.Variable(True, dtype=tf.bool)

		# define LSTM layer[s]
		# LSTM cell
		cell = tf.contrib.rnn.LSTMCell(input_.hidden_size, state_is_tuple=True)
		if is_training and droprate > 0 :
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1-droprate)
		if num_layers > 1:
			cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])
		#-----------------------------------------------------------------------
		# initial state for C and H
		if num_layers > 1 :
			init_state = tf.placeholder(tf.float32,[num_layers, 2, input_.batch_size, input_.hidden_size],\
				'init_state')
			state_per_layer_list = tf.unstack(init_state, axis=0)
			rnn_state_tuple = tuple(
				tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1])
				for idx in range(num_layers))
		else :
			init_state = tf.placeholder(tf.float32, [2, input_.batch_size, input_.hidden_size], 'init_state')
			rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(init_state[0], init_state[1])
		#--------------------------------------------------------------------------
		# -*-* Forward Propagation -*-*-*
		# add dropout to input_
		if is_training and droprate > 0 :
			tf_x = tf.nn.dropout(tf_x, keep_prob=1-droprate, name='input_dropout')
		# run LSTM
		output, state = tf.nn.dynamic_rnn(cell, tf_x, dtype=tf.float32, initial_state=rnn_state_tuple)
		# output is in shape [batch_size, num_steps, hidden_size]
		# state is in shape [batch_size, cell_state_size]

		# reshape output to 2 dimesions
		output = tf.reshape(output, [input_.num_all_elem, input_.hidden_size])

		# run softmax
		logits = tf.nn.xw_plus_b(output, W, b)
		# logits are in shape [num_all_elem, element_size] ie.(in the same shape as tf_y)
		#---------------------------------------------------------------------------------------
		# -*-* training cost and optimizer -*-*
		cost = sequence_loss(logits, tf_y)
		optimizer = tf.train.AdamOptimizer().minimize(cost)
		#-----------------------------------------------------
	#---------------------------------------------------------------------------------------------
	# -*-*-* DEFINE AND RUN SESSION -*-*-*-**-*-*
	# some variables
	#epochs = 1
	print_step = 50
	best_loss = 999
	global_step = 0
	save_model = False
	curr_time = dt.datetime.now()
	#------------------
	# define session and set our graph to it.
	sess = tf.InteractiveSession(graph=graph)
	print('initializing')
	# initialize placeholders and variables.
	sess.run(tf.global_variables_initializer())
	print('initialized')
	# define model saver for checkpoints
	saver = tf.train.Saver()
	# setting current state to zeros for the first step only.
	if num_layers == 1 :
		current_state = np.zeros([ 2, input_.batch_size, input_.hidden_size])
	else :
		current_state = np.zeros([num_layers, 2, input_.batch_size, input_.hidden_size])
	for epoch in range(epochs) :
		for step in range(input_.num_loops_in_epoch):
		#for i in range(5):
			#--------------------------------------
			# get next batch, define feed dict and run one step of RNN
			x, y = input_.next_batch()
			feed_dict = { tf_x : x, tf_y : y, init_state : current_state }
			loss, _, current_state = sess.run([cost, optimizer, state], feed_dict=feed_dict)
			#------------------------------------------
			if step % print_step == 0 :
				seconds = float((dt.datetime.now() - curr_time).seconds) / print_step
				curr_time = dt.datetime.now()
				print('Epoch {:>3}, step {:>5}, loss {:>6.4f}, current best loss {:>6.4f}, seconds per step {:>4.2f}'\
					.format(epoch, step, loss, best_loss, seconds))
			if loss < best_loss :
				best_loss = loss
				if save_model :
					saver.save(sess, '../checkpoints/1/model-', global_step=global_step)
					global_step += 1
					print('Model saved.')
	#---------------------------------------------------------------------------



#*************************************************************
if __name__ == '__main__' :
	from preprocessing import load_piece, rearrange_data
	num_prev = 5
	batch_size = 2
	num_steps = 2
	element_size = 19*3
	print('loading data')
	raw_data = load_piece('../data/Mozart/', 2)
	raw_data = rearrange_data(raw_data, num_prev)
	print('data loaded in shape',raw_data.shape)

	data_obj = Input(num_prev, batch_size, num_steps, element_size, raw_data)

	Main(data_obj, epochs=2, num_layers=2, droprate=.3)
