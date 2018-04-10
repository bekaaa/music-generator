#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from lib.model_assets import sequence_loss
import datetime as dt

#****************************************************
def lstm_model(input_, c): # c is a Config object
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
		W = tf.Variable(tf.random_uniform([input_.hidden_size, input_.element_size], -c.init_scale,c.init_scale))
		b = tf.Variable(tf.zeros(input_.element_size))

		# set True for training
		is_training = tf.Variable(True, dtype=tf.bool)

		# define LSTM layer[s]
		# LSTM cell
		cell = tf.contrib.rnn.LSTMCell(input_.hidden_size, state_is_tuple=True)
		if is_training and c.droprate > 0 :
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1-c.droprate)
		if c.num_layers > 1:
			cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(c.num_layers)])
		#-----------------------------------------------------------------------
		# initial state for C and H
		if c.num_layers > 1 :
			init_state = tf.placeholder(tf.float32,[c.num_layers, 2, input_.batch_size, input_.hidden_size],\
				'init_state')
			state_per_layer_list = tf.unstack(init_state, axis=0)
			rnn_state_tuple = tuple(
				tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1])
				for idx in range(c.num_layers))
		else :
			init_state = tf.placeholder(tf.float32, [2, input_.batch_size, input_.hidden_size], 'init_state')
			rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(init_state[0], init_state[1])
		#--------------------------------------------------------------------------
		# -*-* Forward Propagation -*-*-*
		# add dropout to input_
		if is_training and c.droprate > 0 :
			tf_x = tf.nn.dropout(tf_x, keep_prob=1-c.droprate, name='input_dropout')
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
	best_loss = 99
	global_step = 0
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
	if c.num_layers == 1 :
		current_state = np.zeros([ 2, input_.batch_size, input_.hidden_size])
	else :
		current_state = np.zeros([c.num_layers, 2, input_.batch_size, input_.hidden_size])
	for epoch in range(c.epochs) :
		for step in range(input_.num_loops_in_epoch):
			#--------------------------------------
			# get next batch, define feed dict and run one step of RNN
			x, y = input_.next_batch()
			feed_dict = { tf_x : x, tf_y : y, init_state : current_state }
			loss, _, current_state = sess.run([cost, optimizer, state], feed_dict=feed_dict)
			#------------------------------------------
			if step % c.print_step == 0 :
				seconds = float((dt.datetime.now() - curr_time).seconds) / c.print_step
				curr_time = dt.datetime.now()
				print('Epoch  {:>9.2f}, step {:>5}, loss {:>6.4f}, '\
					'current best loss {:>7.4f}, seconds per step {:>4.2f}'\
					.format(epoch + (step+0.001)/input_.num_loops_in_epoch, step, loss, best_loss, seconds))
			if loss < best_loss :
				best_loss = loss
				if c.save_model and epoch >= c.save_after_epoch :
					if c.model_save_path == None :
						raise ValueError("model_save_path cann't be None")
					saver.save(sess, c.model_save_path, global_step=global_step)
					global_step += 1
					print('Model saved.')
	#---------------------------------------------------------------------------
#*******************************************************************************
