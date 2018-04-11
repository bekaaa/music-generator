#!/usr/bin/env python3
import datetime as dt
from __init__ import tf, np, sequence_loss
#from __init__ import *

class MusicLSTM(object):
	def __init__(self, input_, config):
		self.i = input_
		self.c = config
		# shared variables
		self.graph = None
		self.tf_x = None
		self.tf_y = None
		self.is_training = None
		self.init_state = None
		self.state = None
		self.logits = None
		self.cost = None
		self.optimizer = None
	#---------------------------------
	def def_graph(self):
		'''
		constructs a Tensorflow graph
		input : input object, config object
		output : tf.graph object
		'''
		self.graph = tf.Graph()
		with self.graph.as_default():
			# input placeholders
			with tf.variable_scope('data_tensors', reuse=tf.AUTO_REUSE):
				self.tf_x = tf.placeholder(tf.float32, shape=self.i.tensor_input_shape, name='data_input')
				self.tf_y = tf.placeholder(tf.float32, shape=self.i.tensor_output_shape, name='data_output')

			# softmax layer variables
			# define weights and biases
			with tf.variable_scope('softmax_Wb', reuse=tf.AUTO_REUSE):
				W = tf.Variable(tf.random_uniform([self.i.hidden_size, self.i.element_size],\
					-self.c.init_scale, self.c.init_scale), name='W')
				b = tf.Variable(tf.zeros(self.i.element_size), name='b')

			# set True for training
			self.is_training = tf.Variable(True, dtype=tf.bool, name='training_flag', trainable=False)

			# define LSTM layer[s]
			# LSTM cell
			with tf.variable_scope('lstm_cell', reuse=tf.AUTO_REUSE):
				cell = tf.contrib.rnn.LSTMCell(self.i.hidden_size, state_is_tuple=True, name='cell')
				if self.is_training and self.c.droprate > 0 :
					cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1-self.c.droprate)
				if self.c.num_layers > 1:
					cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.c.num_layers)])
			#-----------------------------------------------------------------------
			# initial state for C and H
			with tf.variable_scope('initial_state', reuse=tf.AUTO_REUSE):
				if self.c.num_layers > 1 :
					self.init_state = tf.placeholder(tf.float32,[self.c.num_layers, 2,\
						self.i.batch_size, self.i.hidden_size], 'init_state')
					state_per_layer_list = tf.unstack(self.init_state, axis=0)
					rnn_state_tuple = tuple(
						tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1])
						for idx in range(self.c.num_layers))
				else :
					self.init_state = tf.placeholder(tf.float32, [2, self.i.batch_size, self.i.hidden_size], 'init_state')
					rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(self.init_state[0], self.init_state[1])
			#--------------------------------------------------------------------------
			# -*-* Forward Propagation -*-*-*
			with tf.variable_scope('Forward_propagation', reuse=tf.AUTO_REUSE):
				# add dropout to input_
				if self.is_training and self.c.droprate > 0 :
					self.tf_x = tf.nn.dropout(self.tf_x, keep_prob=1-self.c.droprate, name='input_dropout')
				# run LSTM
				output, self.state = tf.nn.dynamic_rnn(cell, self.tf_x, dtype=tf.float32, initial_state=rnn_state_tuple)
				# output is in shape [batch_size, num_steps, hidden_size]
				# state is in shape [batch_size, cell_state_size]

				# reshape output to 2 dimesions
				output = tf.reshape(output, [self.i.num_all_elem, self.i.hidden_size], name='output')

				# run softmax
				self.logits = tf.nn.xw_plus_b(output, W, b, name='logits')
				# logits are in shape [num_all_elem, element_size] ie.(in the same shape as tf_y)
			#---------------------------------------------------------------------------------------
			# -*-* training cost and optimizer -*-*
			with tf.variable_scope('cost_and_optimizer', reuse=tf.AUTO_REUSE):
				self.cost = sequence_loss(self.logits, self.tf_y)
				self.optimizer = tf.train.AdamOptimizer(name='Adam').minimize(self.cost)
			#-----------------------------------------------------
		return
	#------------------------------------
	def initial_state(self):
		if self.c.num_layers == 1 :
			return np.zeros([ 2, self.i.batch_size, self.i.hidden_size])
		else :
			return np.zeros([self.c.num_layers, 2, self.i.batch_size, self.i.hidden_size])
	#---------------------------------------
	def run_training_session(self):
		# some variables
		best_loss = 99
		global_step = 0
		curr_time = dt.datetime.now()
		#------------------
		# define session and set our graph to it.
		sess = tf.InteractiveSession(graph=self.graph)
		print('initializing ...', end='  ')
		# initialize placeholders and variables.
		sess.run(tf.global_variables_initializer())
		print('Done')
		# define model saver for checkpoints
		saver = tf.train.Saver(name='saver')
		# save graph one time only.
		if self.c.save_model and self.c.model_save_path :
			print('Saving graph to disk ...', end='  ')
			saver.save(sess, self.c.model_save_path, global_step=None)
			print('Done.')
		# setting current state to zeros for the first step only.
		current_state = self.initial_state()
		for epoch in range(self.c.epochs) :
			for step in range(self.i.num_loops_in_epoch):
				#--------------------------------------
				# get next batch, define feed dict and run one step of RNN
				x, y = self.i.next_batch()
				feed_dict = { self.tf_x : x, self.tf_y : y, self.init_state : current_state }
				loss, _, current_state = sess.run([self.cost, self.optimizer, self.state], feed_dict=feed_dict)
				#------------------------------------------
				if step % self.c.print_step == 0 :
					seconds = float((dt.datetime.now() - curr_time).seconds) / self.c.print_step
					curr_time = dt.datetime.now()
					print('Epoch  {:>3}-{:>3}, step {:>3}-{:>3}, loss {:>6.4f}, '\
						'current best loss {:>7.4f}, seconds per step {:>4.2f}'\
						.format(epoch, self.c.epochs, step, self.i.num_loops_in_epoch, loss, best_loss, seconds))
				if loss < best_loss :
					best_loss = loss
					if self.c.save_model and epoch >= self.c.save_after_epoch and self.c.model_save_path :
						print('Saving model ...', end='  ')
						saver.save(sess, self.c.model_save_path, global_step=global_step, write_meta_graph=False)
						global_step += 1
						print('Done.')
		#---------------------------------------------------------------------------
		sess.close()
		return
	#------------------------------------------
	def train(self):
		print('constructing the graph ...', end='  ')
		self.def_graph()
		# save graph to disk
		print('Done.\nNow the training session will begin.')
		self.run_training_session()
		print('Finished Training.')
#****************************************************
