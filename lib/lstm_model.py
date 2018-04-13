#!/usr/bin/env python3
import datetime as dt
#from __init__ import tf, np, sequence_loss
from __init__ import *

class MusicLSTM(object):
	def __init__(self, input_, config):
		self.i = input_
		self.c = config
		# shared variables
		self.graph = None
		self.sess = None
		self.tf_x = None
		self.tf_y = None
		self.is_training = None
		self.init_state = None
		self.state = None
		self.logits = None
		self.cost = None
		self.optimizer = None
		self.saver = None
		self.global_step = None
		self.curr_time = None
		#-------------------

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
				# Add dropout.
				if self.is_training and self.c.droprate > 0 :
					cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1-self.c.droprate)
				# cells
				if self.c.num_layers > 1:
					cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.c.num_layers)])
			#-----------------------------------------------------------------------
			# initial state for C and H
			with tf.variable_scope('initial_state', reuse=tf.AUTO_REUSE):
				if self.c.num_layers == 1 :
					self.init_state = tf.placeholder(tf.float32,[2, self.c.batch_size, self.i.hidden_size],'init_state')
					rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(self.init_state[0], self.init_state[1])
				else :
					self.init_state = tf.placeholder(tf.float32,[self.c.num_layers,2, self.c.batch_size, self.i.hidden_size],\
						'init_state')
					state_per_layer_list = tf.unstack(self.init_state, axis=0)
					rnn_state_tuple = tuple(
						tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1])
						for idx in range(self.c.num_layers))
			#--------------------------------------------------------------------------
			# -*-* Forward Propagation -*-*-*
			with tf.variable_scope('forward_propagation', reuse=tf.AUTO_REUSE):
				# add dropout to input_
				if self.is_training and self.c.droprate > 0 :
					self.tf_x = tf.nn.dropout(self.tf_x, keep_prob=1-self.c.droprate, name='input_dropout')
				# run LSTM
				output, self.state = tf.nn.dynamic_rnn(cell, self.tf_x, dtype=tf.float32, initial_state=rnn_state_tuple)
				# output is in shape [batch_size, num_steps, hidden_size]
				# state is in shape [num_layers, batch_size, cell_state_size]
				self.state = tf.identity(self.state, 'output_state')

				# reshape output to 2 dimesions
				output = tf.reshape(output, [self.i.num_all_elem, self.i.hidden_size], name='output')
				# run softmax
				self.logits = tf.nn.xw_plus_b(output, W, b, name='logits')
				# logits are in shape [num_all_elem, element_size] ie.(in the same shape as tf_y)
			#---------------------------------------------------------------------------------------
			# -*-* training cost and optimizer -*-*
			with tf.variable_scope('cost', reuse=tf.AUTO_REUSE):
				self.cost = sequence_loss(self.logits, self.tf_y)
				self.cost = tf.identity(self.cost, 'cost')
			with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
				self.optimizer = tf.train.AdamOptimizer(name='Adam').minimize(self.cost, name='optimizer')
			#-----------------------------------------------------
		return
	#---------------------------------------
	def run_training_session(self):
		# some variables
		self.best_loss = 99
		self.global_step = 0
		self.curr_time = dt.datetime.now()
		#------------------
		# define session and set our graph to it.
		self.sess = tf.InteractiveSession(graph=self.graph)
		print('initializing ...', end='  ')
		# initialize placeholders and variables.
		self.sess.run(tf.global_variables_initializer())
		print('Done')
		# define model saver for checkpoints
		self.saver = tf.train.Saver(name='saver')
		# save graph one time only.
		if self.c.save_model and self.c.model_save_path :
			print('Saving graph to disk ...', end='  ')
			self.saver.save(self.sess, self.c.model_save_path, global_step=None)
			print('Done.')
		# reset saver so it doesn't delete te saved graph
		self.saver = tf.train.Saver(name='saver', max_to_keep=1)
		#-----------------------------------------
		# setting current state to zeros for the first step only.
		current_state = np.zeros([self.c.num_layers, 2, self.i.batch_size, self.i.hidden_size])
		if self.c.num_layers == 1:
			current_state = current_state[0]
		#-------------------------------------------
		# LOOP
		for epoch in range(self.c.epochs) :
			for step in range(self.i.num_loops_in_epoch):
				current_state = self.training_step(epoch, step, current_state)
				#--------------------------------------------------------
		self.sess.close()
		return
	#-----------------------------------------------------------
	def training_step(self, epoch, step, current_state):
		#--------------------------------------
		# get next batch, define feed dict and run one step of RNN
		x, y = self.i.next_batch()
		feed_dict = { self.tf_x : x, self.tf_y : y, self.init_state : current_state }
		loss, _, current_state = self.sess.run([self.cost, self.optimizer, self.state], feed_dict=feed_dict)
		#------------------------------------------
		if loss < self.best_loss :
			self.best_loss = loss
			if self.c.save_model and epoch >= self.c.save_after_epoch and self.c.model_save_path :
				print('Saving model ...', end='  ')
				self.saver.save(self.sess, self.c.model_save_path, global_step=None, write_meta_graph=False)
				#self.global_step += 1
				print('Done.')
		#-----------------------------------------------
		if step % self.c.print_step == 0 :
			seconds = float((dt.datetime.now() - self.curr_time).seconds) / self.c.print_step
			self.curr_time = dt.datetime.now()
			print('Epoch  {:>3}-{:>3}, step {:>3}-{:>3}, loss {:>6.4f}, '\
				'current best loss {:>7.4f}, seconds per step {:>4.2f}'\
				.format(epoch, self.c.epochs, step, self.i.num_loops_in_epoch, loss, self.best_loss, seconds))
		#------------------------------------
		return current_state
	#------------------------------------------
	def train(self, restore_from = None):
		tf.reset_default_graph()
		if restore_from :
			print('loading existing model ...', end='  ')
			expected_files = [restore_from + ext for ext in ['.meta', '.data-00000-of-00001','.index']]
			assert set(glob.glob(restore_from+'*')) == set(expected_files)
			self.sess = tf.InteractiveSession()
			saver = tf.train.import_meta_graph(restore_from + '.meta')
			saver = saver.restore(self.sess, restore_from)
			self.graph = tf.get_default_graph()
			print('Done.\nTraining is starting')
			#---------------------------------------
			# restore tensors and operations.
			self.restore_tensors()
			self.retrain()
		#-------------------------------
		else :
			print('constructing new graph ...', end='  ')
			self.def_graph()
			# save graph to disk
			print('Done.\nNow the training session will begin.')
			self.run_training_session()
			print('Finished Training.')
		#---------------------------------------
	#-----------------------------------------------
	def restore_tensors(self, predicting=False):

		self.tf_x =			self.graph.get_tensor_by_name('data_tensors/data_input:0')
		self.is_training =	self.graph.get_tensor_by_name('training_flag:0')
		self.init_state =	self.graph.get_tensor_by_name('initial_state/init_state:0')
		if predicting :
			self.logits =	self.graph.get_tensor_by_name('forward_propagation/logits:0')
			return
		self.tf_y =			self.graph.get_tensor_by_name('data_tensors/data_output:0')
		self.state =		self.graph.get_tensor_by_name('forward_propagation/output_state:0')
		self.cost = 		self.graph.get_tensor_by_name('cost/cost:0')
		self.optimizer = 	self.graph.get_operation_by_name('optimizer/optimizer')
	#---------------------------
	def retrain(self):
		#some variables
		self.best_loss = 99
		self.curr_time = dt.datetime.now()
		#------------------
		# define model saver for checkpoints
		self.saver = tf.train.Saver(name='saver')
		# save graph one time only.
		if self.c.save_model and self.c.model_save_path :
			print('Saving graph to disk ...', end='  ')
			self.saver.save(self.sess, self.c.model_save_path, global_step=None)
			print('Done.')
		# reset saver to prevent deleting the above graph
		self.saver = tf.train.Saver(name='saver', max_to_keep=1)
		# setting current state to zeros for the first step only.
		current_state = np.zeros([self.c.num_layers, 2, self.i.batch_size, self.i.hidden_size])
		if self.c.num_layers == 1:
			current_state = current_state[0]
		# LOOP
		for epoch in range(self.c.epochs) :
			for step in range(self.i.num_loops_in_epoch):
				current_state = self.training_step(epoch, step, current_state)
				#--------------------------------------------------------
		self.sess.close()
		return
	#------------------------------
	def init_predictor(self, model_path):
		tf.reset_default_graph()
		print('loading existing model ...', end='  ')
		expected_files = [model_path + ext for ext in ['.meta', '.data-00000-of-00001','.index']]
		assert set(glob.glob(model_path+'*')) == set(expected_files)
		self.sess = tf.InteractiveSession()
		saver = tf.train.import_meta_graph(model_path + '.meta')
		saver = saver.restore(self.sess, model_path)
		self.graph = tf.get_default_graph()
		print('Graph is ready.\nRestoring tensors ...', end='  ')
		self.restore_tensors(predicting = True)
		print('Done.')
	#-----------------------------------------------
	def predict(self, x):
		zero_state = np.zeros([2, 1, self.i.hidden_size])
		feed_dict={ self.tf_x : x, self.init_state : zero_state, self.is_training : False }
		[preds] = self.sess.run([self.logits], feed_dict=feed_dict)
		return preds
#****************************************************
