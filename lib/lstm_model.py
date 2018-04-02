#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from lib.loss import sequence_loss
import datetime as dt

#*****************************************************************
def batch_producer(raw_data, batch_size, num_steps):
	raw_data = tf.convert_to_tensor(raw_data, dtype=tf.int32, name='raw_data')
	data_len = raw_data.get_shape().as_list()[0]
	batches = data_len // batch_size

	data = tf.reshape(raw_data[0:batch_size*batches], [batch_size, batches, 100, 57])
	epoch_pieces = (batches - 1) // num_steps

	i = tf.train.range_input_producer(epoch_pieces, num_epochs=1, shuffle=False, seed=0).dequeue()
	x = data[ :, i * num_steps : (i+1) * num_steps, :, : ]
	x.set_shape((batch_size, num_steps, 5700))
	y = data[ :, i * num_steps + 1 : (i+1) * num_steps + 1, 0, : ]
	y.set_shape((batch_size, num_steps, 57))

	return x,y
#*******************************************************************************
class Input(object):
	def __init__(self, batch_size, num_steps, data):
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.epoch_pieces = (len(data) // batch_size - 1) // num_steps
		self.input_data, self.targets = batch_producer(data, batch_size, num_steps)
#*******************************************************************************
class Model(object):
	def __init__(self, input_obj, is_training, hidden_size, num_layers, droprate=.3, init_scale=0.05):
		self.input_obj = input_obj
		self.hidden_size = hidden_size
		self.init_scale = init_scale
		# extract info and data from input object
		self.num_steps = self.input_obj.num_steps
		self.batch_size = self.input_obj.batch_size
		inputs = self.input_obj.input_data
		# add dropout to input if supplied and in training process
		if is_training and droprate > 0 :
			inputs = tf.nn.dropout(inputs, keep_prob=1-droprate, name='input_dropout')
		# set state storage
		self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size],'init_state')
		# unstack it and convert  to LSTM state Tuple
		state_per_layer_list = tf.unstack(self.init_state, axis=0)
		rnn_tuple_state = tuple(
 			tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1])
			for idx in range(num_layers) )
		# create LSTM cell
		cell = tf.contrib.rnn.LSTMCell(hidden_size)
		# add more dropout to it
		if is_training and droprate > 0 :
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1-droprate)
		# adjust incase of more than 1 layer chosen
		if num_layers > 1 :
			cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])

		output, self.state = tf.nn.dynamic_rnn(cell, inputs, initial_state=rnn_tuple_state)
		# output is in shape [ batch_size, num_steps, hidden_size ]
		#----------------------------------------------------------------------------------
		# Now define the softmax, loss and optimizer.
		output = tf.reshape(output, (-1, hidden_size))
		softmax_weights = tf.Variable(tf.random_uniform([hidden_size, note_elements],
														-init_scale,init_scale))
		softmax_biases = tf.Variable(tf.random_uniform([note_elements], -init_scale, init_scale))
		logits = tf.nn.xw_plus_b(output, softmax_weights, softmax_biases)

		# update cost
		self.cost = sequence_loss(logits, self.input_obj.targets, tf.ones([self.batch_size, self.num_steps]),
					average_across_batch=True, average_across_timesteps=True, name='Loss')

		# get predictions
		self.predictions = np.zeros(tf.shape(logits))
		for i in [0, 19, 38] :
			self.predictions[:,:,i:i+19] = tf.nn.sigmoid(logits[:,:,i:i+19])

		# return in case of validating or testing
		if not is_training :
			return

		#self.learning_rate = tf.Variable(0.001, trainable=False)
		optimizer = tf.train.AdamOptimizer().minimize(self.cost)
		return
#***************************************************************************************
def train(train_data, num_layers, num_epochs, batch_size, model_save_name):
	#setup data and input
	best_loss = 999
	print_step = 50
	global_step = 0
	hidden_size = 5700
	training_input = Input(batch_size, 30, train_data)
	model = Model(training_input, is_training=True, hidden_size=hidden_size, num_layers=num_layers)
	init_op = tf.global_variable_initializer()
	with tf.Session() as sess :
		sess.run([init_op])
		coord = tf.train.Coordiantor()
		threads = tf.train.start_queue_runners(coord=coord)
		saver = tf.train.Saver()
		# start training.
		for epoch in range(num_epochs):
			curent_state = np.zeros([num_layers, 2, batch_size, hidden_size])
			curr_time = dt.datetime.now()
			for step in range(training_input.epoch_pieces):
				# run a training step.
				cost, _, current_state = sess.run([model.cost, model.optimizer, model.state],
					feed_dict={model.init_state : current_state})
				# print cost, sconds per step every print_step
				if step % print_step == 0 :
					seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
					curr_time = dt.datetime.now()
					print("epoch {}, step {}, cost: {:.3f}, seconds per step: {:.3f}".format(epoch,step,cost,seconds))
				# save model checkpoint if cost is improved
				if cost < best_loss :
					saver.save(sess, model_save_name, global_step=global_step)
					global_step += 1
					best_loss = cost
					print('model saved.')
		# close threads
		coord.request_stop()
		coord.join(threads)
#*****************************************************************
if __name__ == '__main__':
	from lib.preprocessing import load_piece
	piece = load_piece('./data/Mozart/', 2)
	print(piece.shape)
	train(piece, 1, 1, batch_size=20, model_save_name='./checkpoints/0/model')
