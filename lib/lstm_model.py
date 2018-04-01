import tensorflow as tf
import numpy as np
from loss import sequence_loss

#*****************************************************************
def batch_producer(raw_data, batch_size, num_steps):
	raw_data = tf.convert_to_tensor(raw_data, dtype=tf.int32, name='raw_data')
	data_len = len(raw_data)
	batches = data_len // batch_size - 1

	data = tf.reshape(raw_data[0:batch_size*batches], [batches, batch_size, 100, 57])
	epoch_pieces = batches // num_steps

	i = tf.train.range_input_producer(epoch_pieces, num_epochs=1, shuffle=False, seed=0).dequeue()
	x = data[ i * num_steps : (i+1) * num_steps, :, :, : ]
	x.set_shape((num_steps, batch_size, 5700))
	y = data[ i * num_steps + 1 : (i+1) * num_steps + 1, :, 0, : ]
	y.set_shape((num_steps, batch_size, 57))

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
		#self.is_training = is_training
		self.hidden_size = hidden_size
		#self.num_layers = num_layers
		#self.droprate = droprate
		self.init_scale = init_scale
		self.num_steps = self.input_obj.num_steps
		self.batch_size = self.input_obj.batch_size

		inputs = self.input_obj.input_data
		if is_training and droprate > 0 :
			inputs = tf.nn.dropout(inputs, keep_prob=1-droprate, name='input_dropout')

		# set state storage
		self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size],
										 'init_state')
		# prepare it
		state_per_layer_list = tf.unstack(self.init_state, axis=0)
		rnn_tuple_state = tuple( tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],
															  state_per_layer_list[idx][1])
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
