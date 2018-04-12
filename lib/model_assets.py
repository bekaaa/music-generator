#!/use/bin/env python3
from __init__ import tf, np, instrument, note, stream, chord

class Input(object):
	def __init__(self, raw_data, c): # c is a Config object
		# expecting raw_data to be in shape [-1, num_prev, element_size ]
		assert raw_data.shape == (raw_data.shape[0], c.num_prev, c.element_size)
		# storing given parameters
		self.num_prev = c.num_prev
		self.batch_size = c.batch_size
		self.num_steps = c.num_steps
		self.element_size = c.element_size
		self.data = raw_data
		# some other variable
		self.index = 0
		self.data_len = self.data.shape[0]
		self.num_batches = self.data_len // self.batch_size
		self.num_loops_in_epoch = self.num_batches // self.num_steps
		self.hidden_size = self.element_size * self.num_prev
		self.num_all_elem = self.batch_size * self.num_steps
		self.tensor_input_shape = [None, self.num_steps, self.hidden_size]
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

		# reset index if reached the end
		if self.index >= (self.data_len - self.batch_size*self.num_steps) :
			self.index = 0
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
			raise ValueError('You need to run next_batch() for the first time.')
		#----------------------------------
#***********************************************************************
class Config(object):
	def __init__(self):
		# initialize variables
		self.epochs = 1
		self.num_layers = 1
		self.print_step = 50
		self.droprate = .3
		self.init_scale = 0.05
		# model saving
		self.save_model = False
		self.save_after_epoch = 0
		self.model_save_path = None
		# data related variabes
		self.num_prev = 50
		self.batch_size = 5
		self.num_steps = 5
		self.element_size = 19*3
#*********************************************************************************
def sequence_loss(logits, targets):
	if len(logits.get_shape()) != 2:
		raise ValueError("Logits must be a "
			"[num_of_elements_in_batch x element_size] tensor")
	if len(targets.get_shape()) != 2:
		raise ValueError("Targets must be a "
			"[num_of_elements_in_batch x element_size] tensor")

	with tf.variable_scope("sequence_loss", reuse=tf.AUTO_REUSE):
		#---------------------------
		# computing cross entropy
		cost = tf.zeros(logits.get_shape().as_list()[0], name='cost')
		for i in [0, 19, 38] :
			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets[:,i:i+19], logits=logits[:, i:i+19])
			# loss is in the same shape of labels and logits
			cost += tf.reduce_mean(loss, axis=1)
			# axis 1 will remove second axis.
		cost /= 3.
		assert cost.get_shape().as_list()[0] == targets.get_shape().as_list()[0]
		# reduce crossent to a single number
		cost = tf.reduce_mean(cost, name='cost')
	return cost
#******************************************************************
def predictions_to_notes(preds):
	# preds are in shape [ n_notes x element_size]
	# convert to shape [n_notes] where each element is a string "name1octav1.name2octav2..." if available
	assert preds.shape[1] == 3*19
	note_names = ['A', 'B', 'B-', 'C', 'C#', 'D', 'E', 'E-', 'F', 'F#', 'G', 'G#']
	octaves = ['1', '2', '3', '4', '5', '6']
	notes = []
	for element in preds:
		el = ''
		for sub in [ element[0:19], element[19:38], element[38:]]:
			if sub.argmax() == 0 :
				# note is empty
				continue
			# else
			name = note_names[ sub[1:13].argmax() ]
			octav = octaves[ sub[13:].argmax() ]
			el = el + name + str(octav) + '.'
		notes.append(el)
	return notes
#*************************************************************************
def create_midi(notes, file_name):
	# output notes are 1D list of each element as a string of "name1octave1.name2o..."
	offset = 0
	output_notes = []

	for pattern in notes:
		# pattern is a chord
		notes_in_pattern = pattern.split('.')[:-1]
		if len(notes_in_pattern) > 1 : # is a chord
			notes = []
			for current_note in notes_in_pattern:
				new_note = note.Note(current_note)
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)
			new_chord = chord.Chord(notes)
			new_chord.offset = offset
			output_notes.append(new_chord)
		# pattern is a single note
		else :
			new_note = note.Note(notes_in_pattern[0])
			new_note.storedInstrument = instrument.Piano()
			new_note.offset = offset
			output_notes.append(new_note)
		#increase offset each step
		offset += .5

	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp='../data/generated_midi/'+file_name)
	print('midi is generated and saved to disk')
#******************************************************
#def predict(input_obj, graph, best_model, model_path=None):
