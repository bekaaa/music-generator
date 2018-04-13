#!/usr/bin/env python3
from __init__ import *
#from __init__ import load_piece, rearrange_data, Input, Config, MusicLSTM

def train_one_piece(pieceId):
	config = set_config()

	print('loading data ...', end='  ')
	raw_data = load_piece('../data/Mozart_pickles/', pieceId)
	raw_data = rearrange_data(raw_data, config.num_prev)
	print('data loaded in shape',raw_data.shape)

	data_obj = Input(raw_data, config)
	lstm_model = MusicLSTM(data_obj, config)
	lstm_model.train(restore_from='../checkpoints/2/model-test5')
#**************************************
def train_on_all():
	config = set_config()

	data_source = '../data/Mozart_pickles/'
	num_pieces = len(glob.glob(data_source+'*.pkl')) - 1
	overall_epochs = 2
	for overall_epoch in range(overall_epochs):
		print('starting overall epoch', overall_epoch)
		for pieceId in range(num_pieces):
			print('loading piece number %d of %d ...' % (pieceId, num_pieces), end='  ')
			raw_data = load_piece(data_source, pieceId)
			raw_data = rearrange_data(raw_data, config.num_prev)
			print('data loaded in shape', raw_data.shape)

			config.model_save_path = '../checkpoints/mozart/model0'
			data_obj = Input(raw_data, config)
			model = MusicLSTM(data_obj, config)
			if pieceId == 0 and overall_epoch == 0:
				model.train()
			elif pieceId == 0 :
				model.train(restore_from='../checkpoints/mozart/model0')
			else :
				model.train(restore_from='../checkpoints/mozart/model0' )
#**************************************
def compose_new_piece(data_, config, model_path, length):
	'''
	data_ is an Input object
	initial_sequence should be in shape (num_prev * element_size)
	'''
	idx = np.random.randint(0, data_.num_loops_in_epoch)
	for i in range(idx+1):
		x, _ = data_.next_batch()
	# x is in shape (1, steps, PxN)

	composed = np.zeros([length, config.element_size])
	model = MusicLSTM(data_, config)
	model.init_predictor(model_path)

	for l in range(length) :
		x[0,1:] = np.zeros([config.num_steps-1, data_.hidden_size])
		preds = model.predict(x)
		# preds are in shape [num_steps, N]
		composed[l] = preds[0]
		x = x.reshape([1, config.num_steps, config.num_prev, config.element_size])
		x[0,0,:-1] = x[0,0,1:]
		x[0,0,-1] = preds[0,0]
		x = x.reshape([1, config.num_steps, data_.hidden_size])

	notes = predictions_to_notes(composed)
	create_midi(notes, 'generatedtest.mid')
#-----------------------------------------------------------------------------
def set_config():
	config = Config()
	config.num_prev = 20
	config.batch_size = 1
	config.num_steps = 30
	config.element_size = 19*3
	config.epochs = 100
	config.num_layers = 1
	config.print_step = 20
	config.save_model = True
	config.save_after_epoch = 0
	#config.model_save_path = '../checkpoints/2/model-test6'

	return config

if __name__ == '__main__':
	#main()
	train_on_all()
	# config = set_config()
	# print('loading data ...', end='  ')
	# raw_data = load_piece('../data/Mozart_pickles/', 18)
	# raw_data = rearrange_data(raw_data, config.num_prev)
	# print('data loaded in shape',raw_data.shape)
	#
	# data_obj = Input(raw_data, config)
	#
	# compose_new_piece(data_obj, config, '../checkpoints/mozart/piece-1', 50)
