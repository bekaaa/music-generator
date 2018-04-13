#!/usr/bin/env python3
from __init__ import *
#from __init__ import load_piece, rearrange_data, Input, Config, MusicLSTM

def main():
	config = Config()
	config.num_prev = 20
	config.batch_size = 14
	config.num_steps = 10
	config.element_size = 19*3
	config.epochs = 100
	config.num_layers = 1
	config.print_step = 20
	config.save_model = True
	config.save_after_epoch = 1
	config.model_save_path = '../checkpoints/2/model-test6'

	print('loading data ...', end='  ')
	raw_data = load_piece('../data/Mozart_pickles/', 1)
	raw_data = rearrange_data(raw_data, config.num_prev)
	print('data loaded in shape',raw_data.shape)

	data_obj = Input(raw_data, config)
	lstm_model = MusicLSTM(data_obj, config)
	lstm_model.train(restore_from='../checkpoints/2/model-test5')
#**************************************
def train_on_all():
	config = Config()
	config.num_prev = 20
	config.batch_size = 14
	config.num_steps = 10
	config.element_size = 19*3
	config.epochs = 1
	config.num_layers = 1
	config.print_step = 20
	config.save_model = True
	config.save_after_epoch = 0
	#config.model_save_path = '../checkpoints/2/model-test6'

	data_source = '../data/Mozart_pickles/'
	num_pieces = len(glob.glob(data_source+'*.pkl'))
	overall_epochs = 2
	for overall_epoch in range(overall_epochs):
		print('starting overall epoch', overall_epoch)
		for pieceId in range(num_pieces):
			print('loading piece number %d of %d ...' % (pieceId, num_pieces), end='  ')
			raw_data = load_piece(data_source, pieceId)
			raw_data = rearrange_data(raw_data, config.num_prev)
			print('data loaded in shape', raw_data.shape)

			config.model_save_path = '../checkpoints/mozart/piece-%d' % pieceId
			data_obj = Input(raw_data, config)
			model = MusicLSTM(data_obj, config)
			if pieceId == 0 and overall_epoch == 0:
				model.train()
			elif pieceId == 0 :
				model.train(restore_from='../checkpoints/mozart/piece-%d' % (num_pieces-1))
			else :
				model.train(restore_from='../checkpoints/mozart/piece-%d' % (pieceId-1) )

if __name__ == '__main__':
	#main()
	train_on_all()
