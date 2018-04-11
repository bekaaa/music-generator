#!/usr/bin/env python3
#from __init__ import *
from __init__ import load_piece, rearrange_data, Input, Config, MusicLSTM

def main():
	config = Config()
	config.num_prev = 50
	config.batch_size = 14
	config.num_steps = 10
	config.element_size = 19*3
	config.epochs = 100
	config.num_layers = 1
	config.print_step = 20
	config.save_model = True
	config.save_after_epoch = 1
	config.model_save_path = '../checkpoints/2/model-test'

	print('loading data ...', end='  ')
	raw_data = load_piece('../data/Mozart_pickles/', 1)
	raw_data = rearrange_data(raw_data, config.num_prev)
	print('data loaded in shape',raw_data.shape)

	data_obj = Input(raw_data, config)
	lstm_model = MusicLSTM(data_obj, config)
	lstm_model.train()
#**************************************


if __name__ == '__main__':
	main()
