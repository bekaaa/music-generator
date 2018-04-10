from lib.preprocessing import load_piece, rearrange_data
from lib.model_assets import Input, Config
from lib.lstm_model import lstm_model

def main():
	config = Config()
	config.num_prev = 50
	config.batch_size = 4
	config.num_steps = 10
	config.element_size = 19*3
	config.epochs = 10
	config.num_layers = 3
	config.save_model = True
	config.save_after_epoch = 5
	config.model_save_path = '../checkpoints/0/model'

	print('loading data')
	raw_data = load_piece('../data/Mozart_pickles/', 1)
	raw_data = rearrange_data(raw_data, config.num_prev)
	print('data loaded in shape',raw_data.shape)

	data_obj = Input(raw_data, config)
	lstm_model(data_obj, config)
#**************************************
if __name__ == '__main__':
	main()
