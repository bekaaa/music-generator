from lib.preprocessing import load_piece, rearrange_data
from lib.model_assets import Input, Config
from lib.lstm_model import lstm_model

def main():
	config = Config()
	config.num_prev = 10
	config.batch_size = 5
	config.num_steps = 10
	config.element_size = 19*3
	config.epochs = 5
	config.num_layers = 1
	config.save_model = True
	config.save_after_epoch = 2
	config.model_save_path = '../checkpoints/0/model'

	print('loading data')
	raw_data = load_piece('../data/Mozart/', 2)
	raw_data = rearrange_data(raw_data, config.num_prev)
	print('data loaded in shape',raw_data.shape)

	data_obj = Input(raw_data, config)
	lstm_model(data_obj, config)
#**************************************
if __name__ == '__main__':
	main()
