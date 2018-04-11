#!/use/bin/env python3

import tensorflow as tf
import numpy as np
import glob
import _pickle as pickle
import sys
sys.path.append('..')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from music21 import instrument, note, stream, chord, converter
# local modules
from preprocessing import load_piece, rearrange_data
from model_assets import Input, Config, sequence_loss
from lstm_model import MusicLSTM
