#!/usr/bin/env python3
import numpy as np
from collections import OrderedDict
from music21 import instrument, note, chord, converter
import _pickle as pickle
import glob
#---------------------------------------
def preprocess_raw_data(input_folder, output_folder):

	#input_folder = './music/Mozart/'
	#output_folder = './data/Mozart/'
	bigchords = []
	all_data = []
	note_names = ['A', 'B', 'B-', 'C', 'C#', 'D', 'E', 'E-', 'F', 'F#', 'G', 'G#']
	octaves = ['1', '2', '3', '4', '5', '6']
	element_info = OrderedDict({e:0 for e in ['empty'] + note_names + octaves})

	for index,file in enumerate(glob.glob(input_folder+'*.mid')):
		if index % 10 == 0 : print(index,end='')
		print('.',end='')
		midi = converter.parse(file)
		parts = instrument.partitionByInstrument(midi)

		if parts :
			piano_notes = parts.parts[0].recurse()
		else :
			print('\n*caution no parts are found.')
			piano_notes = parts.flat_notes

		# get number of notes and chords inside it
		notes_and_chords = [element for element in piano_notes \
			if isinstance(element, note.Note) or isinstance(element, chord.Chord)]
		# array to hold the data
		data = np.zeros([len(notes_and_chords), 3, 19], dtype=np.int32)
		# prepare notes
		for elidx, element in enumerate(notes_and_chords):
			elinfo = [ element_info.copy() for i in range(3) ]

			if isinstance(element, note.Note):
				assert str(element.name) in note_names and str(element.octave) in octaves
				elinfo[0][str(element.name)] = 1
				elinfo[0][str(element.octave)] = 1
				elinfo[1]['empty'] = 1
				elinfo[2]['empty'] = 1
				data[elidx, 0, :] = list(elinfo[0].values())
				data[elidx, 1, :] = list(elinfo[1].values())
				data[elidx, 2, :] = list(elinfo[2].values())

			elif isinstance(element, chord.Chord):
				if len(element.pitches) > 3 :
					bigchords.append(element.pitches)
					element.pitches = element.pitches[:3]
				#assert len(element.pitches) <= 3
				for idx,e in enumerate(element.pitches):
					assert str(e.name) in note_names and str(e.octave) in octaves
					elinfo[idx][str(e.name)] = 1
					elinfo[idx][str(e.octave)] = 1
				# incase there is only 2 notes in the chord
				for i in range(idx,3):
					elinfo[i]['empty'] = 1
				data[elidx, 0, :] = list(elinfo[0].values())
				data[elidx, 1, :] = list(elinfo[1].values())
				data[elidx, 2, :] = list(elinfo[2].values())
			#----------
			#del elinfo
		# save data to disk
		fn = output_folder + file.split('/')[-1].split('.')[0] + '.pkl'
		with open(fn, 'wb') as f:
			pickle.dump(data, f)
		all_data.append(data)
		#---------------------
	return bigchords, all_data
#----------------------------------------------------------------------
def load_piece(music_folder, pieceId):
	music_files = glob.glob(music_folder+'*.pkl')
	assert len(music_files) > 0
	assert pieceId < len(music_files)
	with open(music_files[pieceId], 'rb') as f:
		data = pickle.load(f)
	return data
#-------------------------------------------------------------------------
def rearrange_data(data, num_comb=100):
	'''
	input is in shape  [notes_chords, 3, 19]
	output is in shape [new_len, num_comb, 57]
	'''
	new_len = data.shape[0] - num_comb
	new_data = np.zeros([ new_len, 100, 3, 19])
	for i in range(new_len):
		new_data[i] = data[i:i+num_comb]
	new_data = new_data.reshape(-1, 100, 3*19)
	return new_data
