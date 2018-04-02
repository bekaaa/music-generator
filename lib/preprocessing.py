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

	note_names = ['A', 'B', 'B-', 'C', 'C#', 'D', 'E', 'E-', 'F', 'F#', 'G', 'G#']
	octaves = ['1', '2', '3', '4', '5', '6']
	element_info = [ OrderedDict({e:0 for e in ['empty'] + note_names + octaves}) for i in range(3) ]

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

		# array to hold the data
		data = np.zeros([len(piano_notes), 3, 19], dtype=np.int32)
		# prepare notes
		for elidx, element in enumerate(piano_notes):
			elinfo = element_info.copy()

			if isinstance(element, note.Note):
				assert str(element.name) in note_names and str(element.octave) in octaves
				elinfo[0][str(element.name)] = 1
				elinfo[0][str(element.octave)] = 1
				elinfo[1]['empty'] = 1
				elinfo[2]['empty'] = 1
				data[elidx, 0] = list(elinfo[0].values())
				data[elidx, 1] = list(elinfo[1].values())
				data[elidx, 2] = list(elinfo[2].values())

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
				data[elidx, 0] = list(elinfo[0].values())
				data[elidx, 1] = list(elinfo[1].values())
				data[elidx, 2] = list(elinfo[2].values())
			#----------
		# save data to disk
		fn = output_folder + file.split('/')[-1].split('.')[0] + '.pkl'
		with open(fn, 'wb') as f:
			pickle.dump(data, f)
		#---------------------
	return bigchords
#----------------------------------------------------------------------
def load_piece(music_folder, pieceId):
	music_files = glob.glob(music_folder+'*.pkl')
	assert len(music_files) > 0
	assert pieceId < len(music_files)
	with open(music_files[pieceId], 'rb') as f:
		data = pickle.load(f)
	return data
#-------------------------------------------------------------------------
