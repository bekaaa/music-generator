version 1 notes

This verison uses LSTM RNN.

data preparations and shape : 
The input data is a midi piano notes for Mozart.  
Using music21 I extracted the notes from the files,  
available notes are : 'A', 'B', 'B-', 'C', 'C#', 'D', 'E', 'E-', 'F', 'F#', 'G', 'G#'  
and each note could be in one of 6 octaves. Also each chord may have two or three notes.  
My data is in shape  (number of elements x 3 x note_size) where :
  - note size is a boolean sparse array of [ empty, A, B, B-, .... G, G#, 1, 2, 3, 4, 5, 6] where :
    - the first bit is set to 1 if this note is empty [ will explain later]
    - the following 12 will have only one bit set to 1 [note]
    - the last 6 will have only one bet set to 1 [octave]
  - [ 3 x note_size ] so every element would possible have 1-note, 2-notes chord or 3-notes chord, and if it's a 1-note
    the other two [ 2 x note_size ] would have an "empty" bit set to 1.
 ---------------------------  
 
For the model, is used an N-layers LSTM-RNN followed by a regular softmax(X * W + b), and finally a customized cross-entropy
  loss function defined in lib/model_assets.py
  
---------------------------------   

- Model files are in lib which are lib/lstm_model.py, lib/model_assets.py, and preprocessing.py.  
- __init__.py imports the necessary libraries.  
- main.py show you how run a model.  
- The model needs two objects, a config object and a input object. Both are explained inside lib/,model_assets.py  

---------------------------  

Example of output files are in data/generated_midi/  

----------------------------   
