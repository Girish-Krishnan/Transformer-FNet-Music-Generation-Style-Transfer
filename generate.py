from keras.models import load_model
from music21 import stream, note, chord, tempo, metadata, instrument
import music21
import pickle
import numpy as np
import warnings
from model import *
import subprocess

def midi_to_wav(input_path, output_path):
    cmd = ['timidity', input_path, '-Ow', '-o', output_path]
    subprocess.call(cmd)

warnings.filterwarnings("ignore")

# Load your trained model
model = load_model('model.h5', custom_objects={'TransformerBlock': TransformerBlock, 'MultiHeadAttention': MultiHeadAttention, 'FourierTransformLayer': FourierTransformLayer, 'PointwiseFeedForwardNetwork': PointwiseFeedForwardNetwork, 'FNetBlock': FNetBlock})

# Load the vocabulary and artist lookup tables
notes = pickle.load(open('notes.p', 'rb'))
artists = pickle.load(open('artists.p', 'rb'))

# Define the mappings
note_to_int = dict((note, number) for number, note in enumerate(notes))
int_to_note = dict((number, note) for number, note in enumerate(notes))
artist_to_int = dict((artist, idx) for idx, artist in enumerate(artists))

def generate_sequence(seed, artist_name, sequence_length=1500):
    generated = []
    artist_int = artist_to_int[artist_name]
    for i in range(sequence_length):
        # One-hot encode the seed
        one_hot_seed = np.zeros((len(seed), len(notes)))
        for idx, val in enumerate(seed):
            one_hot_seed[idx, val] = 1

        # Reshape to match the model's expected input shape
        input_sequence = np.reshape(one_hot_seed, (1, len(seed), len(notes)))

        input_artist = np.array([[artist_int]])

        prediction = model.predict([input_sequence, input_artist])
        prediction = prediction / np.sum(prediction)
        index = np.random.choice(len(notes), p=prediction[0])
        #index = np.argmax(prediction)
        result = int_to_note[index]
        generated.append(result)
        seed.append(index)
        seed = seed[1:len(seed)]
    return generated

def sequence_to_midi(sequence, artist_name):
    s = stream.Score()
    s.insert(0, metadata.Metadata())
    s.metadata.title = f"Generated Song in the style of {artist_name}"
    s.metadata.composer = artist_name
    
    part = stream.Part()
    s.insert(0, part)
    
    for item in sequence:
        elements = item.split("-")
        
        # Duration
        duration_val = elements[0].replace('<music21.duration.Duration ', '').replace('>', '')
        try:
            if '/' in duration_val:
                num, denom = duration_val.split('/')
                duration_val = float(num) / float(denom)
            else:
                duration_val = float(duration_val)
        except:
            continue
        
        # Note/Chord/Rest
        if 'Rest' in elements[1]:
            new_note = note.Rest(quarterLength=duration_val / 16)
            part.append(new_note)
        elif '<music21.note.Note' in elements[1]:
            # Multiple notes: Chord
            chord_notes = []
            for chord_part in elements[1:]:
                if '<music21.note.Note' in chord_part:
                    pitch_str = chord_part.replace('<music21.note.Note ', '').replace('>', '').strip()
                    chord_notes.append(pitch_str)
            #new_note = chord.Chord(chord_notes, quarterLength=duration_val)
            try:
                new_note = chord.Chord(chord_notes, quarterLength=duration_val)
            except music21.pitch.PitchException as e:
                print(f"Error encountered: {e}. Skipping chord creation.")
                continue
            
            part.append(new_note)

        else:
            # Single note
            pitch = elements[1]
            velocity = int(elements[2])
            new_note = note.Note(pitch, quarterLength=duration_val)
            new_note.volume.velocity = velocity
        
            part.append(new_note)
    
    # Save as MIDI
    mf = s.write('midi', fp=f"{artist_name}_generated.mid")
    midi_to_wav(f"{artist_name}_generated.mid", f"{artist_name}_generated.wav")
    return mf

# Use a seed sequence and an artist name to generate a new sequence
# Take the first 8 notes from the note_to_int dictionary values
seed = list(note_to_int.values())[:8]
artist_name = "Track00001"  # Replace with the actual artist name
generated_sequence = generate_sequence(seed, artist_name)
sequence_to_midi(generated_sequence, artist_name)