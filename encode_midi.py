from music21 import converter, instrument, note, chord, stream, tempo
import glob
import numpy as np
from collections import OrderedDict
import pickle
from keras.utils import to_categorical
from tqdm import tqdm
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def get_list_of_all_possible_notes(glob_query):
    notes = OrderedDict()
    time_signatures = set()  # set to capture unique time signatures
    artists = set()  # set to capture unique artists

    for file in tqdm(glob.glob(glob_query), desc="Parsing MIDI files"): 
        midi = converter.parse(file)
        #print("Parsing %s" % file)

        artist_name = file.split('/')[-2] # second to last item in the path
        artists.add(artist_name)
        
        # Extracting Time Signatures
        time_sig = midi.getTimeSignatures()[0] if midi.getTimeSignatures() else "4/4"
        time_signatures.add(str(time_sig))

        parts = instrument.partitionByInstrument(midi) if midi.hasPartLikeStreams() else [midi]
        for part in parts:
            for element in part.recurse():
                if isinstance(element, note.Note):
                    note_str = str(element.duration) + "-" + str(element.pitch) + "-" + str(element.volume.velocity)
                    notes[note_str] = notes.get(note_str, len(notes))
                elif isinstance(element, chord.Chord):
                    chord_str = str(element.duration) + "-" + '.'.join(str(n) + "-" + str(n.volume.velocity) for n in element.notes)
                    notes[chord_str] = notes.get(chord_str, len(notes))
                elif isinstance(element, note.Rest):
                    rest_str = str(element.duration) + "-Rest"
                    notes[rest_str] = notes.get(rest_str, len(notes))

    note_list = list(notes.keys())
    pickle.dump(note_list, open('notes.p', 'wb'))
    pickle.dump(list(time_signatures), open('time_signatures.p', 'wb'))
    pickle.dump(list(artists), open('artists.p', 'wb'))
    return note_list

def get_artist_index(artist,artists):
    return artists.index(artist)

def get_sequences(glob_query):
    notes = []
    artist_sequences = []
    artists = pickle.load(open("artists.p", "rb"))
    for file in tqdm(glob.glob(glob_query), desc="Getting sequences"):
        midi = converter.parse(file)

        artist_name = file.split('/')[-2] # second to last item in the path
        artist_idx = get_artist_index(artist_name,artists)
        
        parts = instrument.partitionByInstrument(midi) if midi.hasPartLikeStreams() else [midi]
        for part in parts:
            for element in part.recurse():
                if isinstance(element, note.Note):
                    notes.append(str(element.duration) + "-" + str(element.pitch) + "-" + str(element.volume.velocity))
                elif isinstance(element, chord.Chord):
                    notes.append(str(element.duration) + "-" + '.'.join(str(n) + "-" + str(n.volume.velocity) for n in element.notes))
                elif isinstance(element, note.Rest):
                    notes.append(str(element.duration) + "-Rest")

                artist_sequences.append(artist_idx)

    return notes, artist_sequences

def prepare_sequences(sequence, artist_sequences): 
    """ Prepare the sequences used by the Neural Network """
    input_sequence_length = 8
    network_artist_input = []

    # Form "note_to_int" lookup dictionary using notes.p, the file where we have stored a list of all the notes that appear in our vocabulary.
    pitchnames = pickle.load(open("notes.p", "rb"))

    # Dictionary called note_to_int is created where the keys are the elements of the pitchnames list and the values are their indices obtained using the enumerate() function.
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    n_vocab = len(pitchnames)

    network_input = []
    network_output = []

    for i in tqdm(range(0, len(sequence) - input_sequence_length, 1), desc="Processing sequences"):
        sequence_in = sequence[i:i + input_sequence_length]
        sequence_out = sequence[i + input_sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        network_artist_input.append(artist_sequences[i])

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, input_sequence_length, 1))
    network_input = to_categorical(network_input, num_classes = n_vocab)

    network_output = to_categorical(network_output, num_classes = n_vocab)

    return (network_input, network_output, np.array(network_artist_input))


vocabulary = get_list_of_all_possible_notes("babyslakh_16k/*/*.mid")
sequence, artist_sequence = get_sequences("babyslakh_16k/*/*.mid")
network_input, network_output, network_artist_input = prepare_sequences(sequence, artist_sequence)

# Print shapes of each variable
print("network_input.shape:", network_input.shape)
print("network_output.shape:", network_output.shape)
print("network_artist_input.shape:", network_artist_input.shape)
print("vocabulary length:", len(vocabulary))

# Save the variables to pickle files
pickle.dump(network_input, open("network_input.p", "wb"))
pickle.dump(network_output, open("network_output.p", "wb"))
pickle.dump(network_artist_input, open("network_artist_input.p", "wb"))
pickle.dump(vocabulary, open("vocabulary.p", "wb"))