import glob
import subprocess

midi_files = glob.glob("./babyslakh_16k/*/*.mid")

def midi_to_wav(input_path, output_path):
    cmd = ['timidity', input_path, '-Ow', '-o', output_path]
    subprocess.call(cmd)

for i in range(len(midi_files)):
    midi_file = midi_files[i]
    wav_file = f'./wav_data/{i}.wav'
    print(f"Converting {midi_file} to {wav_file}")
    midi_to_wav(midi_file, wav_file)