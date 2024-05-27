import os,json

import pyaudio
import wave
import threading
from pydub import AudioSegment
import torch
from df import enhance, init_df


# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 2048
OUTPUT_FILE = "output.wav"
import soundfile as sf

from df.enhance import enhance, init_df, load_audio, save_audio
model, df_state, _ = init_df()  # Load default model


def change_sample_rate(input_file, output_file, new_sample_rate):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    # Change the sample rate
    audio = audio.set_frame_rate(new_sample_rate)
    # Export the modified audio file
    audio.export(output_file, format="wav")

def change_sample_rate_for_all(folder,SAMPLERATE=24000):
	for file in os.listdir(folder):
		change_sample_rate(folder+file,folder+file,SAMPLERATE)

def list_audio_devices():
	p = pyaudio.PyAudio()
	info = p.get_host_api_info_by_index(0)
	numdevices = info.get('deviceCount')

	for i in range(0, numdevices):
		device_info = p.get_device_info_by_host_api_device_index(0, i)
		if device_info.get('maxInputChannels') > 0:
			print(f"Input Device id {i} - {device_info.get('name')}")

	p.terminate()


def list_supported_sample_rates(device_index):
	p = pyaudio.PyAudio()
	sample_rates = [8000, 16000, 32000, 44100, 48000, 96000, 192000]
	supported_sample_rates = []

	for rate in sample_rates:
		try:
			p.is_format_supported(rate,
								  input_device=device_index,
								  input_channels=1,
								  input_format=pyaudio.paInt16)
			supported_sample_rates.append(rate)
		except ValueError:
			pass

	p.terminate()
	return supported_sample_rates


# Function to record audio
def record_audio(filename,audio,stop_event,Outfolder="Audios/Users/"):
	os.makedirs(Outfolder,exist_ok=True)
	OUTPUT_FILE=Outfolder+filename+".wav"
	stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,input_device_index=8,
						frames_per_buffer=CHUNK)
	print("Recording... Press Enter to stop.")

	frames = []

	try:
		while not stop_event.is_set():
			data = stream.read(CHUNK)
			frames.append(data)
	except Exception as e:
		print(f"An error occurred: {e}")

	print("Recording stopped.")

	# Stop and close the stream
	stream.stop_stream()
	stream.close()

	# Terminate the PortAudio interface
	audio.terminate()

	# Save the recorded data as a WAV file
	wf = wave.open(OUTPUT_FILE, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(audio.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()


def recordAudio(string,filename,Outfolder="Audios/Users/"):
	# Initialize PyAudio
	audio = pyaudio.PyAudio()
	print(string)
	# Event to control the stopping of recording
	stop_event = threading.Event()
	# Function to wait for user input to stop recording
	def wait_for_stop():
		input()  # Wait for user to press Enter
		stop_event.set()

	print("Start")
	# Start the recording thread
	recording_thread = threading.Thread(target=record_audio,args=(filename,audio,stop_event,Outfolder))
	recording_thread.start()

	# Wait for user input in the main thread
	wait_for_stop()

	# Wait for the recording thread to finish
	recording_thread.join()
	OUTPUT_FILE = Outfolder + filename + ".wav"
	print("Audio recording saved to", OUTPUT_FILE)
	audio, _ = load_audio(OUTPUT_FILE, sr=df_state.sr())
	enhanced_audio = enhance(model, df_state, torch.Tensor(audio))
	save_audio(OUTPUT_FILE, enhanced_audio, df_state.sr())
	print(OUTPUT_FILE,"Enhanced")


def getChat():
	with open("Files/Chat2.json") as f:
		data=json.load(f)
	return data


def RecordChat():
	messages =getChat()
	for i,messag in enumerate(messages):
		if i!=6:continue
		if messag["role"] == "user":
			recordAudio(messag["content"],"User"+str(i))


def EnhanceAll():
	for file in os.listdir("Audios/Users"):
		OUTPUT_FILE="Audios/Users/"+file
		model, df_state, _ = init_df()  # Load default model
		audio, _ = load_audio(OUTPUT_FILE, sr=df_state.sr())
		enhanced_audio = enhance(model, df_state, torch.Tensor(audio))
		save_audio(OUTPUT_FILE, enhanced_audio, df_state.sr())



# list_audio_devices()
# RecordChat()
# EnhanceAll()
change_sample_rate_for_all("Audios/Users/")