import json
import os
import random

import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
import soundfile as sf
import sounddevice as sd
from pydub.silence import split_on_silence
from pydub import AudioSegment


def remove_long_silences(audio_file, silence_thresh=-50, min_silence_len=1000, target_dBFS=-20.0):
	# Load the audio file
	audio=sf.read(audio_file)[0]
	print(len(audio),audio[100:110])
	audio = AudioSegment.from_wav(audio_file)
	# audio_segment = AudioSegment(
	# 	channel1.tobytes(),
	# 	frame_rate=SAMPLE_RATE,
	# 	sample_width=channel1.dtype.itemsize,
	# 	channels=1
	# )

	# Split audio where silence is longer than `min_silence_len` milliseconds
	# and silence is quieter than `silence_thresh` dBFS
	chunks = split_on_silence(
		audio,
		min_silence_len=min_silence_len,
		silence_thresh=silence_thresh
	)

	# Normalize each chunk and concatenate them together
	processed_audio = AudioSegment.empty()
	for chunk in chunks:
		# Normalize the chunk
		normalized_chunk = chunk.apply_gain(target_dBFS - chunk.dBFS)
		processed_audio += normalized_chunk

	return processed_audio


def makeAiAudio(text_prompt,filename,speaker="v2/en_speaker_2",Outfolder="Audios/"):
	# speaker="speaker_0" #"v2/en_speaker_2"
	print(speaker)
	os.makedirs(Outfolder,exist_ok=True)
	preload_models()
	audio_array = generate_audio(text_prompt,history_prompt=speaker)
	print(audio_array)
	sf.write(Outfolder+filename+"_"+speaker.split("/")[-1]+".wav" , audio_array,SAMPLE_RATE)
	print("Saved at",Outfolder+filename+"_"+speaker.split("/")[-1] )
	sd.play(audio_array,samplerate=SAMPLE_RATE, blocking=True)


def makeAudioFile(chat,filename="CHAT",Outfolder="Audios/",speaker2="v2/en_speaker_1",speaker1="v2/en_speaker_1"):
	audios=[]
	for i,ch in enumerate(chat):
		if ch["role"]=="assistant":
			speaker=speaker1
		elif ch["role"]=="user":
			speaker=speaker2
		else:
			continue
		if "speaker" in speaker:
			audio_array = np.concatenate((np.zeros(random.randint(4,7)*SAMPLE_RATE),generate_audio(ch["content"], history_prompt=speaker)))
		else:
			audio_array=sf.read(f"Audios/Users/User{i}.wav")[0]
		print(ch["role"],ch["content"])
		sd.play(audio_array, samplerate=SAMPLE_RATE, blocking=True)
		audios.append(audio_array)
		audio_array=np.concatenate(audios)
		sf.write(Outfolder + filename + ".wav", audio_array, SAMPLE_RATE)
		print("Saved at", Outfolder + filename )

def getChat():
	with open("Files/Chat2.json") as f:
		data=json.load(f)
	# print(data)
	# for d in data:
	# 	if d["role"]:
	# 		print(d)
	return data

def BuildAudioFromChat():
	chat=getChat()
	makeAudioFile(chat,speaker2="User")

# text_prompt = """
# 	 Welcome to ABC Hospital! I'm happy to assist you. How can I help you today?.
# """

# audioprocessed=remove_long_silences("Audios/Test1.wav")
#
# audioprocessed.export("Audios/Test2.wav")

# text_prompt="my name is james"
# for i in range(10):
# 	makeAiAudio(text_prompt,"Test",speaker="v2/de_speaker_"+str(i))
#
BuildAudioFromChat()

