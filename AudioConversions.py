import subprocess


def convert_wav_to_mp4(input_wav, output_mp4):
	command = [
		'ffmpeg',
		'-i', input_wav,  # Input file
		'-c:a', 'aac',  # Audio codec
		'-b:a', '192k',  # Audio bitrate
		output_mp4  # Output file
	]

	# Run the command
	subprocess.run(command, check=True)


# Example usage
input_wav = 'Audios/CHAT.wav'
output_mp4 = 'Audios/CHAT.mp4'

convert_wav_to_mp4(input_wav, output_mp4)
print(f"Converted {input_wav} to {output_mp4}")