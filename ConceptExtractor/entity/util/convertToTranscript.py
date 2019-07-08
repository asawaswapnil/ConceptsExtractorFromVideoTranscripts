
def convert_to_transcript( filename,type_of_file):
	if type_of_file==".vtt":
		import webvtt
		vtt = webvtt.read(filename)
		lines = []
		for line in vtt:
			lines.extend(line.text.strip().splitlines())
		transcript = ""
		previous = None
		for line in lines:
			if line == previous:
			   continue
			transcript += " " + line
			previous = line
		#print(transcript)
		return transcript
	elif type_of_file==".txt":
		with open(filename, 'r') as file:
			data = file.read().replace('\n', '')
		#print(data)
		return data 
convert_to_transcript("/home/swapnil/ir/educational_videos/educational_videos/iir-1/4._Analysis_of_Structured_Data/4._Analysis_of_Structured_Data.en.vtt", ".vtt")