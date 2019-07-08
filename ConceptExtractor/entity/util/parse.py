 import csv

import os
import csv
import pandas as pd

from convertToTranscript import convert_to_transcript
def parse_data(datadir):
	filenames_list=[]
	transcript_list = []
	roots=[]
	# for maindir in os.listdir(datadir):
	# 	subdir=datadir+"/"+maindir
	

	for root, directories, filenames in os.walk(datadir):
		for filename in filenames:
			#print(subdir,filename)
			if filename.endswith(type_of_file):
				print(filename)
				filei = os.path.join(root, filename)
				#print(filei)
				roots.append(directories)
				filenames_list.append(filei)
				try:
					transcript_list.append(convert_to_transcript(filei, type_of_file))
				except:
					transcript_list.append("-")
	fin=[[x,z] for x,z in zip(filenames_list, transcript_list)]
	#print(fin[0])
	#csvname= "iir-1.csv"
	#print(csvname)
	
	pd.DataFrame(data=fin).to_csv(csvname,header=['bookname','text'],index_label='docid')

	# with open("output.csv", "wb") as f:
	# 	writer = csv.writer(f)
	# 	writer.writerows(fin)
	#fin.to_csv()
what_to_parse="sections"
if(what_to_parse=="sections"):
	load_from_folder_name="/home/swapnil/ir/sections"
	type_of_file=".txt"
	csvname="sections.csv"
elif(what_to_parse=="videos_iir-1"):
	load_from_folder_name="/home/swapnil/ir/educational_videos/educational_videos/iir-1"
	type_of_file=".vtt"
	csvname="iir-1.csv"

parse_data(load_from_folder_name)