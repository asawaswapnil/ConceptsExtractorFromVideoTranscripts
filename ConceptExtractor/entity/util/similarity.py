import numpy as np
import math 
import copy
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from entity.util.config import config
import pandas as pd
from operator import itemgetter
BOOK_CORPUS = config.BOOK_CORPUS
VIDEO_CORPUS = config.VIDEO_CORPUS 
sections_mapping=config.sections_mapping
def cosine(section,video):
	numerator=0
	dinominator=0
	square_magnitude_of_video=0
	square_magnitude_of_section=0
	for i in range(len(section)):
		for j in range(len(video)):
			if(section[i,0] ==video[j,0]):
				numerator= numerator+section[i,1]*video[j,1]
			square_magnitude_of_video+=video[j,1]**2
		square_magnitude_of_section+=section[i,1]**2
	dinominator=math.sqrt(square_magnitude_of_video)*math.sqrt(square_magnitude_of_section)	
	result=numerator/dinominator
	return result
				
def csv_to_dictionary(df1):
	lst=df1.values.tolist()
	return dict((x, y) for x, y in lst)

def binarize(arr):
	arr[arr>0]=1
	return arr

def cosine_of_arrays(book_docs, video_docs,vid_mat, sec_mat):
	n=20
	topn_index=[[]for i in range(len(sec_mat))]
	topn_matches=[[]for i in range(len(sec_mat))]
	df = pd.read_csv(sections_mapping,header=0)
	mapping_dict =csv_to_dictionary(df)
	sorted_similarity_list=[[0 for i in range(len(vid_mat))] for j in range(len(sec_mat))]
	similarity_list=cosine_similarity(sec_mat,vid_mat)
	for section_id in range(len(sec_mat)):
		sorted_similarity_list[section_id]= [ [index, value] for index, value in enumerate(similarity_list[section_id] )]
		sorted_similarity_list[section_id]=sorted(sorted_similarity_list[section_id],  reverse=True, key=itemgetter(1))
		topn_matches[section_id]=[mapping_dict[book_docs[section_id].otherfields["bookname"]],  book_docs[section_id].text]
		i=0
		j=0 	
		while (j <n):
			if(sorted_similarity_list[section_id][i][0]==0 or video_docs[sorted_similarity_list[section_id][i][0]].text!=topn_matches[section_id][-1]):
				topn_matches[section_id].append(sorted_similarity_list[section_id][i][1])
				topn_matches[section_id].append(video_docs[sorted_similarity_list[section_id][i][0]-1].text)
				j+=1
			i+=1 	
	pd.DataFrame(topn_matches).to_csv(config.SAVE_CSV_PATH+"sections_videos_matches_with_cosine_vocab_no_voting_filter.csv",header=False)
	return topn_index

def euclidean_of_arrays(book_docs, video_docs,vid_mat, sec_mat):
	n=20
	topn_index=[[]for i in range(len(sec_mat))]
	topn_matches=[[]for i in range(len(sec_mat))]
	df = pd.read_csv(sections_mapping,header=0)
	mapping_dict =csv_to_dictionary(df)
	sorted_similarity_list=[[0 for i in range(len(vid_mat))] for j in range(len(sec_mat))]
	similarity_list=euclidean_distances(sec_mat,vid_mat)
	for section_id in range(len(sec_mat)):
		sorted_similarity_list[section_id]= [ [index, value] for index, value in enumerate(similarity_list[section_id] )]
		sorted_similarity_list[section_id]=sorted(sorted_similarity_list[section_id],  key=itemgetter(1))
		topn_matches[section_id]=[mapping_dict[book_docs[section_id].otherfields["bookname"]],  book_docs[section_id].text]
		i=0
		j=0 	
		while (j <n):
			if(sorted_similarity_list[section_id][i][0]==0 or video_docs[sorted_similarity_list[section_id][i][0]].text!=topn_matches[section_id][-1]):
				topn_matches[section_id].append(sorted_similarity_list[section_id][i][1])
				topn_matches[section_id].append(video_docs[sorted_similarity_list[section_id][i][0]-1].text)
				j+=1
			i+=1 	
	pd.DataFrame(topn_matches).to_csv(config.SAVE_CSV_PATH+"sections_videos_matches_with_euclidian_distance_vocab_no_voting_filter.csv",header=False)
	return topn_index

def jaccard_of_arrays(book_docs, video_docs,vid_mat, sec_mat):
	n=20
	topn_index=[[]for i in range(len(sec_mat))]
	topn_matches=[[]for i in range(len(sec_mat))]
	binary_videos=[[-1 for i in range(len(vid_mat[0]))] for j in range(len(vid_mat))]
	binary_sections=[[-1 for i in range(len(sec_mat[0]))] for j in range(len(sec_mat))]
	similarity_list=[[-1 for i in range(len(vid_mat))] for j in range(len(sec_mat))] 
	sorted_similarity_list=[[0 for i in range(len(vid_mat))] for j in range(len(sec_mat))]
	binary_sections=binarize(sec_mat)
	binary_videos=binarize(vid_mat)
	df = pd.read_csv(sections_mapping,header=0)
	mapping_dict =csv_to_dictionary(df)
	for section_id in range(len(sec_mat)):
		for video_id in range(len(vid_mat)):
			similarity_list[section_id][video_id]=jaccard_score(binary_sections[section_id],binary_videos[video_id])
		sorted_similarity_list[section_id]=[ [index, value] for index, value in enumerate(similarity_list[section_id] )]
		sorted_similarity_list[section_id]=sorted(sorted_similarity_list[section_id], reverse=True, key=itemgetter(1))
		#topn_matches[section_id]=[mapping_dict[book_docs[section_id].otherfields["bookname"]],  book_docs[section_id].text]
		i=0
		j=0 	
		while (j <n):
			if(sorted_similarity_list[section_id][i][0]==0 or video_docs[sorted_similarity_list[section_id][i][0]].text!=topn_matches[section_id][-1]):
				topn_matches[section_id].append(sorted_similarity_list[section_id][i][1])
				topn_matches[section_id].append(video_docs[sorted_similarity_list[section_id][i][0]-1].text)

				if(section_id==0):
					print(book_docs[0].otherfields)
					print(sorted_similarity_list[section_id])
					# print(video_docs[sorted_similarity_list[section_id][i][0]-1,0].text)
					print(video_docs[sorted_similarity_list[section_id][i][0],0].text)
				j+=1
			i+=1 	
	pd.DataFrame(topn_matches).to_csv(config.SAVE_CSV_PATH+"sections_videos_matches_with_jaccard_vocab_no_voting_filter.csv",header=False)
	return topn_index

def vedios_sections_similarity(book_docs, video_docs,books_matrix,vid_matrix, type_of_similarity):
	vid_mat=np.asarray(vid_matrix)
	sec_mat=np.asarray(books_matrix)
	if( type_of_similarity=="cosine"):
		topn=cosine_of_arrays(book_docs, video_docs,vid_mat, sec_mat)	
	elif(type_of_similarity=="jaccard"):
		topn=jaccard_of_arrays(book_docs, video_docs,vid_mat, sec_mat)
	return topn
