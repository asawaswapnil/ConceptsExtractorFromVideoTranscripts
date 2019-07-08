# encoding: utf-8
#from __future__ import generators
import os
from entity.util.config import config
from entity.util import document as dc
from entity.util import common as cm
from entity.util import similarity as sim 
#from entity.util.parse import parse_data
#from entity.util.convertToTranscript import convert_to_transcript
from entity.models import tfidf,keyword, tf
from entity.util.extractors import extract_top_kcs
import pdb
import pandas as pd

# what is  self.maxdf = max(len(documents) * 0.90,mindf) in tfidf.py
if __name__=='__main__':
	#parse_data("/home/swapnil/ir/educational_videos/educational_videos")
	#BOOK_CORPUS = 'data/sampleSyllabus				.csv'
	# for root, directories, filenames in os.walk("./transcripts"):
	# 	for filename in filenames:
	#print(filename)
	BOOK_CORPUS = "sections.csv"
	VIDEO_CORPUS = "transcripts.csv"

	listbooks = []
	concept = config.TF #'list_filter'
	concept_list_path = config.expert_vocab_for_ir_book #'data/wordlist/irbook_glossary_707.txt' # has the glosary of the book ( all important names)
	video_docs = dc.load_document(VIDEO_CORPUS,
										  booknames=[],
										  textfield=['text'],
										  idfield="docid",
										  otherfields=['bookname'],
										  booknamefield='bookname',
								   )

	train_docs = video_docs #object list. Each object has 1. the text( with a minor formatting) from 1 row (from samplefile.csv). 2. other metadata about text. 
	#pickle.save
	#print(train_docs)
	extract_docs = video_docs

	book_docs = dc.load_document(BOOK_CORPUS,
										  booknames=[],
										  textfield=['text'],
										  idfield="docid",
										  otherfields=['bookname'],
										  booknamefield='bookname',
								   )

	model_dir = "model/"
	concept_dir = "concepts/"
	Define_kc = False
	no_of_topics = 100
	vocabulary_filename = "expert_vocab_for_ir_book.txt"

	
	if concept == config.LIST_FILTER:
		ingram = (1,5) 	
		ptfidf =  model_dir +"m_"+ concept+"ngram"+str(ingram[0])+"_"+str(ingram[1])+".pickle"  # 'model/m_list_filterngram1_5.pickle'
		outdir =concept_dir + config.dir_sep + concept+"_"+vocabulary_filename
		if no_of_topics == -1:
			outdir += "all"
		else:
			outdir += "top" + str(no_of_topics) #concepts//list_filter_irbook_tfidf_keywordtop10
		model = None

		# if not os.path.exists(ptfidf)  or config.Remove_Prev_models: # if u don't want to load existing model
		print("tf-idf model learning")
		# TODO: send in the minibatches with pack pad sequence.
		model = tfidf.TFIDFExtractor(train_docs,vocabulary_filename,ngram=ingram	) #stop wrods, stemming, #<entity.models.tfidf.TFIDFExtractor object at 0x7f93b1fec518>
		model.idf()
		model.save_model(ptfidf)
		# else:
		# 	print(" Model Exists : "+ptfidf)

		#keyword_list = keyword.KeywordList(vocabulary_filename,concept_list_path)
		#  extract_docs are testing documents
		vid2concepts,_ = model.get_Topics(extract_docs) 
		#vid2concepts = keyword_list.get_Topics_secondFilter(vid2concepts) # finally scans glossary to check if the grams of your vid are in the glossary. If yes, add it to the final csv
		vid_df = extract_top_kcs(doc2concepts=vid2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)
		vid_df.to_csv("Video_concepts_"+VIDEO_CORPUS+".csv",header=False)#['bookname','text'],index_label='docid')
		
		doc2concepts,_ = model.get_Topics(book_docs) # gets weights of tf-df for all 1-5 grams of your docs 
		#doc2concepts = keyword_list.get_Topics_secondFilter(doc2concepts) # finally scans glossary to check if the grams of your docs are in the glossary. If yes, add it to the final csv
		doc_df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)
		doc_df.to_csv("sections_concepts_"+BOOK_CORPUS+".csv",header=False)#['bookname','text'],index_label='docid')
		# df = extract_top_kcs_sm(doc2concepts=doc2concepts,define_kc=False,topk=no_of_topics)
		# df.to_csv(outdir,index=None)
		sim.vedios_sections_similarity(doc_df,vid_df,"cosine")

	if concept==config.TF:
		outdir =concept_dir + config.dir_sep + concept+"_"+vocabulary_filename
		if no_of_topics == -1:
			outdir += "all"
		else:
			outdir += "top" + str(no_of_topics) #concepts//list_filter_irbook_tfidf_keywordtop10
		
		ingram = (1,5) 	
		vocabulary_filename = "expert_vocab_for_ir_book.txt"
		ptfidf =  model_dir +"m_"+ concept+"ngram"+str(ingram[0])+"_"+str(ingram[1])+".pickle"  # 'model/m_list_filterngram1_5.pickle'

		# if not os.path.exists(ptfidf)  or config.Remove_Prev_models: # if u don't want to load existing model
		print("tf model learning")
		# TODO: send in the minibatches with pack pad sequence.
		model = tf.TFExtractor(train_docs,vocabulary_filename,ngram=ingram	) #stop wrods, stemming, #<entity.models.tfidf.TFIDFExtractor object at 0x7f93b1fec518>
		model.tf()
		#model.save_model(ptfidf)
		# else:
		#	print(" Model Exists : "+ptfidf)
		#keyword_list = keyword.KeywordList(vocabulary_filename,concept_list_path) # returns the trie after preprocessing the concepts of vocabulary
		# assuming extract_docs are testing documents
		# for i in range(5):
		#_, vid_matrix ,i2w = 
		model.get_Topics(extract_docs,1) # gets weights of tf for all 1-5 grams of your docs , using here any vocabulary
		# vid_mat_df=pd.DataFrame(vid_matrix)
		# i2w_df=pd.DataFrame(i2w)
		#vid_df = extract_top_kcs(doc2concepts=vid2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)
		# vid_mat_df.to_csv("Video_concepts_"+str(i)+VIDEO_CORPUS,header=False)#['bookname','text'],index_label='docid')
		# i2w_df.to_csv("Video_concepts_indexes"+str(i)+VIDEO_CORPUS,header=False)#['bookname','text'],index_label='docid')
		

		# #_, books_matrix,_ = model.get_Topics(book_docs) # gets weights of tf-df for all 1-5 grams of your docs , not using here any glosaary
		# #doc2concepts = keyword_list.get_Topics_secondFilter(doc2concepts) # finally scans glossary to check if the grams of your docs are in the glossary. If yes, add it to the final csv
		# # doc_df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)
		# # doc_df.to_csv("sections_concepts_"+BOOK_CORPUS+".csv",header=False)#['bookname','text'],index_label='docid')
		
		# sim.vedios_sections_similarity(vid_matrix,books_matrix,"jaccard")
