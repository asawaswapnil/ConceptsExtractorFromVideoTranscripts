# encoding: utf-8
from __future__ import generators
import os
from entity.util.config import config
from entity.util import document as dc
from entity.util import common as cm
from entity.models import tfidf,keyword
from entity.util.extractors import extract_top_kcs
import pdb

if __name__=='__main__':


	#BOOK_CORPUS = 'data/sampleSyllabus.csv'
	BOOK_CORPUS = 'data/sample_file.csv'
	#pdb.set_trace()
	listbooks = []

	concept = config.LIST_FILTER #'list_filter'
	keyword_list_path = config.irbook_glossary_list #'data/wordlist/irbook_glossary_707.txt' # has the glosary of the book ( all important names)


	bookdocs_16 = dc.load_document(BOOK_CORPUS,
										  booknames=[],
										  textfield=['text'],
										  idfield="docid",
										  otherfields=['bookname'],
										  booknamefield='bookname',
								   )

	train_docs = bookdocs_16#object list. Each object has the text(from samplefile.csv) seperated by period.
	extract_docs = bookdocs_16
	model_dir = "model/"
	concept_dir = "concepts/"
	Define_kc = False
	no_of_topics = 10



	if concept == config.TFIDFNP:

		ptfidf =  model_dir +"m_"+ concept +".pickle"
		outdir =concept_dir + config.dir_sep + concept

		if no_of_topics == -1:
			outdir += "all" + config.file_ext
		else:
			outdir += "top" + str(no_of_topics) + config.file_ext

		model = None

		if not os.path.exists(ptfidf) or config.Remove_Prev_models:
			model = tfidf.TFIDFExtractor(train_docs,ngram=(1,5),mindf=1)
			model.train()
			model.save_model(ptfidf)
		else:
			print(" Model Exists : "+ptfidf)

		model = cm.load_model(ptfidf)

		doc2concepts = model.get_Topics_npFilter(extract_docs)

		df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)

	if concept == config.TFIDF:

		ptfidf =  model_dir +"m_"+ concept +".pickle"
		outdir =concept_dir + config.dir_sep + concept

		if no_of_topics == -1:
			outdir += "all" + config.file_ext
		else:
			outdir += "top" + str(no_of_topics) + config.file_ext

		model = None

		if not os.path.exists(ptfidf)  or config.Remove_Prev_models:
			model = tfidf.TFIDFExtractor(train_docs,ngram=(1,5),mindf=1)
			model.train()
			model.save_model(ptfidf)
		else:
			print(" Model Exists : "+ptfidf)

		model = cm.load_model(ptfidf)

		doc2concepts = model.get_Topics(extract_docs)

		df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)

	if concept == config.NGRAMS:

		ptfidf =  model_dir +"m_"+ concept +".pickle"
		outdir =concept_dir + config.dir_sep + concept

		if no_of_topics == -1:
			outdir += "all" + config.file_ext
		else:
			outdir += "top" + str(no_of_topics) + config.file_ext

		model = None

		if not os.path.exists(ptfidf)  or config.Remove_Prev_models:
			model = tfidf.TFIDFExtractor(train_docs,ngram=(1,5),mindf=1)
			model.train()
			model.save_model(ptfidf)
		else:
			print(" Model Exists : "+ptfidf)

		model = cm.load_model(ptfidf)
		doc2concepts = model.get_Topics(extract_docs)
		df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=-1)

	if concept == config.LIST_FILTER:
		ingram = (1,5)
		name = "irbook_tfidf_keyword"
		ptfidf =  model_dir +"m_"+ concept+"ngram"+str(ingram[0])+"_"+str(ingram[1])+".pickle"  # 'model/m_list_filterngram1_5.pickle'
		# what is this pikle file?
		#hwat is pgram why is it (1,5) 
		outdir =concept_dir + config.dir_sep + concept+"_"+name

		if no_of_topics == -1:
			outdir += "all"
		else:
			outdir += "top" + str(no_of_topics) #concepts//list_filter_irbook_tfidf_keywordtop10
 	 	
		model = None

		if not os.path.exists(ptfidf)  or config.Remove_Prev_models:
			model = tfidf.TFIDFExtractor(train_docs,ngram=ingram) #<entity.models.tfidf.TFIDFExtractor object at 0x7f93b1fec518>

			model.train()
			model.save_model(ptfidf)
		else:
			print(" Model Exists : "+ptfidf)

		model = cm.load_model(ptfidf)
		keyword_list = keyword.KeywordList(name,keyword_list_path)
		doc2concepts_ngram = model.get_Topics(extract_docs) # gets weights of tf-df for all 1-5 grams
		doc2concepts = keyword_list.get_Topics_secondFilter(doc2concepts_ngram)
		#doc2concepts=
		#{'2101': [(0.26174105944121057, 'gamma encod'), (0.11618668413038978, 'test')], 
		#'2102': [(0.11618668413038978, 'test')], 
		#'2103': [(0.11618668413038978, 'test')],
		#'2104': [(0.20704219891568781, 'test')], 
		#'2105': [(0.20704219891568781, 'test')], 
		#'2111': [(0.12531911933479278, 'test')]}


		df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)
		print(df)
		# df = extract_top_kcs_sm(doc2concepts=doc2concepts,define_kc=False,topk=no_of_topics)
		# df.to_csv(outdir,index=None)