
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import itertools
import heapq
from entity.util import nlp,common as cm
import nltk
import pickle
import spacy
import numpy as np
import pandas as pd
nlp_spacy = spacy.load('en')

class TFExtractor():
	'''
	Word2Vec_features
	'''
	def __init__(self, documents,vocabulary_filename,ngram = (1,3) ,mindf = 1):
		print(np.intp)
		pdocuments = nlp.preprocessed_docs(documents) # remove stop words and stem
		self.raw_documents = documents
		self.docs = [ ' '.join(doc.tokens) for doc in pdocuments] # list of str per line of csv: ['book text test code load text gamma encod', 'book text test code load text basket flower', 'book text test code load text walk roam', 'book text test code load text', 'book text test code load text', 'book mir text test code load text']
		length=len(self.docs)
		# self.docs1=np.array(self.docs[0:length/5])
		# self.docs2=np.array(self.docs[length/5:2*length/5])
		# self.docs3=np.array(self.docs[2*length/5:3*length/5])
		# self.docs4=np.array(self.docs[3*length/5:4*length/5])
		# self.docs5=np.array(self.docs[4*length/5:length])

		self.maxdf = max(len(documents) * 0.90,mindf) # max of( no.of docs*0.9, 1)
		#maxdf=When building the vocabulary tfidf table ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
		#mindf=Ignore when frequency strictly higher than the given threshold
		self.mindf = mindf 
		self.ngram = ngram 
		self.trained = False
		self.vocabulary_filename="/home/swapnil/ir/gitDocs/ConceptExtractor/data/wordlist/expert_vocab_for_ir_book_v2.csv"
		vocabname=str(self.vocabulary_filename)
		vocabulary=pd.read_csv(vocabname,header=0,delimiter=',').values.tolist() 
		self.vocabulary_dict={row[1]: row[0]-1 for row in vocabulary }
		print(self.vocabulary_dict)

	def tf(self):
		print("TFIDF training for " + str(self.ngram))
		self.i=0
		countModel = CountVectorizer(analyzer='word', ngram_range=self.ngram,min_df=self.mindf,max_df=self.maxdf, vocabulary=self.vocabulary_dict )#initialize  to Convert a collection of raw documents to a matrix of TF features.
		#self.docs=np.array(self.docs[self.i*length/5:(self.i+1)*length/5])
		self.matrix = countModel.fit(self.docs)
		self.model=countModel
		self.i2w = countModel.get_feature_names()
		#print(vectorizer.get_feature_names())
		self.trained = True
		#Convert a collection of raw documents to a matrix of TF-IDF features.  
	def save_model(self,filename):
		pickle.dump(self, open(filename,'wb'))

	def get_Topics(self,topicdocs,i):
		# same preprocessing as in init function but for testing docs  
		ptopicdocs = nlp.preprocessed_docs(topicdocs)
		docs = [' '.join(doc.tokens) for doc in ptopicdocs]
		if not self.trained == True:
			self.train()
		print("before matrix")
		matrix=self.model.transform(docs).todense()
		print("before saving")
		pd.DataFrame(matrix).to_csv("Video_concepts_"+"v2_transcripts.csv",header=False)#['bookname','text'],index_label='docid')
		i2w_df=pd.DataFrame(self.i2w)
		i2w_df.to_csv("Video_concepts_indexes"+"v2_transcripts.csv",header=False)#['bookname','text'],index_label='docid')

		#print("matrix",len(matrix),len(matrix[0]))
		# topic_dic = {}
		# i = 0
		# for doci in ptopicdocs:
		# 	#print("inside wierdo",matrix[i].tolist()[0])
		# 	#print(matrix[i].tolist()[0])
		# 	temptokens = zip(matrix[i].tolist()[0], itertools.count())
		# 	#print("temptokens",temptokens)
		# 	temptokens1 = []
		# 	for (x, y) in temptokens:
		# 		#print(x,y)
		# 		#doc_to_all_words.append((x, self.i2w[y]))
		# 		if x > 0.0  :
		# 			temptokens1.append( (x, self.i2w[y]))
		# 			#examples of (x, self.i2w[y]) : 
		# 			#0.01506181969686083 text inform
		# 			#0.01506181969686083 text inform retriev
		# 			# this is all in 1 document
		# 	# print("doci",doci)
		# 	# print("doc id",doci[0])
		# 	topic_dic[doci.id ]= temptokens1

		# 	i += 1
		# print("topic",topic_dic)
		# print("mat",matrix)
		#return topic_dic,matrix, self.i2w # dictionary of every docid: document list, with tf-idf value. 




		def npchunk(self,doc):
			npchunklist = []
			for sen in doc:
				ichunklist = list(nlp_spacy(sen).noun_chunks)
				ichunklist = [ nlp.preprocessText(str(ichunk.text)) for ichunk in ichunklist]
				ichunklist = [ichunk for ichunk in ichunklist if len(ichunk) > 0]
				# ichunklistt = [' '.join(ichunk)  for ichunk in ichunklist if len(ichunk) <= 3 and len(ichunk) > 0]
				for ichunk in ichunklist:
					if len(ichunk) <= 3  and len(ichunk) >0 :
						npchunklist.append(' '.join(ichunk))
					elif len(ichunk) > 3:
						newchunks = nltk.ngrams(ichunk,3)
						for nc in newchunks:
							npchunklist.append(' '.join(nc))

			return list(set(npchunklist))

	def get_Topics_npFilter(self,topicdocs):
		ptopicdocs = nlp.preprocessed_docs(topicdocs)
		docs = [' '.join(doc.tokens) for doc in ptopicdocs]

		if not  self.trained == True:
			self.train()
		matrix = self.model.transform(docs).todense()
		topic_dic = {}
		i = 0
		for doci in ptopicdocs:
			chunks = self.npchunk(doci.sentences)
			temptokens = zip(matrix[i].tolist()[0], itertools.count())
			temptokens1 = []
			tfidf_dic={}
			for (x, y) in temptokens:
				if x > 0.0 :
					tfidf_dic[self.i2w[y]] = x

			for chunk in chunks:
				if chunk in tfidf_dic:
					temptokens1.append((tfidf_dic[chunk],' '.join(nlp.preprocessText(chunk,stemming=False,stopwords_removal=False))))

			topic_dic[doci.id] = temptokens1
			i+=1

		return topic_dic

	def get_Topics_goldFilter(self,topicdocs):
		gconcepts = cm.getConcepts()
		ptopicdocs = nlp.preprocessed_docs(topicdocs)
		docs = [' '.join(doc.tokens) for doc in ptopicdocs]

		if not  self.trained == True:
			self.train()
		matrix = self.model.transform(docs).todense()
		topic_dic = {}
		i = 0
		for doci in ptopicdocs:
			temptokens = zip(matrix[i].tolist()[0], itertools.count())
			temptokens1 = []
			tfidf_dic={}
			for (x, y) in temptokens:
				if x > 0.0 :
					tfidf_dic[self.i2w[y]] = x

			for chunk in gconcepts:
				if chunk in tfidf_dic:
					temptokens1.append((tfidf_dic[chunk],chunk))
			if len(temptokens1) == 0:
				temptokens1.append((1,"dummy"))
			topic_dic[doci.id] = temptokens1
			i+=1

		return topic_dic


	def get_Topics_listFilter(self,topicdocs,concept_list_file,gold_concept_file):
		gconcepts = cm.getConcepts(gold_concept_file)
		ptopicdocs = nlp.preprocessed_docs(topicdocs)
		docs = [' '.join(doc.tokens) for doc in ptopicdocs]

		if not  self.trained == True:
			self.train()
		matrix = self.model.transform(docs).todense()
		topic_dic = {}
		i = 0
		for doci in ptopicdocs:
			temptokens = zip(matrix[i].tolist()[0], itertools.count())
			temptokens1 = []
			tfidf_dic={}
			for (x, y) in temptokens:
				if x > 0.0 :
					tfidf_dic[self.i2w[y]] = x

			for chunk in gconcepts:
				if chunk in tfidf_dic:
					temptokens1.append((tfidf_dic[chunk],chunk))
			if len(temptokens1) == 0:
				temptokens1.append((1,"dummy"))
			topic_dic[doci.id] = temptokens1
			i+=1

		return topic_dic

	def get_Topics_partialSectionFilter(self,topicdocs,section_wise_folder):
		gconcepts,sectionwise_concepts = getPartialSectionConcepts(section_wise_folder)
		ptopicdocs = nlp.preprocessed_docs(topicdocs)
		docs = [' '.join(doc.tokens) for doc in ptopicdocs]

		if not  self.trained == True:
			self.train()
		matrix = self.model.transform(docs).todense()
		topic_dic = {}
		i = 0
		for doci in ptopicdocs:
			temptokens = zip(matrix[i].tolist()[0], itertools.count())
			temptokens1 = []
			tfidf_dic={}
			for (x, y) in temptokens:
				if x > 0.0 :
					tfidf_dic[self.i2w[y]] = x

			for chunk in gconcepts:
				if chunk in tfidf_dic:
					temptokens1.append((tfidf_dic[chunk],chunk))
			if len(temptokens1) == 0:
				temptokens1.append((1,"dummy"))
			print("doc id",doci.id)
			topic_dic[doci.id] = temptokens1
			i+=1

		return topic_dic

def getGlobalngrams(grams,documents,threshold):

	singlecorpus = ""
	for doc in documents:
		singlecorpus += ' '+ doc.text + '\n'


	ncorpus = ' '.join(nlp.preprocessText(singlecorpus))
	tf = TfidfVectorizer(analyzer='word', ngram_range=grams, stop_words=nlp.stopwords)
	tfidf_matrix = tf.fit_transform([ncorpus])
	feature_names = tf.get_feature_names()
	doc = tfidf_matrix.todense()
	temptokens = zip(doc.tolist()[0], itertools.count())
	temptokens = [(x, y) for (x, y) in temptokens if x > threshold]
	tokindex = heapq.nlargest(len(temptokens), temptokens)
	global1grams = dict([(feature_names[y],x) for (x, y) in tokindex ])
	topindex = [ (feature_names[y],x)  for (x,y) in tokindex ]
	f = open('data/file'+str(grams[0])+".txt",'w')
	for key in global1grams:
		f.write(key+","+global1grams[key]+"\n")


	return  global1grams,topindex