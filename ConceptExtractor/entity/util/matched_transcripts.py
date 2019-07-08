def matched_transcripts_fn(book_docs, video_docs, topn):
	lst=[ book_docs[i].txt for i in range(len(book_docs))
	for i in range(len(topn))
		lst[i].append([book_docs[i].text])
