def cosine(df1, df2):

	arr1=df1.values
	print(arr1)
	arr2=df2.values
	result=[0 for i in range(len(weights))]
	for i in range(len(df1)):
		magOfDoc=0
		d1=0
		d2=0
		numerator=0
		#print(UserWt)
		for l in (UserWt):
			d1=d1+UserWt[l]*UserWt[l]
			d2=d2+weights[i][l]*weights[i][l]
			numerator=numerator+UserWt[l]*weights[i][l]
		dinominator=math.sqrt(d1)*math.sqrt(d2)
		# print(magOfDoc)
		#numerator=sum(a*b for a, b in zip(UserWt.values(),listDictDocs[i].values()) )		
		result[i]=numerator/dinominator
	return result
