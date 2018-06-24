from nltk import ngrams

def N_gram(discretizeTrainData, discretizedTestData, window_size, threshold):

	#states object
	states = {}

	#get the ngrams
	grams = ngrams(discretizeTrainData, window_size)
	count = 0.0

	previousGram = ""

	try:
   		while True:
   			#get the next gram
			gram = grams.next()

			#create key string
			currentGram = ''.join(gram)
			key = previousGram+"-"+currentGram


			#check if the transition already existed
			if key in states:
				states[key] = states[key]+1.0
			else:
				states[key] = 1.0

			count = count + 1.0

			#store the old key
			previousGram = currentGram
	except StopIteration:
		pass	#python has no hasNext() kinde function, so we stop with StopIteration error

	# determine the odds
	for key in states:
		odds = states[key] / count
		#print key + " => " + str(odds)
		states[key] = states[key] / count

	# now check the test data
	grams = ngrams(discretizedTestData, window_size)


	print states

	itemCount = 0

	previousGram = ""

	anomalyList = []

	try:
   		while True:
   			#get the next gram
			gram = grams.next()

			itemCount = itemCount + 1

			#create key string
			currentGram = ''.join(gram)
			key = previousGram+"-"+currentGram

			#check the odds of this transition
			if previousGram == "":
				# nothing
				anomalyList.append(0)
			elif key not in states:
				#no botnet, because it's a unique connection costs
				anomalyList.append(0)
			elif states[key] > threshold:
				print "Bot net connection: "+key+" higher then threshold"
				anomalyList.append(1)
			else:
				anomalyList.append(0)

			#store the old key
			previousGram = currentGram
	except StopIteration:
		pass

	#print len(itemCount)

	# anomalyList.append(0)
	# anomalyList.append(0)
	#print len(anomalyList)
	#print len(discretizedTestData)

	#print discretizedTestData




	#create the probabilities array
	#for i in range(len(grams)):
	#	states[createKey(i, grams)] = 1
	#
	#print states

	return anomalyList

def createKey(idx, grams):
	if idx == len(grams):
		return "-"

	currentState = ""
	for j in range(len(grams[idx])):
		currentState += grams[idx][j]

	futureState = ""
	for j in range(len(grams[idx+1])):
		futureState += grams[idx+1][j]

	return currentState + "-" + futureState
