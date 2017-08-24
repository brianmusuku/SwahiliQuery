import csv, random
from collections import Counter
import numpy as np
import re
import collections
import numpy as np

def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

def build_vocab(sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

def buildDictionary(myStr):
        myStr = clean_str(myStr)
        words = myStr.split()
        return build_vocab(words)

def categoricalEncode(words):
        """
        This function takes in a list of words and one hot encodes them.
        """
        result = []
        words = [x.strip() for x in words]
        dicti, reverse = build_vocab(words)
        dimen = len(dicti)
        for w in words:
                enc = np.zeros(dimen)
                enc[dicti[w]] = 1
                result.append(enc)
        return result, dicti, reverse

def oneHotencode(sentence, dictionary, maxLength):
        '''
        Takes in a sentence and returns one hot encoding ector
        representation given a dictionary
        '''
        encoded = []
        sentence = clean_str(sentence)
        words = sentence.split()[:maxLength]
        indices = [dictionary[w] for w in words]
        dimen = len(dictionary)
        for index in indices:
                vector = [0 ]* dimen
                vector[index] = 1
                encoded.append(vector)
        remainder = maxLength-len(indices)
        if remainder>0:
                for i in range(remainder):
                        vector = [0 ]* dimen
                        encoded.append(vector)
        return encoded, len(indices)

def binaryEncode(sentence, dictionary, maxLength, randomVectors):
        encoded = []
        sentence = clean_str(sentence)
        words = sentence.split()[:maxLength]
        indices = [dictionary[w] for w in words]
        dimen = len(dictionary)
        #[[int(x) for x in list(np.random.normal(100, 100, 25))] for i in range(dimen)]
        for index in indices:
                #encoded.append(list(randomVectors[index]))
                encoded.append([int(x) for x in list(np.binary_repr(index, width=25))])
        remainder = maxLength-len(indices)
        if remainder>0:
                for i in range(remainder):
                        vector = [0 ]* 25
                        encoded.append(vector)
        return encoded, len(indices)

def readData(path):
	'''
	This function will read data from a csv file
	and return the corresponding list
	'''
	f = open(path)
	return list(csv.reader(f))

def readLines(path):
	f = open(path)
	return f.readlines()

def underline(sentence, word):
	words = sentence.split()[:10]
	for wordd in words:
		if wordd == word:
			index = words.index(wordd)
			words[index] = '\033[91m'+wordd+'\033[0m'
	return " ".join(words)
			
def simiWords(sent1, sent2):
	"""
	This function will output the number of simlar
	words between two sentences
	"""
	words1 = [x.strip() for x in clean_str(sent1).replace(",", "").split()]
	words2 = [x.strip() for x in clean_str(sent2).replace(",", "").split()]
	a = len([w for w in words2 if w in words1])
	b = len([w for w in words1 if w in words2])
	if a>b: return round(a, 2)
	else: return round(b, 2)






def main():
	maxLength = 15
	path = 'mpesaAnon.csv'
	memPath = 'docAnon.csv'
	data = readData(path)
	#english
	#data = [x for x in data if x[3].strip()=='1']
	memData = readLines(memPath)
	existData = []
	for i, que in enumerate(data):
		msm = []
		for m in memData:
			msm.append(simiWords(que[0], m))
		existData.append(msm)
	no = 500
	sentences = [x[0] for x in data][:no]
	memSentences = [x.replace(",","").replace("\n","") for x in memData]
	yDataRaw = [x[1] for x in data][:no]
	myStr = " ".join(sentences)+" "+" ".join(memSentences)
	dicti = buildDictionary(myStr)[0]
	#initialise all words with random integers
	randomVectors = np.random.random_integers(0,2, size=[len(dicti),25])
	xData = [binaryEncode(sent, dicti, maxLength, randomVectors) for sent in sentences]
	memData = [binaryEncode(sent, dicti, maxLength, randomVectors) for sent in memSentences]
	myStr = "".join(yDataRaw)
	dicti2 = buildDictionary(myStr)[0]
	yData, dicti2, reverse = categoricalEncode(yDataRaw)
	yData = []
	yDataRaw = [int(y) for y in yDataRaw]
	for x in yDataRaw:
		zeros = [0]*16
		zeros[x-1] = 1
		yData.append(zeros)
	#randomly shuffle x and y datasets
	c = list(zip(xData, yData, sentences, existData))
	random.shuffle(c)
	xData, yData, sentences, existData = zip(*c)
	xD = [x[0] for x in xData]
	sL = [x[1] for x in xData]
	mem = [x[0] for x in memData]
	memSl = [x[1] for x in memData]
	split = int(0.7 * len(xData))
	dimen = len(xD[0][0])
	return xD[:split], sL[:split], yData[:split], dimen, maxLength, xD[split:], sL[split:], yData[split:], [existData[split:], existData[:split]], mem, memSl