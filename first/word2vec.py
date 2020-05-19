window_size = 2
dimension_size = 64
import string 
punctuation = set(string.punctuation)
from nltk.corpus import stopwords
from numpy import array,append
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

stop_words = set(stopwords.words('english'))
file1 = open('../../../nltk_data/corpora/abc/science.txt',encoding='iso-8859-1')
txt1 = file1.read()
lis1 = txt1.split('\n')

total_sentences = []
for i in lis1:
	if(i!=""):
		total_sentences.append(i)

file2 = open('../../../nltk_data/corpora/abc/rural.txt')
txt2 = file2.read()
lis2 = txt2.split('\n')
for i in lis2:
	if(i!=""):
		total_sentences.append(i)

total_sentences_remove = []
for i in total_sentences:
	sentence = ''.join(j for j in i if j not in punctuation)
	filter_sentence = " ".join([j for j in sentence.lower().split() if j not in stop_words])
	print(filter_sentence)
