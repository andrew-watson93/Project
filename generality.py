import re
import nltk
from nltk.corpus import brown
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist
from nltk import word_tokenize
from nltk import wordpunct_tokenize
import nltk.data
from nltk import pos_tag
import vp

fname = 'moviequotes.memorable_nonmemorable_pairs.txt'
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

third_person =  ['anybody','anyone','anything','everybody','everyone','everything','he','her','hers','herself','him','himself','his','I','it','its','itself','neither','nobody','nothing','she','somebody','someone','something','their','theirs','theirself','theirselves','them','themself','themselves','they','this',"what's-his-face","what's-his-name",'you-know-what','you-know-who']

indef = ['a,','an']

past = ['VBD', 'VBN']

present = ['VBG','VBP','VBZ']


third_person =  [word.lower() for word in third_person]

def extract_info():
	

	lines = list(line.strip() for line in open(fname))

	quotes = lines[1::5]
	antiquotes = lines[3::5]

	sent_tokenized_qs = [sent_detector.tokenize(quote) for quote in quotes]
	sent_tokenized_aqs = [sent_detector.tokenize(antiquote) for antiquote in antiquotes]
	
	word_tokenized_qs = [[wordpunct_tokenize(sent) for sent in quote] for quote in sent_tokenized_qs]
	word_tokenized_aqs = [[wordpunct_tokenize(sent) for sent in antiquote] for antiquote in sent_tokenized_aqs]
#	
	#for quote in word_tokenized_qs:
	#	quote[0]=quote[0][1:]
	
	for antiquote in word_tokenized_aqs:
		antiquote[0]=antiquote[0][1:]

	
	flattened_qs = [[tok for sent in quote for tok in sent] for quote in word_tokenized_qs]
	flattened_aqs = [[tok for sent in antiquote for tok in sent] for antiquote in word_tokenized_aqs]
	
	
	lowered_quotes = []
	lowered_antiquotes = []
	for line in flattened_qs:
		lowered_quotes.append([w.lower() for w in line])
	for line in flattened_aqs:
		lowered_antiquotes.append([w.lower() for w in line])
	
	
	mem_non_mem_pairs = zip(lowered_quotes, lowered_antiquotes)
	
	return [(pos_tag(mem), pos_tag(non_mem)) for (mem,non_mem) in mem_non_mem_pairs]
	
def count_feature(quote):
	q_tags = [tag for (w,tag) in quote]
	return sum(q_tags.count(tag) for tag in present)
	
def compare((mem,nonmem)):
	return count_feature(mem) > count_feature(nonmem)

quotes=extract_info()
filter_quotes = [(mem,nonmem) for (mem,nonmem) in quotes if count_feature(mem) > 0]
l = [compare(pair) for pair in filter_quotes]

	
