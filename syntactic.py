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
	
	tagged_pairs = list((pos_tag(mem), pos_tag(nonmem)) for (mem,nonmem) in mem_non_mem_pairs) 
	
	return tagged_pairs
	  
def unigram_model():
 	bw = [w.lower() for w in brown.words(categories='news')]
 	tagged_bw = pos_tag(bw)
 	#tagged_bw = brown.tagged_words(categories='news')
 	#tagged_bw = [(word.lower(), tag) for (word,tag) in tagged_bw]
 	ugm = NgramModel(2, tagged_bw, lambda f,b:LidstoneProbDist(f,0.2))
 	return ugm
	
def test_quote(quote, ugm):
 	bigrams = nltk.bigrams(quote)
 	prob = 1.0
 	for (w1,w2) in bigrams:
 		prob*= ugm.prob(w2,[w1],True)
 	return prob
	
			
quotes = extract_info()
ugm= unigram_model()
l = list(test_quote(mem, ugm) < test_quote(nonmem,ugm) for (mem,nonmem) in quotes)
