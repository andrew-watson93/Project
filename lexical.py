import nltk
from nltk.corpus import brown
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
import vp

fname = 'moviequotes.memorable_nonmemorable_pairs.txt'
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def extract_info():
	
	lines = list(line.strip() for line in open(fname))

	quotes = lines[1::5]
	antiquotes = lines[3::5]

	sent_tokenized_qs = [sent_detector.tokenize(quote) for quote in quotes]
	sent_tokenized_aqs = [sent_detector.tokenize(antiquote) for antiquote in antiquotes]
	
	word_tokenized_qs = [[word_tokenize(sent) for sent in quote] for quote in sent_tokenized_qs]
	word_tokenized_aqs = [[word_tokenize(sent) for sent in antiquote] for antiquote in sent_tokenized_aqs]
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


	return mem_non_mem_pairs
	  
def unigram_model():
	#bw = [w.lower() for w in brown.words(categories='news')]
	ugm = NgramModel(1,[w.lower() for w in brown.words(categories='news')], lambda f,b:LidstoneProbDist(f,0.2))
	#num_words = len(set(bw))
	return ugm
	
def test_quote(quote, ugm):
	
	#trigrams = nltk.bigrams(quote)
	prob = 1.0
	for word in quote:
		prob*= ugm.prob(word,[],True)

	return prob
	
#def 
			
quotes = extract_info()
ugm= unigram_model()
l = list(test_quote(mem, ugm) < test_quote(nonmem,ugm) for (mem,nonmem) in quotes)

