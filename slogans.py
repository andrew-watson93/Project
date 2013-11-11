import nltk
from nltk.corpus import brown
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist
from nltk import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk import pos_tag
import vp

def compare_models(quote):
#	print str(quote)


#	#print str(bigrams)
#	/afs/inf.ed.ac.uk/user/s10/s1013798/4th Year/Project
	if len(quote) == 1:
#		#quote = [] + quote
		
		mem_prob = mem_model.prob(quote[0],['<S>'], True)
		nonmem_prob = nonmem_model.prob(quote[0],['<S>'], True)

		#bigrams = [(('<S>'), quote[0])]
	if len(quote) == 2:
		#bigrams = [(('<S>'), quote[0])] + nltk.bigrams(quote)
		mem_prob = mem_model.prob(quote[1], ['<S>', quote[0]], True)
		nonmem_prob = nonmem_model.prob(quote[1], ['<S>', quote[0]], True)
#		mem_prob = mem_model.prob(quote[0], [], True)
#		nonmem_prob = nonmem_model.prob(quote[0], [], True)
#		
#	elif len(quote) ==2:
#		mem_prob = mem_model.prob(quote[1], [quote[0]], True)
#		nonmem_prob = nonmem_model.prob(quote[1], [quote[0]], True)
#		
#	else:
#	

#		trigrams =  nltk.trigrams(quote)
	
	
	if len(quote) > 2:
	
		mem_prob = 1.0
		nonmem_prob = 1.0
		
		trigrams = nltk.trigrams(['<S>'] + quote)
	
		print str(trigrams)
	
		for (w1,w2,w3) in trigrams:
			mem_prob*= mem_model.prob(w3,[w1,w2],True)
			nonmem_prob*= nonmem_model.prob(w3,[w1,w2],True)
		
	if mem_prob >= nonmem_prob:
		return 'mem'
	else:
		return 'nonmem'
		
def remove_line_no(line):
	val = len(line)
	i = 0
	new = ''
	numend=False
	while i < val:
		if numend:
			new += line[i]
		else:
			if line[i] == ' ':
				numend = True
		i+=1
	return new

quote_file = 'moviequotes.memorable_nonmemorable_pairs.txt'
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
slogans = 'slogans_distribution.txt'

brown_sents = brown.sents(categories='news')

quote_lines = list(line.strip() for line in open(quote_file))

slogan_lines = list(line.strip() for line in open(slogans))

quotes = quote_lines[1::5]
antiquotes = quote_lines[3::5]

antiquotes = [remove_line_no(line) for line in antiquotes]

quote_str = ''
for quote in quotes:
	quote_str += quote + ' '
	
antiquote_str = ''
for aq in antiquotes:
	antiquote_str += aq + ' '
	
init_word_tokenized_qs = WhitespaceTokenizer().tokenize(quote_str)
init_word_tokenized_aqs =  WhitespaceTokenizer().tokenize(antiquote_str)

word_tokenized_qs = []
word_tokenized_aqs = []

for w in init_word_tokenized_qs:    
	if '.' in w or '?' in w or '!' in w or ',' in w:     
		word_tokenized_qs+= word_tokenize(w)
	else:        
		word_tokenized_qs+=[w]
	 	
for w in init_word_tokenized_aqs:    
	if '.' in w or '?' in w or '!' in w or ',' in w:     
		word_tokenized_aqs+= word_tokenize(w)
	else:        
		word_tokenized_aqs+=[w]



word_tokenized_qs = [w.lower() for w in word_tokenized_qs]
word_tokenized_aqs = [w.lower() for w in word_tokenized_aqs]


	
#sent_tokenized_qs = sent_detector.tokenize(quote_str)
#sent_tokenized_aqs = sent_detector.tokenize(antiquote_str)

#word_tokenized_qs = [word_tokenize(sent) for sent in sent_tokenized_qs]
#word_tokenized_aqs = [word_tokenize(sent) for sent in sent_tokenized_aqs]

#word_tokenized_qs = [word_tokenize(sent) for sent in sent_tokenized_qs]
#word_tokenized_aqs = [word_tokenize(sent) for sent in sent_tokenized_aqs]


#flattened_qs = [word for sent in word_tokenized_qs for word in sent]
#flattened_aqs = [word for sent in word_tokenized_aqs for word in sent]

init_tagged_qs = pos_tag(word_tokenized_qs)
init_tagged_aqs = pos_tag(word_tokenized_aqs)

init_tagged_qs = [tag for (word,tag) in init_tagged_qs]
init_tagged_aqs = [tag for (word,tag) in init_tagged_aqs]

tagged_qs = []

for tag in init_tagged_qs:
	tagged_qs.append(tag)
	if tag == '.':
		tagged_qs.append('<S>')
tagged_aqs = []		
	
for tag in init_tagged_aqs:
	tagged_aqs.append(tag)
	if tag == '.':
		tagged_aqs.append('<S>')

#i = 0
#end = False

#while end == False:
#	(word,tag) = tagged_qs[i]
#	if '.' in word or '?' in word or '!' in word:
#		tagged_qs.insert(i+1, ('<S>', '<S>'))
#		if (i + 2) > (len(tagged_aqs) -1):
#			end = True
#		else:
#			i += 1
#	elif (i + 1) > (len(tagged_qs) -1):
#		end = True
#	else:
#		i+=1
#		
#tagged_qs.append(('<S>','<S>'))




#tagged_aqs = []

#for (word,tag) in tagged_aqs:
#	print str((word,tag))
#	tagged_aqs.append((word,tag))
#	if tag == '.':
#		tagged_aqs.append(('<S>', '<S>'))



#while end == False:
#	(word,tag) = tagged_aqs[i]
#	if ('.' in word) or ('?' in word) or ('!' in word):
#		tagged_aqs.insert(i+1, ('<S>', '<S>'))
#		if (i + 1) == (len(tagged_aqs) -1):
#			end = True
#		else:
#			i += 1
#	elif (i + 1) == (len(tagged_aqs) -1):
#		end = True
#	else:
#		i+=1

#tagged_aqs.append(('<S>','<S>'))


mem_model =  NgramModel(3, tagged_qs, lambda f,b:LidstoneProbDist(f,0.2))
nonmem_model =  NgramModel(3, tagged_aqs, lambda f,b:LidstoneProbDist(f,0.2))

#word_tokenized_news = brown.sents(categories='news')
#lowered_news = [[word.lower() for word in sent] for sent in word_tokenized_news]

#tagged_news = [pos_tag(sent) for sent in lowered_news]
#lowered_news = [[(word.lower(),tag) for (word,tag) in sent] for sent in tagged_news]



sent_tokenized_slogans = [sent_detector.tokenize(slogan) for slogan in slogan_lines]
init_word_tokenized_slogans = [[WhitespaceTokenizer().tokenize(sent) for sent in slogan] for slogan in sent_tokenized_slogans]

word_tokenized_slogans = []

for slogan in init_word_tokenized_slogans:
	flattened_slogan = [word for sent in slogan for word in sent]
	new_slogan = []
	for w in flattened_slogan:
		if '.' in w or '?' in w or '!' in w or ',' in w:
			new_slogan += word_tokenize(w)
		else:
			new_slogan += [w]
	word_tokenized_slogans += [new_slogan] 	     

#for slogan in init_word_tokenized_slogans:
#	new_slogan = []
#	for word in slogan:    
#		if '.' in w or '?' in w or '!' in w or ',' in w:     
#			new_slogan+= word_tokenize(w)
#	 	else:        
#	 		new_slogan+=[w]
#	word_tokenized_slogans+= [new_slogan]

#word_tokenized_slogans = [[word_tokenize(sent) for sent in slogan] for slogan in sent_tokenized_slogans]

lowered_slogans = [[word.lower() for word in slogan] for slogan in word_tokenized_slogans]

init_tagged_slogans = [pos_tag(slogan) for slogan in lowered_slogans]
init_tagged_slogans = [[tag for (word,tag) in slogan] for slogan in init_tagged_slogans]




tagged_slogans = []

for slogan in init_tagged_slogans:
	
	new_slogan = []
	for tag in slogan:
		#print str((word,tag))
		new_slogan.append(tag)
		if tag == '.':
			new_slogan.append('<S>')
	
	tagged_slogans.append(new_slogan)



results = [compare_models(slogan) for slogan in tagged_slogans]

#results = [compare_models(sent) for sent in lowered_news]

percentage_more_likely_in_mem = (float(results.count('mem'))/float(len(results))) *100.0

print str(percentage_more_likely_in_mem)

