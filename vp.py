from nltk.model import NgramModel
from itertools import chain
from nltk.probability import (ConditionalProbDist, ConditionalFreqDist,
                              MLEProbDist, FreqDist, LidstoneProbDist,
                              ProbDistI)
from nltk.util import ingrams
from math import log

# add cutoff
def __init__(self, n, train, estimator=None):
  """
  Creates an ngram language model to capture patterns in n consecutive
  words of training text.  An estimator smooths the probabilities derived
  from the text and may allow generation of ngrams not seen during training.

  @param n: the order of the language model (ngram size)
  @type n: C{int}
  @param train: the training text
  @type train: C{list} of C{string}
  @param estimator: a function for generating a probability distribution
  @type estimator: a function that takes a C{ConditionalFreqDist} and returns
    a C{ConditionalProbDist}
  """

  self._n = n
  self._N = 1+len(train)-n

  if estimator is None:
    estimator = lambda fdist, bins: MLEProbDist(fdist)

  if n == 1:
    fd=FreqDist(train)
    self._model=estimator(fd,fd.B())
  else:
    cfd = ConditionalFreqDist()
    self._ngrams = set()
    self._prefix = ('',) * (n - 1)

    for ngram in ingrams(chain(self._prefix, train), n):
      self._ngrams.add(ngram)
      context = tuple(ngram[:-1])
      token = ngram[-1]
      cfd[context].inc(token)

    self._model = ConditionalProbDist(cfd, estimator, len(cfd))

  # recursively construct the lower-order models
  if n > 1:
    self._backoff = NgramModel(n-1, train, estimator)

NgramModel.__init__ = __init__

def prob(self, word, context,verbose=False):
  '''Evaluate the probability of this word in this context.'''

  context = tuple(context)
  if self._n==1:
    if not(self._model.SUM_TO_ONE):
      # Smoothing model will do the right thing for unigrams
      return self._model.prob(word)
    else:
      raise RuntimeError("No probability mass assigned to word"
                         "%s in context %s" % (word,
                                               ' '.join(context)))
  if context + (word,) in self._ngrams:
    return self[context].prob(word)
  else:
    #if verbose:
     # print "backing off for %s"%(context+(word,),)
    return self._alpha(context) * self._backoff.prob(word, context[1:], verbose)

NgramModel.prob = prob

def logprob(self, word, context, verbose=False):
  '''Evaluate the (negative) log probability of this word in this context.'''

  return -log(self.prob(word, context, verbose), 2) 

NgramModel.logprob = logprob

def entropy(self, text, verbose=False, perItem=False):
  '''Evaluate the total entropy of a text with respect to the model.
  This is the sum of the log probability of each word in the message.'''

  e = 0.0
  m = len(text)
  cl = self._n - 1
  for i in range(cl, m):
    context = tuple(text[i - cl : i ])
    token = text[i]
    e += self.logprob(token, context, verbose)
  if perItem:
    return e/(m-cl)
  else:
    return e

NgramModel.entropy = entropy

def __repr__(self):
  return '<NgramModel with %d %d-grams>' % (self._N, self._n)

NgramModel.__repr__=__repr__

def __contains__(self, item):
  return item in self._model
NgramModel.__contains__=__contains__

def __getitem__(self, item):
  return self._model[item]
NgramModel.__getitem__=__getitem__

# cf self.SUM_TO_ONE
def discount(self):
    """
    @return: The ratio by which counts are discounted on average: c*/c
    @rtype: C{float}
    """
    return 1.0

ProbDistI.discount=discount

def __contains__(self, item):
  return item in self._freqdist
  
LidstoneProbDist.__contains__=__contains__
