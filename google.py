import urllib2
import re

def results(word):
    text = urllib2.urlopen('http://www.google.com/search?q=%s'%word).read()
    m = re.search('About ([0-9,]+) results', text)
    if m is None:
        return None
    else:
        return int(m.group(1).replace(',', ''))
