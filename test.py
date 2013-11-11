

def contains_letters(string):
	if string == '':
		return False
	else:
		w=string[0]
		if w.isalpha():
			return True
		else:
			rest = string[1:]
			return contains_letters(rest)

