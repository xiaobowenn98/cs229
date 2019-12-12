import pandas as pd
import numpy as np
import io 
def read_book(file_name):
  book = io.open(file_name, 'r', encoding = 'ISO-8859-1')
  line  =  book.read()
  return line.lower().replace("\n", " ").split('.')

def get_words(message):
	message = message.replace("?", " ? ")
	message = message.replace("!", " ! ")
	message = message.replace("_", " ").replace("  ", " ")
	a = [word for word in message.split() if word != ""]
	return a

def read_char_lines(char_file):
	file=open(char_file, 'r')
	label_dict= {}
	chars =[]
	for line in file.readlines():
		line = line.replace("\n","").replace(" ","")
		a = line.lower().split(',')
		label_dict[a[1]] = a[0][-1::]
		chars.append(a[1::])
	return label_dict, chars


def get_character_lines(akas_by_char, df_lines):
	chars_all =[]
	lines = []
	for line in df_lines:
		chars = []
		words =get_words(line)
		if words == []:
			continue
		for akas in  akas_by_char:
  			applies =  np.sum([1 for aka in akas if aka in words])>0
  			if applies: 
  				chars.append(akas[0])
		chars_all.append(chars)
		lines.append(words)
	return lines, chars_all


def splitter(book_file_name, akas_by_chars):
	df_book = read_book(book_file_name)
	# charceters, labels = read_label_data(label_file)
	return get_character_lines(akas_by_chars, df_book)

def splitter_2(file_name, akas_by_chars):
	book = io.open(file_name, 'r', encoding = 'ISO-8859-1')
	line = book.read().lower()
	words = get_words(line)
	lines =[]
	chars = []
	for i in range(len(words)):
		word = words[i]
		for akas in akas_by_chars:
			if word in akas:
				lines.append(words[max(i-5,0):min(i+5, len(words)-1)])
				chars.append([akas[0]])
	return lines, chars

	
# hamlet = open("../books/hamlet.txt")
# hamlet_new = open("../books/hamlet_new.txt",'w')
# for line in hamlet.readlines():
# 	print (line)
# 	line2= line.replace(".",":",1)
# 	print (line2)
# 	hamlet_new.write(line)

# hamlet.close()
# hamlet_new.close()
