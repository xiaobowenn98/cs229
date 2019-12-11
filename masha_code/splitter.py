import pandas as pd
import numpy as np
import io 
def read_book(file_name):
  book = io.open(file_name, 'r', encoding = 'ISO-8859-1')
  line  =  book.read()
  return line.lower().replace("\n", " ").split('.')
  # return df

def get_words(message):
	message = message.replace("?", " ? ")
	message = message.replace("!", " ! ")
	message = message.replace("_", " ")
	return message.split()

def read_char_lines(char_file):
	file=open(char_file, 'r')
	label_dict= {}
	chars =[]
	for line in file.readlines():
		line = line.replace("\n","")
		a = line.lower().split(',')
		label_dict[a[1]] = a[0]
		chars.append(a[1::])
	print (chars)
	return label_dict, chars


def get_character_lines(akas_by_char, df_lines):
	chars_all =[]
	lines = []
	# print
	for line in df_lines:
		chars = []
		words =get_words(line)
		# print (words)
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

	
