import splitter
from splitter import *
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.append( '../')
import rnn
import embed
# import ethan_code/model_rnn
def analyze_book(model_predict_function, splitter, book, characters):
	lines, character_list = splitter(book, characters)
	print (character_list)
	# clean_input = [clean_and_process(line) for line in lines]
	output = model_predict_function(lines)
	# print (output)
	character_score_counts = {}
	character_score_avarages = {}
	for i in range(len(output)):
		for c in character_list[i]:
			if not c in character_score_counts:
				character_score_counts[c] = (0,0)
			character_score_counts[c] = (character_score_counts[c][0]+ output[i], character_score_counts[c][1]+1)
	for key, value in character_score_counts.items():
		character_score_avarages[key] = value[0]/value[1]
	print (character_score_avarages)
	return character_score_avarages


# def analyze_book_weighted_split(model_predict_function, splitter, book, characters)
def fake_model(list_of_lists):
	# print (list_of_lists)
	return [0]*len(list_of_lists)

books_file_train_names = ["ashputtel","ThePhantomTollbooth", "beauty_beast"]
books_file_test_names = ["cinderella","rumplestiltskin", "blue_beard"]
RNN = rnn.neuralNet()
RNN.makeCRNN()
hist = rnn.train('../project/imdb_csv/imdb_train.csv', '../project/imdb_csv/imdb_test.csv',bigMem=False)

for book in books_file_train_names:
	book_file = "../books/"+book+".txt"
	char_file = "../books/"+ book+".csv"
	label_dict, chars = read_char_lines(char_file)
	# char_avarages = analyze_book()
	print (analyze_book(rnn.predict, splitter, book_file, chars))

# def accuracy(character_score_averages, character_scores_true):
# 	accuracy[]
# 	for key in character_scores_averages.keys():


