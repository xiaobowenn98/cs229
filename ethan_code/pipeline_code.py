import splitter
from splitter import *
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.append( '../')
import rnn
import embed
import tensorflow as tf
# import math
def analyze_book(model_predict_function, splitter, book, characters):
	lines, character_list = splitter(book, characters)
	# clean_input = [clean_and_process(line) for line in lines]
	output = model_predict_function("cnn.h5",lines)


	character_score_counts = {}
	character_score_avarages = {}
	lines2 = []
	output2 = []
	for i in range(len(output)):
		for c in character_list[i]:
			# print (c)
			if not c in character_score_counts:
				character_score_counts[c] = [0,0]
			lines2.append(" ".join(lines[i]))
			output2.append(output[i])
			character_score_counts[c] = [character_score_counts[c][0]+ output[i][0], character_score_counts[c][1]+1]
	for key, value in character_score_counts.items():
		character_score_avarages[key] = value[0]/value[1]
	df = pd.DataFrame(lines2, output2)
	df.to_csv(book[-6::-3]+"book.csv")
	return character_score_avarages


# def analyze_book_weighted_split(model_predict_function, splitter, book, characters)
def fake_model(file_name, list_of_lists):
	return [0*len(list_of_lists)]

books_file_train_names = ["ashputtel","ThePhantomTollbooth", "beauty_beast", "the_giving_tree","count_of_monte_cristo", "hamlet_new" ]
# books_file_train_names = ["ThePhantomTollbooth"]
books_file_test_names = ["cinderella","rumplestiltskin", "blue_beard"]
# model_new = tf.keras.models.load_model("model.h5")

r_model = rnn.neuralNet()
# r_model.makeCRNN()
# hist = r_model.train('../../project/amazon_csv/AmazonBooks_train.csv', '../../project/amazon_csv/AmazonBooks_test.csv',bigMem=True)
# new_model = load_model("model.h5")
total_char = 0.0
correct = 0.0
correct2= 0.0
for book in books_file_train_names:
	book_file = "../books/"+book+".txt"
	char_file = "../books/"+ book+".csv"
	label_dict, chars = read_char_lines(char_file)
	# char_avarages = analyze_book()
	# char_score_dict= analyze_book(fake_model, splitter_2, book_file, chars)
	# char_score_dict2 = analyze_book(r_model.predict, splitter, book_file, chars)
	char_score_dict = analyze_book(r_model.predict_from_model, splitter, book_file, chars)

	for key in char_score_dict.keys():
		if int(round(char_score_dict[key])) == int(label_dict[key]):
			print ("correctly labeled: " + key +"with value: " +str(char_score_dict[key]))
			correct +=1
		else:
			print ("incorrectly labeled: " + key +"with value: " +str(char_score_dict[key]))
		# if int(round(char_score_dict2[key][0])) == int(label_dict[key]):
			# print ("correctly labeled: " + key +"with value: " +str(char_score_dict[key]))
			# correct2 +=1
		# if char_score_dict2[key][0]!= character_score_dict[key]:
			# print ("models conflict")
		total_char +=1
print (correct/total_char)

# def accuracy(character_score_averages, character_scores_true):
# 	accuracy[]
# 	for key in character_scores_averages.keys():


