def analyze_book(model_predict_function, splitter, book, characters):
	lines, character_list = splitter(book, characters)
	clean_input = [clean_and_process(line) for line in lines]
	output = model_predict_function(clean_input)
	character_score_counts = {}
	character_score_averages = {}
	for i in range(len()):
		for c in character_list[i]:
			if not c in character_scores:
				character_score_counts[c] = (0,0)
			character_score_counts[c] = [character_scores[c][0]+ output[i], character_scores[c][1]+1]
	for key, value in character_score_counts.items():
		character_score_avarages[key] = value[0]/value[1]
	return character_score_averages

# def analyze_book_weighted_split(model_predict_function, splitter, book, characters)
def accuracy(character_score_averages, character_scores_true):
	total = len 
	for key in character_scores_averages.keys():

