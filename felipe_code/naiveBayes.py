import collections
import pandas as pd
import numpy as np

def readData(filename):
    data = pd.read_csv(filename, header=0, encoding='ISO-8859-1')
    X_data = data['Comments']
    Y_data = data['Binary Rating']
    return X_data, Y_data

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    message = message.lower()
    messageList = message.split()
    return messageList
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    wordsDict = dict()

    for m in messages:
        msgWordList = get_words(m)
        # Remove duplicates
        msgWordList = list(set(msgWordList))

        for w in msgWordList:
            if w not in wordsDict:
                wordsDict[w] = 1
            else:
                wordsDict[w] += 1

    wordsDict = {k:v for k,v in wordsDict.items() if v > 4}

    # Create indices
    wordsDict = sorted(list(wordsDict))
    numbers = list(range(0, len(wordsDict)))
    wordsDict = dict(zip(wordsDict, numbers))

    return wordsDict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    textArray = np.zeros((len(messages), len(word_dictionary)))

    for i, m in enumerate(messages):
        msgWordList = get_words(m)
        for w in msgWordList:
            if w in word_dictionary:
                textArray[i, word_dictionary[w]] += 1

    return textArray
    # *** END CODE HERE ***

class NVM:
    def __init__(self, phi_y, phi_k_y1, phi_k_y0):
        self.phi_y = phi_y
        self.phi_k_y1 = phi_k_y1
        self.phi_k_y0 = phi_k_y0

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    vocSize = matrix.shape[1]

    phi_y = sum(labels)/len(labels)
    phi_k_y1 = (1+np.dot(matrix.T, labels)) / (vocSize + np.sum(np.dot(matrix.T, labels)))
    phi_k_y0 = (1+np.dot(matrix.T, 1-labels)) / (vocSize + np.sum(np.dot(matrix.T, 1-labels)))

    model = NVM(phi_y, phi_k_y1, phi_k_y0)
    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    prob1 = np.dot(matrix, np.log(model.phi_k_y1)) + np.log(model.phi_y)
    prob0 = np.dot(matrix, np.log(model.phi_k_y0)) + np.log(1-model.phi_y)
    pred = (prob1 > prob0)

    return pred
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    metric = np.log(model.phi_k_y1/model.phi_k_y0)

    numbers = list(range(0, len(dictionary)))
    tokens = tuple(zip(numbers, metric))
    tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
    tokens = [i[0] for i in tokens[0:5]]

    top5Words = list()

    for i in tokens:
        w = list(dictionary.keys())[list(dictionary.values()).index(i)]
        top5Words.append(w)

    return top5Words

    # *** END CODE HERE ***



def main():
    Discarded_percent = 0.95
    Train_percent = 0.8 * (1 - Discarded_percent)
    Dev_percent = 0.1 * (1 - Discarded_percent)
    Test_percent = 0.1 * (1 - Discarded_percent)

    database = "IMDB_dataset.csv"
    database = "twitter_train.csv"

    X_data, Y_data = readData(database)

    m = len(X_data)
    temp_list = list(range(m))
    np.random.shuffle(temp_list)
    Train_list = temp_list[0:int(Train_percent * m)]
    Dev_list = temp_list[int(Train_percent * m):int(Train_percent * m) + int(Dev_percent * m)]
    Test_list = temp_list[int(Train_percent * m) + int(Dev_percent * m):
                          int(Train_percent * m) + int(Dev_percent * m) + int(Test_percent * m)]

    wordDictionary = create_dictionary(X_data)
    print("Dictionary size: " + str(len(wordDictionary)))
    X_train_temp = X_data[Train_list]
    Y_train = Y_data[Train_list]
    X_test_temp = X_data[Test_list]
    Y_test = Y_data[Test_list]

    X_train = transform_text(X_train_temp, wordDictionary)
    X_test = transform_text(X_test_temp, wordDictionary)

    print("Size X_train: " + str(np.shape(X_train)))
    print("Size X_test " + str(np.shape(X_test)))

    naive_bayes_model = fit_naive_bayes_model(X_train, Y_train)
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, X_test)
    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == Y_test)
    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    # top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)
    # print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

if __name__ == "__main__":
    main()
