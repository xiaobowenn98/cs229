import pandas as pd
import numpy as np
from NNModel import *
from naiveBayes import *
from keras.optimizers import *
from keras.models import load_model


def readData(filename):
    data = pd.read_csv(filename, header=None, names=['Binary Rating', 'Comments'], encoding='ISO-8859-1', nrows=100000)
    X_data = data['Comments']
    Y_data = data['Binary Rating']
    return X_data, Y_data

def extractWordFeatures(x):
    """
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    words = x.split()
    count = {}

    for w in words:
        if w not in count:
            count[w] = 1
        else:
            count[w] += 1

    return count


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


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.
    Args: messages: A list of strings containing SMS messages
    Returns: A python dict mapping words to integers.
    """

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

    wordsDict = {k: v for k, v in wordsDict.items() if v > 4}

    # Create indices
    wordsDict = sorted(list(wordsDict))
    numbers = list(range(0, len(wordsDict)))
    wordsDict = dict(zip(wordsDict, numbers))

    return wordsDict


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

def splitData(X_data, Y_data, discarded=0.0, trainP=0.8, testP=0.1):
    Train_percent = trainP * (1 - discarded)
    Dev_percent = (1-trainP-testP) * (1 - discarded)
    Test_percent = testP * (1 - discarded)

    np.random.seed(0)
    m = len(X_data)
    temp_list = list(range(m))
    np.random.shuffle(temp_list)
    Train_list = temp_list[0:int(Train_percent * m)]
    Dev_list = temp_list[int(Train_percent * m):int(Train_percent * m) + int(Dev_percent * m)]
    Test_list = temp_list[int(Train_percent * m) + int(Dev_percent * m):
                          int(Train_percent * m) + int(Dev_percent * m) + int(Test_percent * m)]

    X_train_temp = X_data[Train_list]
    Y_train = Y_data[Train_list]
    X_test_temp = X_data[Test_list]
    Y_test = Y_data[Test_list]

    return (X_train_temp, Y_train, X_test_temp, Y_test)

def trainNN(database, modelFileName):

    X_data, Y_data = readData(database)
    (X_train_temp, Y_train, X_test_temp, Y_test) = splitData(X_data, Y_data, discarded=0.5)
    wordDictionary = create_dictionary(X_data)

    print("Dictionary size: " + str(len(wordDictionary)))

    X_train = transform_text(X_train_temp, wordDictionary)
    X_test = transform_text(X_test_temp, wordDictionary)

    print("Size X_train: " + str(np.shape(X_train)))
    print("Size X_test " + str(np.shape(X_test)))

    # Model Parameters
    nodes = 1000
    layers = 3
    l2reg = 0.0
    dropout = 0.5
    loss = "binary_crossentropy"
    epochs = 15
    batch = 1024

    SAModel = NNModel(X_train[0].shape, layers=layers, nodes=nodes, dropout_rate=dropout, l2reg=l2reg)
    SAModel.summary()

    optimizer = Adam()
    SAModel.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])

    history = SAModel.fit(X_train, Y_train, epochs=epochs, shuffle=False, batch_size=batch)

    train_metrics = SAModel.evaluate(X_train, Y_train, verbose=1)
    test_metrics = SAModel.evaluate(X_test, Y_test, verbose=1)

    SAModel.save(modelFileName)

    print("Train Metrics: " + str(train_metrics))
    print("Test Metrics: " + str(test_metrics))


def runNN(trainReq, trainDatabase, testDatabase, modelFileName):

    if trainReq:
        trainNN(trainDatabase, modelFileName)
    else:
        SAModel = load_model(modelFileName)

        SAModel.summary()

        # Get Dictionary for training dataset
        X_data, Y_data = readData(trainDatabase)
        wordDictionary = create_dictionary(X_data)

        # Get testing dataset
        X_data, Y_data = readData(testDatabase)
        (_, _, X_test_temp, Y_test) = splitData(X_data, Y_data)

        X_test = transform_text(X_test_temp, wordDictionary)

        test_metrics = SAModel.evaluate(X_test, Y_test, verbose=1)

        print("Test Metrics: " + str(test_metrics))

def runNaiveBayes(trainDatabase, testDatabase):

    # Fitting
    print("Loading Train Data")
    X_data, Y_data = readData(trainDatabase)
    (X_train_temp, Y_train, X_test_temp, Y_test) = splitData(X_data, Y_data, discarded=0.0)
    wordDictionary = create_dictionary(X_data)

    print("Dictionary size: " + str(len(wordDictionary)))

    X_train = transform_text(X_train_temp, wordDictionary)
    X_test = transform_text(X_test_temp, wordDictionary)

    print("Size X_train: " + str(np.shape(X_train)))
    print("Size X_test " + str(np.shape(X_test)))
    print("Fitting model")
    naive_bayes_model = fit_naive_bayes_model(X_train, Y_train)

    # Prediction

    if trainDatabase != testDatabase:
        print("Loading Test Data")
        X_data, Y_data = readData(testDatabase)
        (X_train_temp, Y_train, X_test_temp, Y_test) = splitData(X_data, Y_data, discarded=0)
        X_test = transform_text(X_test_temp, wordDictionary)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, X_test)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == Y_test)
    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

def main():
    trainDatabase = "./Datasets/AmazonBooks_train.csv"
    testDatabase = "./Datasets/AmazonBooks_train.csv"   # "./Datasets/twitter_train.csv", "./Datasets/IMDB_train.csv"
    modelFileName = "NN_with_Amazon.h5"
    trainReq = False

    # runNN(trainReq, trainDatabase, testDatabase, modelFileName)
    runNaiveBayes(trainDatabase, testDatabase)

if __name__ == "__main__":
    main()