import numpy as np 
import csv
from keras.models import Sequential
#from keras.layers.convolutional import *
#from keras.layers.recurrent import *
#from keras.layers.pooling import *
from keras.layers import *
from keras.layers import LSTM
#from keras.callbacks.callbacks import *
from keras.utils import Sequence
import embed
import itertools
import rnn



class CNN(rnn.neuralNet):
    def __init__(self, embedding = 'glove', maxLength = 100):
        self.ed = embed.Embedding(embedding, maxLength)

    def makeCNN(self, maxLength = 100, kernel_size = 5, filters = 64, pool_size = 4):
        self.model = Sequential()
        self.model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1, 
                 input_shape=(maxLength, self.ed.dim)))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

class LSTM(rnn.neuralNet):
    def __init__(self, embedding = 'glove', maxLength = 100):
        self.ed = embed.Embedding(embedding, maxLength)

    def makeLSTM(self, maxLength = 100, lstm_output_size = 70):
        self.model = Sequential()
        self.model.add(LSTM(70))
        # self.modell.add(Dropout(.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])    

def doPredict(valFile, Rnn):
    sentences, labels = rnn.load_dataset(valFile)
    return Rnn.evaluate(sentences, labels)[1]

def main():
    cnn = CNN()
    cnn.makeCNN()
    hist = cnn.train('twitter_train.csv',test_path='twitter_test.csv',epochs=1,saveName='CNN_t.h5',bigMem=True)
    for test in ['IMDB_test.csv', 'AmazonBooks_test.csv', 'twitter_test.csv']:
        with open('output_CNN_t.txt','a') as f:
            print("Test: " + test + " Accuracy: " + str(doPredict(test, cnn)), file = f)
    #cnn = CNN()
    #cnn.makeCNN()
    #cnn.train('IMDB_train.csv','IMDB_test.csv',epochs = 15, saveName = 'cnn.h5')
    #for test in ['IMDB_test.csv', 'AmazonBooks_test.csv', 'twitter_test.csv']:
    #    with open('output_CNN.txt','a') as f:
    #        print("Train: IMDB" + " Test: " + test + " Accuracy: " + str(doPredict(test, cnn)), file = f)


if __name__ == "__main__":
    main()