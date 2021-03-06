import numpy as np 
import csv
from keras.models import Sequential
#from keras.layers.convolutional import *
#from keras.layers.recurrent import *
#from keras.layers.pooling import *
from keras.layers import *
#from keras.callbacks.callbacks import *
from keras.utils import Sequence
import embed
import itertools
import tensorflow as tf

def get_words(message):
        message = message.replace("."," ").replace(",", " ")
        message = message.replace("!"," ! ").replace("?"," ? ")
        message = message.replace('"','')
        return message.lower().split()

def load_dataset(tsv_path):

        messages = []
        labels = []

        with open(tsv_path, 'r', newline='', encoding='ISO-8859-1') as tsv_file:
            reader = csv.reader(tsv_file, delimiter=',')
            for label, message in reader:
                messages.append(get_words(message))
                labels.append(label[-1])

        return messages, np.asarray(labels,dtype=np.float32)

class BatchData(Sequence):
    def __init__(self, batchSize, trainPath, ed):
        self.trainPath = trainPath
        self.batchSize = batchSize
        with open(self.trainPath, 'r', encoding='ISO-8859-1') as dataFile:
            for self.trainSize, l in enumerate(dataFile):
                pass
        self.limits = np.concatenate((np.arange(0,self.trainSize,self.batchSize),np.array([self.trainSize])))
        self.ed = ed

    def __len__(self):
        return (np.ceil(1.0 * self.trainSize / self.batchSize)).astype(np.int)

    def __getitem__(self, i):
        sentences, labels = self.loadDataset(self.limits[i],self.limits[i+1])
        messageEmbed = self.ed.embed(sentences)
        return messageEmbed, labels

    def loadDataset(self, start, end):
        labels = []
        sentences = []
        with open(self.trainPath, 'r', encoding='ISO-8859-1') as dataFile:
            for line in itertools.islice(dataFile, start, end):
                line = line.split(',')
                label = line[0]
                sentence = ''.join(line[1:])
                labels.append(label[-1])
                sentences.append(get_words(sentence))
        
        return sentences, np.asarray(labels,dtype=np.int8)

    def get_words(self, message):
        message = message.replace("."," ").replace(",", " ")
        message = message.replace("!"," ! ").replace("?"," ? ")
        message = message.replace('"','')
        return message.lower().split()

class neuralNet:
    def __init__(self, embedding = 'glove', maxLength = 100):
        self.ed = embed.Embedding(embedding, maxLength)
        self.maxLength = maxLength
        
    def makeCRNN(self, kernel_size = 5, filters = 64, pool_size = 4, lstm_output_size = 70):
        self.model = Sequential()
        self.model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1, 
                 input_shape=(self.maxLength, self.ed.dim)))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(LSTM(lstm_output_size))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(self, train_path, test_path, epochs = 5, batchSize = 10000, bigMem = False, trainLog = 'training.log', saveName = "model.h5"):
        #csv_logger = CSVLogger(trainLog)
        testMessages, testLabels = load_dataset(test_path)
        testMessages = self.ed.embed(testMessages)
        if bigMem == False:
            messages, labels = load_dataset(train_path)
            messages = self.ed.embed(messages)
            hist = self.model.fit(messages, labels, epochs=epochs, batch_size=batchSize, validation_data=(testMessages, testLabels))#,callbacks=[csv_logger])
        if bigMem == True:
            trainData = BatchData(batchSize, train_path, self.ed)
            self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
            hist = self.model.fit_generator(generator = trainData, steps_per_epoch = int(np.ceil(trainData.trainSize / trainData.batchSize)), epochs=epochs, validation_data = (testMessages, testLabels), validation_steps = 1)#, use_multiprocessing = True, workers = 4)#, callbacks = [csv_logger]) 
        self.model.save(saveName)
        return hist

    def predict(self, validData):
        # validData: a list of lists of strings (a list of sentences, each of which is a list of words)
        # returns an array of predictions in [0,1]
        validEmbed = self.ed.embed(validData)
        return self.model.predict(validEmbed)

    def predict_from_model(self, model_file, validData):
        model_new = tf.keras.models.load_model(model_file)
        validEmbed = self.ed.embed(validData)
        return model_new.predict(validEmbed)

    def evaluate(self, validData, validLabels):
        validEmbed = self.ed.embed(validData)
        return self.model.evaluate(validEmbed, validLabels)
                
def doPredict(valFile, rnn):
    sentences, labels = load_dataset(valFile)
    return rnn.evaluate(sentences, labels)[1]

def main():
    #rnn = neuralNet()
    #rnn.makeCRNN()
    #hist = rnn.train('IMDB_train_1000.csv',test_path='IMDB_test.csv',epochs=15,saveName='CRNN_IMDb_1000.h5',bigMem=False)
    #for test in ['IMDB_test.csv', 'AmazonBooks_test.csv', 'twitter_test.csv']:
    #    with open('output_CRNN_1000_glove.txt','a') as f:
    #        print("Test: " + test + " Accuracy: " + str(doPredict(test, rnn)), file = f)
    for train, mem, epoch, name, maxLen in zip(['IMDB_train.csv', 'AmazonBooks_train.csv', 'twitter_train.csv'], [False, True, True], [40, 1, 1], ['IMDBv2.h5','AmazonBooksv2.h5','twitterv2.h5'], [200, 200, 100]):
        rnn = neuralNet(maxLength=maxLen)
        rnn.makeCRNN()
        hist = rnn.train(train, 'IMDB_test.csv',bigMem=mem, epochs = epoch, saveName=name)
        for test in ['IMDB_test.csv', 'AmazonBooks_test.csv', 'twitter_test.csv']:
            with open('outputv2.txt','a') as f:
                print("Train: " + train + " Test: " + test + " Accuracy: " + str(doPredict(test, rnn)), file = f)
    #ed = embed.Embedding()
    #bd = BatchData(10000, 'IMDB_train.csv', ed)
    #print(bd.trainSize)
    #print(bd.limits)
    #print(np.floor(bd.trainSize / bd.batchSize))


if __name__ == "__main__":
    main()
