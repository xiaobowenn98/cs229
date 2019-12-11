import numpy as np 

class Embedding:
    # Class container for embedding
    # embedding: the kind of embedding you want to do
    # maxLength: maximum length of the sentences
    def __init__(self, embedding = 'glove', maxLength = 100):
        self.embedding = embedding
        self.maxLength = maxLength
        self.vectors = self.loadEmbedding()

    def loadEmbedding(self):
        # Loads specified embedding into a dictionary
        # returns a dictionary where the key is a word and the value is the associated vector
        if self.embedding == 'glove':
            path = '../../project/glove.twitter.27B.200d.txt'
            self.dim = 200
        elif self.embedding == 'elmo':
            print("Not implemented yet")
        else:
            print("Unsupported embedding")
        vectors = {}
        print("Loading glove data")
        with open(path, 'r', encoding='ISO-8859-1') as fl:
            for line in fl:
                split = line.split()
                word = split[0]
                if all([ord(char) < 123 and ord(char) > 31 for char in list(word)]):
                    vectors[word] = np.asarray(split[1:],dtype=np.float64)
        print("Finish loading embedding data")
        return vectors

    def embed(self, text):
        # Embeds a piece of text within the specified vector embedding
        # text: a list of lists of strings (a list of sentences, each of which is a list of words)
        # returns: a matrix of dimension (number of sentences x maxLength x embedding dimension) 
        # this format is consistent with the default data_format for convolutional keras layers
        print("Embedding text")
        matrix = np.zeros((len(text), self.maxLength, self.dim))
        tooLong = 0
        for i in np.arange(len(text)):
            k = 0
            for j in np.arange(len(text[i])):
                vec = self.vectors.get(text[i][j], np.array([-1]))
                if vec.shape[0] != 1:
                    matrix[i, k, :] = vec
                    k += 1
                    if k >= self.maxLength:
                        tooLong += 1
                        break
        if tooLong > 0:
            print(str(tooLong) + " messages were too long to be fully embedded; consider increasing embedding length or splitting sentences")
        return matrix

# Sample usage

#def main():
#    ed = Embedding(embedding = 'glove', maxLength=3)
#    text = []
#    text.append("this is a sentence".split())
#    text.append("wheeeee look at me go".split())
#    print(text)
#    print(ed.embed(text))
#
#if __name__ == "__main__":
#    main()


