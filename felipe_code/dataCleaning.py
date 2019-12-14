import os
import numpy as np
import pandas as pd

def extractRatingFromFileName(fileName):
    s1 = fileName.split(".")
    s2 = s1[0].split("_")
    return s2[1]

def readFolder(folderPath):
    print("Reading folder " + folderPath)
    counter = 1

    comments = []
    IMDBRating = []
    binaryRating = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(folderPath):
        for file in f:
            if '.txt' in file:
                if counter % 1000 == 0:
                    print("Iteration: " + str(counter))
                filePath = inputPath + "/" + file
                fileObj = open(filePath, 'r', encoding="ISO-8859-1")
                origRating = extractRatingFromFileName(file)
                IMDBRating.append(origRating)
                data = fileObj.read()
                comments.append(data)
                binaryRating.append(1*(int(origRating) > 6))
                counter += 1

    ds = list(zip(comments, IMDBRating, binaryRating))
    return ds


# Positive data points
outputPath = "Dataset"
outputFileName = "imdb_training.csv"

inputPath = "aclImdb/aclImdb/train/pos"
training_pos = readFolder(inputPath)
inputPath = "aclImdb/aclImdb/train/neg"
training_neg = readFolder(inputPath)
inputPath = "aclImdb/aclImdb/test/pos"
test_pos = readFolder(inputPath)
inputPath = "aclImdb/aclImdb/test/neg"
test_neg = readFolder(inputPath)

dataset = training_pos + training_neg + test_pos + test_neg

np.random.shuffle(dataset)

df = pd.DataFrame(data=dataset, columns=['Comments', 'IMDB Rating', 'Binary Rating'])
df.to_csv(outputPath+outputFileName, index=False, header=True)



