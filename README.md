# Application of Sentiment Analysis to Labeling Characters as Good or Evil

Sentiment analysis models were trained with the objective of transferring this learning onto determining if characters in books are good or evil. Typically, sentiment analysis aims to test on a set-apart subset of the training set, however, with a small, hand-labeled training set, this approach was infeasible. Instead, we employed common models to learn sentiment on the training database, then developed code to label the sentiment of characters in books based on the sentiment of sentences in which they appeared. We screened models and datasets to determine which model/training dataset combination generalized most accurately to the other sets, then applied the resulting models to characters in books. In spite of trying several different models and character-labeling methods, none produced accuracies above 0.50.


