from keras import layers
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.models import Model
from keras import regularizers


def NNModel(input_shape, layers, nodes, dropout_rate, l2reg):
    """
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

    X = Dropout(dropout_rate)(X_input)
    X = Dense(nodes, input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(l2reg))(X)

    # Variable number of layers used during arch sensitivity testing
    for i in range(layers):
        X = Dropout(dropout_rate)(X)
        X = Dense(2 * nodes, activation='relu', kernel_regularizer=regularizers.l2(l2reg))(X)

    predictions = Dense(1, activation='sigmoid')(X)

    # Create model
    model = Model(inputs=X_input, outputs=predictions, name='NNModel')

    return model