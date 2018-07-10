"""A dense neural network intended for processing of audio recordings in the frequency domain, separated into various high-resolution time and frequency bands"""
# Created by Brendon Matusch, July 2018

from keras.layers import Dense, Dropout, InputLayer
from keras.models import Model, Sequential
from keras.regularizers import l2


def create_model() -> Model:
    """Create and return a new instance of the high-resolution frequency network"""
    # Create a neural network composed of dense layers with dropout regularization, using rectified linear activations
    activation = 'relu'
    model = Sequential()
    model.add(InputLayer(input_shape=(32_768,)))
    # Use dropout even on the input layer, to avoid over-reliance on one particular frequency band
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation=activation))
    model.add(Dropout(0.5))
    # Use a sigmoid activation on the last layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    # Output a summary of the model's architecture
    print(model.summary())
    # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )
    # Return the untrained model
    return model
