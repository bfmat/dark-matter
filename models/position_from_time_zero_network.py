"""A fully connected neural network architecture intended to process time zero information and predict positions"""
# Created by Brendon Matusch, July 2018

from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Model, Sequential
from keras.regularizers import l2


def create_model() -> Model:
    """Create and return a new instance of the fully connected network for processing time zero information"""
    # Create a neural network model that includes several dense layers with hyperbolic tangent activations, L2 regularization, and batch normalization
    regularizer = l2(0)
    dropout = 0
    activation = 'tanh'
    model = Sequential([
        InputLayer(input_shape=(6,)),
        BatchNormalization(),
        Dense(24, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(12, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(6, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(3, kernel_regularizer=regularizer)
    ])
    # Output a summary of the model's architecture
    print(model.summary())
    # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    # Return the untrained model
    return model
