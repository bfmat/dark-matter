"""A fully connected neural network architecture intended to process low-resolution banded frequency domain audio information"""
# Created by Brendon Matusch, July 2018

from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Model, Sequential


def create_model() -> Model:
    """Create and return a new instance of the fully connected network for banded frequency domain information"""
    # Create a neural network model that includes several dense layers with hyperbolic tangent activations, dropout, and batch normalization
    activation = 'tanh'
    model = Sequential([
        InputLayer(input_shape=(72,)),
        BatchNormalization(),
        Dense(12, activation=activation),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
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
