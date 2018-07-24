"""A 2-dimensional convolutional neural network intended for processing image windows containing bubbles"""
# Created by Brendon Matusch, July 2018

from keras.layers import Conv2D, Flatten, Dropout, InputLayer, BatchNormalization, Dense
from keras.models import Model, Sequential
from keras.regularizers import l2

from data_processing.bubble_data_point import WINDOW_SIDE_LENGTH


def create_model() -> Model:
    """Create and return a new instance of the image classification convolutional network"""
    # Create a network with hyperbolic tangent activations and dropout regularization on the fully connected layers
    activation = 'tanh'
    dropout = 0.5
    model = Sequential([
        InputLayer(input_shape=(WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, 1)),
        BatchNormalization(),
        Conv2D(filters=16, kernel_size=4, strides=2, activation=activation),
        Conv2D(filters=32, kernel_size=3, strides=2, activation=activation),
        Conv2D(filters=32, kernel_size=3, strides=2, activation=activation),
        Conv2D(filters=64, kernel_size=2, strides=1, activation=activation),
        Flatten(),
        Dense(64, activation=activation),
        Dropout(dropout),
        Dense(16, activation=activation),
        Dropout(dropout),
        Dense(1)
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
