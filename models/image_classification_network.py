"""A 2-dimensional convolutional neural network intended for processing image windows containing bubbles"""
# Created by Brendon Matusch, July 2018

from keras.layers import Conv2D, Flatten, Dropout, InputLayer, BatchNormalization, Dense
from keras.models import Model, Sequential
from keras.regularizers import l2

from data_processing.bubble_data_point import WINDOW_SIDE_LENGTH, START_IMAGE_INDEX, END_IMAGE_INDEX


def create_model() -> Model:
    """Create and return a new instance of the image classification convolutional network"""
    # Calculate the number of images there are stacked along the channels axis
    channels = END_IMAGE_INDEX - START_IMAGE_INDEX
    # Create a network with hyperbolic tangent activations, dropout regularization on the fully connected layers, and L2 regularization everywhere
    activation = 'tanh'
    dropout = 0.25
    regularizer = l2(0.003)
    model = Sequential([
        InputLayer(input_shape=(WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, channels)),
        BatchNormalization(),
        Conv2D(filters=16, kernel_size=4, strides=2, activation=activation, kernel_regularizer=regularizer),
        Conv2D(filters=32, kernel_size=3, strides=2, activation=activation, kernel_regularizer=regularizer),
        Conv2D(filters=32, kernel_size=3, strides=2, activation=activation, kernel_regularizer=regularizer),
        Conv2D(filters=64, kernel_size=2, strides=1, activation=activation, kernel_regularizer=regularizer),
        Flatten(),
        Dense(64, activation=activation, kernel_regularizer=regularizer),
        Dropout(dropout),
        Dense(16, activation=activation, kernel_regularizer=regularizer),
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
