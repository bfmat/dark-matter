"""A dense neural network intended for processing of audio recordings in the frequency domain, separated into various high-resolution time and frequency bands"""
# Created by Brendon Matusch, July 2018

from keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from keras.models import Model
from keras.regularizers import l2


def create_model() -> Model:
    """Create and return a new instance of the high-resolution frequency network"""
    # Create a neural network composed of dense layers with dropout and L2 regularization, using hyperbolic tangent activations
    activation = 'tanh'
    regularizer = l2(0)
    dropout = 0
    # Create two inputs, one for the audio data and one for the position, and concatenate them together
    inputs = Input((40,))
    x = BatchNormalization()(inputs)
    x = Dense(12, activation=activation, kernel_regularizer=regularizer)(x)
    x = Dropout(dropout)(x)
    x = Dense(8, activation=activation, kernel_regularizer=regularizer)(x)
    x = Dropout(dropout)(x)
    output = Dense(1, kernel_regularizer=regularizer)(x)
    model = Model(inputs=inputs, outputs=output)
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
