"""A very deep 1-dimensional fully convolutional network intended for processing of raw audio waveforms and inspired by the M34 architecture"""
# Created by Brendon Matusch, July 2018

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense, add
from keras.models import Model, Sequential
from keras.regularizers import l2


def create_model() -> Model:
    """Create and return a new instance of the very deep convolutional network"""
    # Create a one-dimensional convolutional neural network model with rectified linear activations, using the Keras functional API
    # It should take both microphone channels and an entire clip of audio
    # Use residual blocks in which the input and output of convolutional layer are added together, to help prevent information loss in a very deep network
    activation = 'relu'
    padding = 'same'
    regularizer = l2(0)
    inputs = Input((100_000, 2))
    x = BatchNormalization()(inputs)
    # Skip the residual block for this layer since it has a stride greater than 1
    x = Conv1D(
        filters=48,
        kernel_size=80,
        strides=4,
        activation=activation,
        kernel_regularizer=regularizer,
        padding=padding
    )(x)
    x = MaxPooling1D(6)(x)
    x = BatchNormalization()(x)
    for _ in range(3):
        initial_conv_in = x
        for _ in range(2):
            x = Conv1D(
                filters=48,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=regularizer,
                padding=padding
            )(x)
            x = BatchNormalization()(x)
        x = add([x, initial_conv_in])
    x = MaxPooling1D(6)(x)
    x = BatchNormalization()(x)
    for i in range(4):
        initial_conv_in = x
        for _ in range(2):
            x = Conv1D(
                filters=96,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=regularizer,
                padding=padding
            )(x)
            x = BatchNormalization()(x)
        # Skip the residual block if this is the first iteration (meaning the number of filters of the input versus the output is different)
        if i != 0:
            x = add([x, initial_conv_in])
    x = MaxPooling1D(6)(x)
    x = BatchNormalization()(x)
    for i in range(6):
        initial_conv_in = x
        for _ in range(2):
            x = Conv1D(
                filters=192,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=regularizer,
                padding=padding
            )(x)
            x = BatchNormalization()(x)
        if i != 0:
            x = add([x, initial_conv_in])
    x = MaxPooling1D(6)(x)
    x = BatchNormalization()(x)
    for i in range(3):
        initial_conv_in = x
        for _ in range(2):
            x = Conv1D(
                filters=384,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=regularizer,
                padding=padding
            )(x)
            x = BatchNormalization()(x)
        if i != 0:
            x = add([x, initial_conv_in])
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=regularizer)(x)
    model = Model(inputs=inputs, outputs=outputs)
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
