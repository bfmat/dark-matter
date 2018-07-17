"""A dense neural network intended for processing of audio recordings in the frequency domain, separated into various high-resolution time and frequency bands"""
# Created by Brendon Matusch, July 2018

from keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from keras.models import Model


def create_model() -> Model:
    """Create and return a new instance of the high-resolution frequency network"""
    # Create a neural network composed of dense layers with dropout regularization, using rectified linear activations
    activation = 'relu'
    # Create two inputs, one for the audio data and one for the position, and concatenate them together
    audio_input = Input((48,))
    position_input = Input((3,))
    x = concatenate([audio_input, position_input])
    x = BatchNormalization()(x)
    x = Dense(12, activation=activation)(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation=activation)(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    # Create a model with both inputs
    model = Model(inputs=[audio_input, position_input], outputs=output)
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
