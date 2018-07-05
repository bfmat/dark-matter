"""Code related to a semi-supervised learning technique in which a network is trained on a small data set and the most confident previously unlabeled examples are added to the training set on each iteration"""
# Created by Brendon Matusch, July 2018

import copy

from keras.models import Model

from data_processing.bubble_data_point import BubbleDataPoint, load_bubble_audio, RunType
from data_processing.event_data_set import EventDataSet

# The distance from 0 or 1 an example must be to be added to the training set
TRAINING_THRESHOLD_DISTANCE = 0.025


def train_iteratively(model: Model, event_data_set: EventDataSet) -> None:
    """A function that iteratively trains a given model on a given event data set and prints training accuracy"""
    # Combine the training and validation sets into one
    bubbles = event_data_set.training_events + event_data_set.validation_events
    # Create a list for the gradually expanding set of training bubbles, and add the confident bubbles to it initially
    training_bubbles = [
        bubble for bubble in bubbles
        if confident_enough_for_initial_set(bubble)
    ]
    # Iterate for 50 epochs
    for _ in range(50):
        # Create a training data generator with the current list of bubbles
        training_generator_callable, _, _ = event_data_set.arbitrary_alpha_classification_generator(
            load_bubble_audio,
            storage_size=256,
            batch_size=32,
            examples_replaced_per_batch=16,
            custom_training_data=training_bubbles
        )
        training_generator = training_generator_callable()
        # Weight the alpha class so it is worth eight times as much as the neutron class (since there are many fewer alphas)
        class_weight = {0: 1, 1: 8}
        # Train the model for one epoch on the generator
        model.fit_generator(
            training_generator,
            steps_per_epoch=128,
            epochs=1,
            class_weight=class_weight
        )
        # Iterate over the entire list of bubbles, running predictions
        for bubble in bubbles:
            # Load the audio samples for that bubble (there will either be 0 or 1)
            audio_samples = load_bubble_audio(bubble)
            # If there are no audio samples, skip to the next iteration
            if not audio_samples:
                continue
            # Run a prediction on the audio sample using the existing neural network
            prediction = model.predict(audio_samples[0])
            # If the prediction is within a certain threshold distance of either 0 or 1
            if min([prediction, 1 - prediction]) < TRAINING_THRESHOLD_DISTANCE:
                # Copy the bubble and set its run type so that it is in the corresponding ground truth
                bubble_copy = copy.deepcopy(bubble)
                bubble_copy.run_type = RunType.AMERICIUM_BERYLLIUM if bool(round(prediction)) \
                    else RunType.LOW_BACKGROUND
                # Add the modified bubble to the training list
                training_bubbles.append(bubble_copy)


def confident_enough_for_initial_set(bubble: BubbleDataPoint) -> bool:
    """A function that, given a bubble data point, returns whether or not its classification is sufficiently confident for the initial training data set"""
    return False
