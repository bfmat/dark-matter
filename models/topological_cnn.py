"""Code related to a convolutional neural network that can convolve over an arbitrary surface topology"""
# Created by Brendon Matusch, August 2018

import copy
from typing import Dict, List, Optional, Union

from keras.layers import concatenate, Dense, Input
from keras.models import Model
import numpy as np

from data_processing.surface_topology import SurfaceTopologyNode, SurfaceTopologySet
from training.pulse_count_train import evaluate_predictions


class TopologicalCNN:
    """A convolutional neural network that can convolve over an arbitrary surface topology; it contains the training data and can fit without any further information"""

    def __init__(self, surface_topology_set: SurfaceTopologySet, convolutional_layers: Dict[str, Union[int, str]], remaining_model: Model, optimizer: str, loss: str, epochs: int, validation_size: int, class_weight: Dict[int, float]) -> None:
        """Create and train a CNN corresponding to a specified set, with layers and training arguments provided"""
        # Add single-element input tensor placeholders to each of the nodes, so graph construction can begin
        for node in surface_topology_set.nodes:
            node.tensor = Input(shape=(1,))
        # Get references to all of the input layers for model creation
        input_layers = [node.tensor for node in surface_topology_set.nodes]
        # Make a changing variable to hold the nodes produced by each layer
        layer_nodes = surface_topology_set.nodes
        # Create a convolutional layer for each of the layer parameter dictionaries provided
        for convolutional_layer in convolutional_layers:
            # Use the nodes from the last layer
            # Also, apply the contents of the dictionary as keyword arguments
            layer_nodes = self.convolve_surface_topology(layer_nodes, **convolutional_layer)
        # Concatenate all the tensors from the last convolutional layer together
        combined_tensor = concatenate([node.tensor for node in layer_nodes])
        # Apply the provided Keras model to the combined tensor to get a final output
        output = remaining_model(combined_tensor)
        # Create a model that encompasses the convolutional layers and the remaining model, leading from all of the input layers to the output layer
        model = Model(inputs=input_layers, outputs=output)
        # Compile the model with the provided optimizer and loss, printing accuracy
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # Convert the inputs to the network to lists
        inputs = [list(node.values) for node in surface_topology_set.nodes]
        # Generate a permutation for the number of inputs to the network, and scramble them
        input_permutation = np.random.permutation(len(inputs[0]))
        inputs = np.array([[input_list[index] for index in input_permutation] for input_list in inputs])
        # Also scramble the ground truths accordingly
        ground_truths = np.array([surface_topology_set.ground_truths[index] for index in input_permutation])
        # Split the inputs and ground truths into training and validation sets
        validation_inputs, training_inputs = np.split(inputs, [validation_size], axis=1)
        validation_ground_truths, training_ground_truths = np.split(ground_truths, [validation_size])
        # Convert the inputs to lists containing a single series of data points for every example
        validation_inputs = [array for array in validation_inputs]
        training_inputs = [array for array in training_inputs]
        # Create a list to hold the numbers of (false and true) (positives and negatives) for each training run
        performance_statistics = []
        # Iterate for a certain number of epochs
        for epoch in range(epochs):
            # Print out the epoch number (the fit function does not)
            print('Epoch', epoch)
            # Train the model for a single epoch
            model.fit(training_inputs, training_ground_truths, validation_data=(validation_inputs, validation_ground_truths), class_weight=class_weight)
            # Run predictions on the validation set with the trained model, removing the single-element second axis
            validation_predictions = model.predict(validation_inputs)[:, 0]
            # Evaluate the network's predictions and add the statistics to the list, only if we are in the last few epochs (we don't care about the other ones, it is still learning then)
            if epoch >= epochs - 10:
                performance_statistics.append(evaluate_predictions(validation_ground_truths, validation_predictions, None, epoch, set_name='validation'))

    @classmethod
    def convolve_surface_topology(cls, surface_topology_nodes: List[SurfaceTopologyNode], kernel_radius: int, filters: int, activation: str, regularizer) -> List[SurfaceTopologyNode]:
        """Create a partial convolutional neural network graph, defining the operations for a single layer, and producing a new topology with the resulting graph"""
        # Create a single shared dense layer that outputs the number of filters
        filters_layer = Dense(filters, activation=activation, kernel_regularizer=regularizer)
        # Create an empty list to add modified nodes to
        modified_nodes = []
        # Iterate over each of the original nodes, creating graphs with them
        for node in surface_topology_nodes:
            # Attempt to form a kernel with this node and the provided radius
            kernel = cls.form_kernel(node, surface_topology_nodes, kernel_radius)
            # If the kernel is None, this node is near an edge; skip to the next iteration
            if kernel is None:
                continue
            # Otherwise, get the tensors corresponding to each of the nodes in this kernel
            tensors = [node.tensor for node in kernel]
            # Concatenate them together so they can be put into a dense layer (equivalent to a CNN filter)
            combined_tensor = concatenate(tensors)
            # Create a shallow copy of the node that can have the dense layer assigned as its tensor
            node_copy = copy.copy(node)
            # Run the dense layer on the combined tensors, placing it in the new node
            node_copy.tensor = filters_layer(combined_tensor)
            # Add the node to the new surface topology set
            modified_nodes.append(node_copy)
        # Return the list with newly defined tensors
        return modified_nodes

    @staticmethod
    def form_kernel(node: SurfaceTopologyNode, searchable_nodes: List[SurfaceTopologyNode], radius: int) -> Optional[List[SurfaceTopologyNode]]:
        """Given a specific node and a kernel radius, return a list of nodes contained within a kernel around that node (or None if an edge was hit and a kernel could not be formed)"""
        # Create a list to add the nodes to
        nodes = []

        def traverse_node_tree(search_node: SurfaceTopologyNode, depth: int) -> bool:
            """A recursive function to search outward from a specific node, adding more nodes to the list; return whether or not the search succeeded"""
            # If the search node is None, the search has hit an edge and needs to fail immediately
            if search_node is None:
                return False
            # Add the node to the list
            nodes.append(search_node)
            # If the depth is 0, also return with success (but the node should still be added to the list)
            if depth == 0:
                return True
            # Otherwise, get a list of nodes connected to the current node
            # Get the identifiers present in the list of searchable nodes, which is required for the search
            searchable_identifiers = [searchable_node.identifier for searchable_node in searchable_nodes]
            # Create a list to add the connected nodes to
            connected_nodes = []
            # Iterate over the list of connected node identifiers
            for connection in search_node.connections:
                # If the connection is not present in the list of searchable nodes, add None to the list of nodes
                if connection not in searchable_identifiers:
                    connected_nodes.append(None)
                # Otherwise, search for the relevant node in the list of search nodes and add it
                else:
                    connected_nodes.append([possible_node for possible_node in searchable_nodes if possible_node.identifier == connection][0])
            # Iterate over the possibly modified list of connected nodes and traverse the trees corresponding to them as well
            for connected_node in connected_nodes:
                # Reduce the depth by 1 so the search is not endless
                success = traverse_node_tree(connected_node, depth - 1)
                # If any traversal of a connected node fails, so does this one
                if not success:
                    return False
            # If everything has succeeded, return with success
            return True

        # Traverse the tree with the provided radius
        success = traverse_node_tree(node, radius)
        # If the search failed, return nothing
        if not success:
            return None
        # Otherwise, return the resulting list of nodes
        return nodes
