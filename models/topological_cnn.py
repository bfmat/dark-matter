"""Code related to a convolutional neural network that can convolve over an arbitrary surface topology"""
# Created by Brendon Matusch, August 2018

import copy
from typing import Dict, List, Optional, Union

from keras.layers import concatenate, Dense, Input
from keras.models import Model

from data_processing.surface_topology import SurfaceTopologyNode, SurfaceTopologySet


class TopologicalCNN:
    """A convolutional neural network that can convolve over an arbitrary surface topology; it contains the training data and can fit without any further information"""

    def __init__(self, surface_topology_set: SurfaceTopologySet, convolutional_layers: Dict[str, Union[int, str]], remaining_model: Model, optimizer: str, loss: str) -> None:
        """Create and train a CNN corresponding to a specified set, with layers and training arguments provided"""
        # Create a full copy of the surface topology set so the original is not modified
        set_copy = copy.deepcopy(surface_topology_set)
        # Add single-element input tensor placeholders to each of the nodes, so graph construction can begin
        for node in set_copy.nodes:
            node.tensor = Input(shape=(1,))
        # Get references to all of the input layers for model creation
        inputs = [node.tensor for node in set_copy.nodes]
        # Make a changing variable to hold the set produced by each layer
        layer_set = set_copy
        # Create a convolutional layer for each of the layer parameter dictionaries provided
        for convolutional_layer in convolutional_layers:
            # Use the set from the last layer
            # Also, apply the contents of the dictionary as keyword arguments
            layer_set = self.convolve_surface_topology(set_copy, **convolutional_layer)
        # Concatenate all the tensors from the last convolutional layer together
        combined_tensor = concatenate([node.tensor for node in layer_set.nodes])
        # Apply the provided Keras model to the combined tensor to get a final output
        output = remaining_model(combined_tensor)
        # Create a model that encompasses the convolutional layers and the remaining model, leading from all of the input layers to the output layer
        model = Model(inputs=inputs, outputs=output)
        # Compile the model with the provided optimizer and loss
        model.compile(optimizer=optimizer, loss=loss)
        # Print out a summary of the architecture
        print(model.summary())
        # Train the model, using the list of values for each node as input data, and the list of ground truths included in the original data set
        model.fit(
            x=[node.values for node in surface_topology_set.nodes],
            y=surface_topology_set.ground_truths
        )

    @classmethod
    def convolve_surface_topology(cls, surface_topology_set: SurfaceTopologySet, kernel_radius: int, filters: int, activation: str) -> SurfaceTopologySet:
        """Create a partial convolutional neural network graph, defining the operations for a single layer, and producing a new topology with the resulting graph"""
        # Create a single shared dense layer that outputs the number of filters
        filters_layer = Dense(filters, activation=activation)
        # Create an empty surface topology set to add modified nodes to
        modified_set = SurfaceTopologySet()
        # Iterate over each of the original nodes, creating graphs with them
        for node in surface_topology_set.nodes:
            # Attempt to form a kernel with this node and the provided radius
            kernel = cls.form_kernel(node, kernel_radius)
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
            modified_set.nodes.append(node_copy)
        # Return the set with newly defined tensors
        return modified_set

    @staticmethod
    def form_kernel(node: SurfaceTopologyNode, radius: int) -> Optional[List[int]]:
        """Given the identifier of a specific node and a kernel radius, return a list of identifiers corresponding to the nodes contained within a kernel around that node (or None if an edge was hit and a kernel could not be formed)"""
        # Create a list to add the nodes to
        nodes = []

        def traverse_node_tree(search_node: SurfaceTopologyNode, previous_node: Optional[SurfaceTopologyNode], depth: int) -> bool:
            """A recursive function to search outward from a specific node, adding more nodes to the list; return whether or not the search succeeded"""
            # If the search node is None, the search has hit an edge and needs to fail immediately
            if search_node is None:
                return False
            # If the node has already been added to the list, return without stopping the search (this is a normal endpoint)
            if search_node in nodes:
                return True
            # Add the node to the list
            nodes.append(search_node)
            # If the depth is 0, also return with success (but the node should still be added to the list)
            if depth == 0:
                return True
            # Otherwise, make a copy of the list of nodes connected to the current node
            connected_nodes = search_node.connected_nodes.copy()
            # The previous node is one of them; if it is provided, some extra steps must be taken to make sure the anticlockwise order is preserved
            if previous_node is not None:
                # Get the index of the previous node in the anticlockwise list of connections
                previous_node_index = connected_nodes.index(previous_node)
                # If it is at the beginning or the end of the list of connections, the anticlockwise order will be preserved, so it can simply be removed
                if previous_node_index in [0, len(connected_nodes) - 1]:
                    connected_nodes.remove(previous_node)
                # Otherwise, it is in the middle, and removing it will flip the order of the list, so take the last followed by the first
                # This is under the valid assumption that the nodes have up to 3 connections
                else:
                    connected_nodes = [connected_nodes[2], connected_nodes[0]]
            # Iterate over the possibly modified list of connected nodes and traverse the trees corresponding to them as well
            for connected_node in connected_nodes:
                # Use the current node as the next previous node, and reduce the depth by 1 so the search is not endless
                success = traverse_node_tree(connected_node, search_node, depth - 1)
                # If any traversal of a connected node fails, so does this one
                if not success:
                    return False
            # If everything has succeeded, return with success
            return True

        # Traverse the tree with the provided radius; do not provide a previous node to remove
        success = traverse_node_tree(node, None, radius)
        # If the search failed, return nothing
        if not success:
            return None
        # Otherwise, return the resulting list of nodes
        return nodes
