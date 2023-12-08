from typing import List
from layers.fully_connected import FullyConnected
from numpy import ndarray
import copy
import numpy as np
from layers.utils.activation_functions import Sigmoid, Softmax
from layers.utils.optimization_methods import GradientDescent
from networks.utils.loss import CrossEntropy, LossFunction, MeanSquaredError

from utils.dataset_loader import load_digits_dataset


class MLP:
    def __init__(
        self, layers: List[FullyConnected], input_size: int, loss_function: LossFunction
    ):
        self.layers = layers

        self.loss_function = loss_function

        for layer in self.layers:
            layer.initialize(input_size)
            input_size = layer.get_output_shape()

    def forward_prop(self, input: ndarray):
        output = input
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output

    def back_prop(self, loss_gradient: ndarray):
        for layer in reversed(self.layers):
            loss_gradient = layer.back_prop(loss_gradient)

    def train(self, data: ndarray, labels: ndarray, epochs: int):
        loss_per_epoch = []
        best_loss = np.inf
        best_model = None

        for epoch in range(epochs):
            losses = []

            for sample, label in zip(data, labels):
                output = self.forward_prop(sample)

                loss = self.loss_function.call(output, label)
                losses.append(loss)

                loss_gradient = self.loss_function.derivative(output, label)
                self.back_prop(loss_gradient)

            loss = np.mean(losses)
            loss_per_epoch.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(self)

            # Each 10% of the epochs, print the loss
            if epoch % (epochs // 10) == 0:
                print(f"Epoch: {epoch} Loss: {loss}")

        if best_model is not None:
            self = best_model

        return loss_per_epoch


