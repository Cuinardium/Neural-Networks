from typing import List, Tuple

from numpy import ndarray
import numpy as np
import copy
from layers.fully_connected import FullyConnected

from layers.layer import Layer
from layers.utils.activation_functions import Identity
from layers.utils.optimization_methods import GradientDescent
from networks.utils.loss import LossFunction


class Autoencoder:
    def __init__(
        self,
        encoder: List[Layer],
        decoder: List[Layer],
        latent_dim: int,
        input_shape: int | Tuple,
        loss_function: LossFunction,
    ):
        self.encoder = encoder
        self.decoder = decoder

        for layer in self.encoder:
            layer.initialize(input_shape)
            input_shape = layer.get_output_shape()

        # TODO: Parametrize the optimizer
        self.latent_space = FullyConnected(
            latent_dim, Identity(), GradientDescent(0.01)
        )
        self.latent_space.initialize(input_shape)

        input_shape = self.latent_space.get_output_shape()
        for layer in self.decoder:
            layer.initialize(input_shape)
            input_shape = layer.get_output_shape()

        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.loss_function = loss_function

    def forward_prop(self, input: ndarray):
        output = input
        for layer in self.encoder:
            output = layer.forward_prop(output)

        output = self.latent_space.forward_prop(output)
        encoded = output

        for layer in self.decoder:
            output = layer.forward_prop(output)

        return output, encoded

    def back_prop(self, loss_gradient: ndarray):
        output = loss_gradient
        for layer in reversed(self.decoder):
            output = layer.back_prop(output)

        output = self.latent_space.back_prop(output)

        for layer in reversed(self.encoder):
            output = layer.back_prop(output)

    def train(self, data: ndarray, epochs: int):
        loss_per_epoch = []
        best_loss = np.inf
        best_model = None

        for epoch in range(epochs):
            losses = []

            for sample in data:
                output, _ = self.forward_prop(sample)

                loss = self.loss_function.call(output, sample)
                losses.append(loss)

                loss_gradient = self.loss_function.derivative(output, sample)
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
