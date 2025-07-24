from typing import List, Tuple
import numpy as np
from numpy import ndarray
import copy
from layers.layer import Layer
from layers.fully_connected import FullyConnected
from layers.utils.activation_functions import Identity
from layers.utils.optimization_methods import GradientDescent
from networks.utils.loss import LossFunction


class VAE:
    def __init__(
        self,
        encoder: List[Layer],
        decoder: List[Layer],
        latent_dim: int,
        input_shape: int | Tuple,
        loss_function: LossFunction,
        beta: float = 1.0,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.loss_function = loss_function
        self.beta = beta

        # Initialize encoder layers
        for layer in self.encoder:
            layer.initialize(input_shape)
            input_shape = layer.get_output_shape()

        # Latent space layers for mu and logvar
        self.fc_mu = FullyConnected(latent_dim, Identity(), GradientDescent(0.01))
        self.fc_logvar = FullyConnected(latent_dim, Identity(), GradientDescent(0.01))
        self.fc_mu.initialize(input_shape)
        self.fc_logvar.initialize(input_shape)

        # Initialize decoder layers
        decoder_input_shape = latent_dim
        for layer in self.decoder:
            layer.initialize(decoder_input_shape)
            decoder_input_shape = layer.get_output_shape()

    def encode(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        output = x
        for layer in self.encoder:
            output = layer.forward_prop(output)
        mu = self.fc_mu.forward_prop(output)
        logvar = self.fc_logvar.forward_prop(output)
        return mu, logvar

    def reparameterize(self, mu: ndarray, logvar: ndarray) -> ndarray:
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: ndarray) -> ndarray:
        output = z
        for layer in self.decoder:
            output = layer.forward_prop(output)
        return output

    def forward_prop(self, x: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z

    def vae_loss(
        self, reconstructed: ndarray, x: ndarray, mu: ndarray, logvar: ndarray
    ) -> float:
        # Reconstruction loss
        recon_loss = self.loss_function.call(reconstructed, x)

        # KL divergence between q(z|x) and N(0,1)
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

        return recon_loss + self.beta * kl_loss

    def back_prop(
        self,
        reconstructed: ndarray,
        x: ndarray,
        mu: ndarray,
        logvar: ndarray,
        z: ndarray,
    ):
        # Gradiente de la reconstrucción respecto a la salida del decoder
        # (Esto es igual que en un autoencoder normal)
        recon_grad = self.loss_function.derivative(reconstructed, x)

        # Propagamos el gradiente por el decoder (de salida a entrada)
        # Salimos con el gradiente respecto a z
        grad_z = recon_grad
        for layer in reversed(self.decoder):
            grad_z = layer.back_prop(grad_z)

        # --- Reparameterization trick ---
        # Queremos los gradientes respecto a mu y logvar, usando la regla de la cadena
        # z = mu + std * eps, donde std = exp(0.5 * logvar), eps ~ N(0,1)
        # grad_z es dL/dz

        # Calculamos eps (lo que usamos para muestrear z)
        # Si z = mu + std * eps => eps = (z - mu) / std
        std = np.exp(0.5 * logvar)
        eps = (z - mu) / std

        # Gradiente de la reconstrucción respecto a mu
        # dz/dmu = 1, entonces dL/dmu = dL/dz * 1 = grad_z
        recon_grad_mu = grad_z

        # Gradiente de la reconstrucción respecto a logvar
        # dz/dlogvar = std * eps * 0.5, entonces dL/dlogvar = dL/dz * dz/dlogvar
        recon_grad_logvar = grad_z * 0.5 * std * eps

        # --- KL divergence ---
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # Gradiente respecto a mu: dKL/dmu = mu
        # Gradiente respecto a logvar: dKL/dlogvar = 0.5 * (exp(logvar) - 1)
        kl_grad_mu = mu
        kl_grad_logvar = 0.5 * (np.exp(logvar) - 1)

        # --- Sumar gradientes ---
        # Total gradiente respecto a mu y logvar
        total_grad_mu = recon_grad_mu + self.beta * kl_grad_mu
        total_grad_logvar = recon_grad_logvar + self.beta * kl_grad_logvar

        # --- Backprop por las capas fc_mu y fc_logvar ---
        # Propagamos los gradientes por las dos ramas
        grad_encoder_mu = self.fc_mu.back_prop(total_grad_mu)
        grad_encoder_logvar = self.fc_logvar.back_prop(total_grad_logvar)

        # --- Sumar gradientes de ambas ramas ---
        # El encoder recibe la suma de los gradientes de ambas ramas
        grad_encoder = grad_encoder_mu + grad_encoder_logvar

        # --- Backprop por el encoder ---
        for layer in reversed(self.encoder):
            grad_encoder = layer.back_prop(grad_encoder)

    def train(self, data: ndarray, epochs: int):
        loss_per_epoch = []
        best_loss = np.inf
        best_model = None

        for epoch in range(epochs):
            losses = []

            for x in data:
                reconstructed, mu, logvar, z = self.forward_prop(x)

                loss = self.vae_loss(reconstructed, x, mu, logvar)
                losses.append(loss)

                self.back_prop(reconstructed, x, mu, logvar, z)

            avg_loss = np.mean(losses)
            loss_per_epoch.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = copy.deepcopy(self)

            if epoch % (epochs // 10) == 0:
                print(f"Epoch: {epoch} Loss: {avg_loss}")

        if best_model is not None:
            self = best_model

        return loss_per_epoch
