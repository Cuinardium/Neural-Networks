import json
import time

from networks.vae import VAE
from networks.utils.loss import MeanSquaredError
from utils.dataset_loader import load_emoji_data
from layers.fully_connected import FullyConnected
from layers.utils.activation_functions import get_act_func
from layers.utils.optimization_methods import get_optimization_method
from utils.plots import (
    create_image,
    plot_latent_space,
    plot_errors_per_epoch,
)
from utils.save import save_errors_per_epoch


def main():
    start_time = time.time()
    config_file = "config.json"

    with open(config_file) as json_file:
        config = json.load(json_file)

        print("Cargando el dataset de emojis")

        emojis, labels = load_emoji_data()
        training_data = emojis.reshape((emojis.shape[0], -1))
        data_shape = training_data.shape[1]
        emoji_shape = emojis.shape[1:]

        epochs = 30000
        activation_function = get_act_func(config["fully_connected_activation"])

        encoder = [
            FullyConnected(
                64, activation_function, get_optimization_method(config["optimizer"])
            ),
            FullyConnected(
                8, activation_function, get_optimization_method(config["optimizer"])
            ),

            FullyConnected(
                8, activation_function, get_optimization_method(config["optimizer"])
            ),
        ]
        decoder = [
            FullyConnected(
                8, activation_function, get_optimization_method(config["optimizer"])
            ),
            FullyConnected(
                8, activation_function, get_optimization_method(config["optimizer"])
            ),
            FullyConnected(
                64, activation_function, get_optimization_method(config["optimizer"])
            ),
        ]

        latent_dim = 2

        vae = VAE(
            encoder,
            decoder,
            latent_dim,
            data_shape,
            MeanSquaredError(),
            beta=0.001,
        )

    print("Entrenando el VAE")
    loss_per_epoch = vae.train(training_data, epochs)

    plot_errors_per_epoch(loss_per_epoch)
    save_errors_per_epoch(loss_per_epoch)

    # --- Visualización ---
    print("Reconstruyendo emojis y visualizando espacio latente")
    reconstructed = []
    encoded = []
    for sample in training_data:
        reconstruction, mu, logvar, z = vae.forward_prop(sample)
        print(f"mu: {mu}")
        reconstruction = reconstruction.reshape(emoji_shape)
        reconstructed.append(reconstruction)
        encoded.append(mu)  # Usamos mu como representación latente

    create_image(emojis, "original_emojis")
    create_image(reconstructed, "reconstructed_emojis")
    plot_latent_space(encoded, labels)

    end_time = time.time()
    print(f"Terminado en {int(end_time - start_time)} segundos")


if __name__ == "__main__":
    main()
