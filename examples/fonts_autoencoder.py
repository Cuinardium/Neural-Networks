import json
import time

from networks.autoencoder import Autoencoder
from networks.utils.loss import CrossEntropy
from utils.dataset_loader import load_font_data
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

        print("Loading dataset")

        fonts, labels = load_font_data()
        training_data = fonts.reshape((fonts.shape[0], -1))
        data_shape = training_data.shape[1]
        font_shape = fonts.shape[1:]

        epochs = config["epochs"]
        activation_function = get_act_func(config["fully_connected_activation"])

        encoder = [
            FullyConnected(35, activation_function, get_optimization_method(config["optimizer"])),
            FullyConnected(20, activation_function, get_optimization_method(config["optimizer"])),
            FullyConnected(10, activation_function, get_optimization_method(config["optimizer"])),
        ]
        decoder = [
            FullyConnected(10, activation_function, get_optimization_method(config["optimizer"])),
            FullyConnected(20, activation_function, get_optimization_method(config["optimizer"])),
            FullyConnected(35, activation_function, get_optimization_method(config["optimizer"])),
        ]
        latent_dim = 2

        autoencoder = Autoencoder(
            encoder,
            decoder,
            latent_dim,
            data_shape,
            CrossEntropy(),
        )

    print("Starting training")

    loss_per_epoch = autoencoder.train(training_data, epochs)

    plot_errors_per_epoch(loss_per_epoch)
    save_errors_per_epoch(loss_per_epoch)

    # --- Visualization ---

    # Reconstruct the original fonts
    reconstructed = []
    encoded = []
    for sample in training_data:
        reconstruction, encoding = autoencoder.forward_prop(sample)
        reconstruction = reconstruction.reshape(font_shape)
        reconstructed.append(reconstruction)
        encoded.append(encoding)

    create_image(fonts, "original")
    create_image(reconstructed, "reconstructed")

    # Plot latent space
    plot_latent_space(encoded, labels)
  
    end_time = time.time()
    print(f"Finished in {int(end_time - start_time)} seconds")


if __name__ == "__main__":
    main()
