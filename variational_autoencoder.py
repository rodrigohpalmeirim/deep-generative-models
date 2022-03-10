import matplotlib.pyplot as plt
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from stacked_mnist import DataMode, StackedMNISTData
from verification_net import VerificationNet
from autoencoder import AutoEncoder, plot, comparison_plot


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, latent_dim=16, channels=1, force_learn: bool = False, file_name: str = "./models/variational_autoencoder_model") -> None:
        super(AutoEncoder, self).__init__()

        self.channels = channels
        self.force_relearn = force_learn
        self.file_name = file_name

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = Flatten()(encoder_inputs)
        x = Dense(256, activation="relu")(x)
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = Dense(256, activation="relu")(latent_inputs)
        x = Dense(784, activation="sigmoid")(x)
        decoder_outputs = Reshape((28, 28, 1))(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        self.compile(optimizer=keras.optimizers.Adam())

        self.done_training = self.load()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        for c in range(self.channels):
            channel_data = tf.expand_dims(data[:, :, :, c], axis=-1)

            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(channel_data)
                reconstruction = self.decoder(z)

                reconstruction_loss = keras.losses.binary_crossentropy(channel_data, reconstruction)
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=(1, 2)))

                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                total_loss = reconstruction_loss + kl_loss

            gradient = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradient, self.trainable_weights))

            self.kl_loss_tracker.update_state(kl_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.total_loss_tracker.update_state(total_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

    def train(self, generator: StackedMNISTData, epochs: int = 10) -> bool:
        self.done_training = self.load()

        if self.force_relearn or self.done_training is False:
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)
            data = np.concatenate([x_train, x_test], axis=0)

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            self.fit(data, batch_size=1024, epochs=epochs, callbacks=[tensorboard_callback])

            self.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def call(self, inputs: np.ndarray) -> np.ndarray:
        outputs = []
        for c in range(self.channels):
            z_mean, z_log_var, z = self.encoder(inputs[:, :, :, c])
            outputs.append(self.decoder(z))
        print(outputs)
        outputs = keras.layers.concatenate(outputs)
        return outputs


if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)

    verifier = VerificationNet(force_learn=False)
    verifier.train(generator=gen, epochs=5)

    vae = VariationalAutoEncoder(force_learn=False, channels=3, file_name="./models/vae_missing_model")
    vae.train(generator=gen, epochs=30)

    # Show some examples
    x_test, y_test = gen.get_random_batch(training=False, batch_size=10)
    x_reconstructed = vae(x_test)
    comparison_plot(x_test, x_reconstructed, y_test)

    # Generate images from noise
    random = np.random.randn(20, 16, 3)
    x_reconstructed = vae.decode(random)
    plot(x_reconstructed)

    # Evaluate performance
    x_test, y_test = gen.get_full_data_set(training=False)
    x_reconstructed = vae.predict(x_test)

    print("\nEvaluating performance...")
    cov = verifier.check_class_coverage(data=x_reconstructed, tolerance=.98)
    pred, acc = verifier.check_predictability(data=x_reconstructed, correct_labels=y_test)
    print(f"Coverage: {100 * cov:.2f}%")
    print(f"Predictability: {100 * pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")

    # Show most anomalous images
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    losses = []
    for i in range(x_test.shape[0]):
        loss = bce(y_true=x_test[i], y_pred=x_reconstructed[i])
        losses.append((loss, x_test[i], x_reconstructed[i], y_test[i]))

    losses = sorted(losses, key=lambda x: x[0], reverse=True)
    x_reconstructed = np.array([x[2] for x in losses[:10]])
    x_test = np.array([x[1] for x in losses[:10]])
    y_test = np.array([x[3] for x in losses[:10]])
    comparison_plot(x_test, x_reconstructed, y_test, block=True)
