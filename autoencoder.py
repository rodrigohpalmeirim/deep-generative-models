import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Model, Sequential
from stacked_mnist import DataMode, StackedMNISTData
from verification_net import VerificationNet


class AutoEncoder(Model):
    def __init__(self, latent_dim=64, channels=1, force_learn: bool = False, file_name: str = "./models/autoencoder_model") -> None:
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.force_relearn = force_learn
        self.file_name = file_name

        self.encoder = Sequential()
        self.encoder.add(Flatten())
        self.encoder.add(Dense(256, activation="relu"))
        self.encoder.add(Dense(latent_dim, activation="relu"))

        self.decoder = Sequential()
        self.decoder.add(Dense(256, activation="relu"))
        self.decoder.add(Dense(784, activation="sigmoid"))
        self.decoder.add(Reshape((28, 28, 1)))

        self.compile(loss=keras.losses.binary_crossentropy,
                     optimizer=keras.optimizers.Adam(lr=.01),
                     metrics=["accuracy"])

        self.done_training = self.load()

    def call(self, inputs: np.ndarray) -> np.ndarray:
        outputs = []
        for c in range(self.channels):
            outputs.append(self.decoder(self.encoder(inputs[:, :, :, c])))
        outputs = keras.layers.concatenate(outputs)
        return outputs

    def load(self):
        try:
            self.load_weights(filepath=self.file_name)
            done_training = True
        except Exception:
            print("Could not read weights for autoencoder_model from file. Must retrain...")
            done_training = False

        return done_training

    def train(self, generator: StackedMNISTData, epochs: int = 10) -> bool:
        self.done_training = self.load()

        if self.force_relearn or self.done_training is False:
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            self.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                     validation_data=(x_test, x_test), callbacks=[tensorboard_callback])

            self.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def decode(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(shape=(x.shape[0], 28, 28, x.shape[-1]))
        for c in range(x.shape[-1]):
            y[:, :, :, c] = self.decoder(x[:, :, c])[:, :, :, 0]
        return y


def comparison_plot(original, reconstructed, labels, title="", block=False) -> None:
    no_cols = original.shape[0]
    no_channels = original.shape[-1]
    images = np.concatenate((original, reconstructed), axis=1)
    plt.figure()
    plt.suptitle(title)
    for i in range(no_cols):
        plt.subplot(2, no_cols // 2, i + 1)
        if no_channels == 1:
            plt.imshow(images[i, :, :], cmap="binary")
        else:
            plt.imshow(images[i, :, :].astype(float))
        plt.xticks([])
        plt.yticks([])
        plt.title(str(labels[i]).zfill(no_channels))
    plt.show(block=block)
    plt.pause(0.1)


def plot(images, title="", block=False) -> None:
    no_cols = int(np.ceil(np.sqrt(images.shape[0])))
    no_lines = int(np.sqrt(images.shape[0]))
    no_channels = images.shape[-1]
    plt.figure()
    plt.suptitle(title)
    for i in range(len(images)):
        plt.subplot(no_lines, no_cols, i + 1)
        if no_channels == 1:
            plt.imshow(images[i, :, :], cmap="binary")
        else:
            plt.imshow(images[i, :, :].astype(float))
        plt.xticks([])
        plt.yticks([])
    plt.show(block=block)
    plt.pause(0.1)


if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)

    verifier = VerificationNet(force_learn=False)
    verifier.train(generator=gen, epochs=5)

    ae = AutoEncoder(force_learn=False, channels=3, file_name="./models/ae_missing_model")
    ae.train(generator=gen, epochs=10)

    # Show some examples
    x_test, y_test = gen.get_random_batch(training=False, batch_size=10)
    x_reconstructed = ae(x_test)
    comparison_plot(x_test, x_reconstructed, y_test, title="Reconstruction")

    # Generate images from noise
    random = np.random.randn(20, ae.latent_dim, ae.channels)
    r_reconstructed = ae.decode(random)
    plot(r_reconstructed, title="Generation from Random Samples")

    # Evaluate reconstruction performance
    x_test, y_test = gen.get_full_data_set(training=False)
    x_reconstructed = ae(x_test)

    print("\nEvaluating reconstruction performance...")
    cov = verifier.check_class_coverage(data=x_reconstructed, tolerance=.98)
    pred, acc = verifier.check_predictability(data=x_reconstructed, correct_labels=y_test)
    print(f"Coverage: {100 * cov:.2f}%")
    print(f"Predictability: {100 * pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")

    # Evaluate generative performance
    random = np.random.randn(10000, ae.latent_dim, ae.channels)
    r_reconstructed = ae.decode(random)

    print("\nEvaluating generative performance...")
    cov = verifier.check_class_coverage(data=r_reconstructed, tolerance=.98)
    pred, _ = verifier.check_predictability(data=r_reconstructed, correct_labels=y_test)
    print(f"Coverage: {100 * cov:.2f}%")
    print(f"Predictability: {100 * pred:.2f}%")

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
    comparison_plot(x_test, x_reconstructed, y_test, title="Most Anomalous Images", block=True)
