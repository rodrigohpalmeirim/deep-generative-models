import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from stacked_mnist import DataMode, StackedMNISTData
from verification_net import VerificationNet


class AutoEncoder():
    def __init__(self, latent_dim=64, force_learn: bool = False, file_name: str = "./models/autoencoder_model") -> None:
        self.force_relearn = force_learn
        self.file_name = file_name

        self.encoder = Sequential()
        self.encoder.add(Flatten())
        self.encoder.add(Dense(latent_dim, activation="relu"))

        self.decoder = Sequential()
        self.decoder.add(Dense(latent_dim, activation="relu"))
        self.decoder.add(Dense(784, activation="sigmoid"))
        self.decoder.add(Reshape((28, 28)))

        self.model = Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=keras.optimizers.Adam(lr=.01),
                           metrics=["accuracy"])
        
        self.done_training = self.load_weights()

    def load_weights(self):
        try:
            self.model.load_weights(filepath=self.file_name)
            done_training = True
        except Exception:
            print("Could not read weights for autoencoder_model from file. Must retrain...")
            done_training = False

        return done_training

    def train(self, generator: StackedMNISTData, epochs: np.int = 10) -> bool:
        self.done_training = self.load_weights()

        if self.force_relearn or self.done_training is False:
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)

            x_train = x_train[:, :, :, 0]
            x_test = x_test[:, :, :, 0]

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            self.model.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                           validation_data=(x_test, x_test), callbacks=[tensorboard_callback])

            self.model.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training


def plot(original, reconstructed, labels, channels=1) -> None:
    no_cols = original.shape[0]
    images = np.concatenate((original, reconstructed), axis=1)
    plt.Figure()
    for i in range(no_cols):
        plt.subplot(2, no_cols // 2, i + 1)
        if channels == 1:
            plt.imshow(images[i, :, :], cmap="binary")
        else:
            plt.imshow(images[i, :, :].astype(np.float))
        plt.xticks([])
        plt.yticks([])
        plt.title(str(labels[i]).zfill(channels))

    plt.show()


if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)

    verifier = VerificationNet(force_learn=False)
    verifier.train(generator=gen, epochs=5)

    ae = AutoEncoder(force_learn=False)
    ae.train(generator=gen, epochs=10)

    x_test, y_test = gen.get_full_data_set(training=False)
    x_reconstructed = ae.model(x_test)
    x_reconstructed = np.expand_dims(x_reconstructed, axis=-1)

    print(x_test.shape)
    print(x_reconstructed.shape)

    cov = verifier.check_class_coverage(data=x_reconstructed, tolerance=.98)
    pred, acc = verifier.check_predictability(data=x_reconstructed, correct_labels=y_test)
    print(f"Coverage: {100*cov:.2f}%")
    print(f"Predictability: {100*pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")

    x_test, y_test = gen.get_random_batch(training=False, batch_size=10)
    x_test = x_test[:, :, :, 0]

    x_reconstructed = ae.model(x_test)
    
    plot(x_test, x_reconstructed, y_test)