"""Minimal variational autoencoder example."""
import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(latent_dim * 2),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(784, activation="sigmoid"),
        ])
        self.latent_dim = latent_dim

    def call(self, x):
        params = self.encoder(x)
        mean, log_var = tf.split(params, 2, axis=1)
        eps = tf.random.normal(tf.shape(mean))
        z = mean + tf.exp(0.5 * log_var) * eps
        reconstruction = self.decoder(z)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, reconstruction), axis=-1))
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))
        self.add_loss(recon_loss + kl)
        return reconstruction

vae = VAE()
vae.compile(optimizer="adam")
