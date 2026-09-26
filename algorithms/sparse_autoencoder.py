"""Sparse autoencoder using L1 activity regularization."""
import tensorflow as tf

X = tf.random.uniform((1000, 20), seed=42)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,)),
    tf.keras.layers.Dense(10, activation="relu", activity_regularizer=tf.keras.regularizers.L1(1e-5)),
    tf.keras.layers.Dense(20, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, X, epochs=10, batch_size=32, verbose=0)
