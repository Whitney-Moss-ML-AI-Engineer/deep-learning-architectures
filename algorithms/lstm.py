"""LSTM sequence model example."""
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20, 4)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.summary()
