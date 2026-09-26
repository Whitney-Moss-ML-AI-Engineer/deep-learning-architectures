"""GRU sequence-regression example."""
import numpy as np
import tensorflow as tf

X = np.array([[[i + j] for j in range(3)] for i in range(200)], dtype=float)
y = np.array([i + 3 for i in range(200)], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3, 1)),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, batch_size=16, verbose=0)
print(model.predict(X[:5], verbose=0).ravel())
