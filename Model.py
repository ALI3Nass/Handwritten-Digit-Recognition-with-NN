import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

x_train = x_train / 255
x_test = x_test / 255

x_tr_f = x_train.reshape(len(x_train), 28 * 28)
x_ts_f = x_test.reshape(len(x_test), 28 * 28)

model = tf.keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_tr_f, y_train, epochs=10)

model.evaluate(x_ts_f, y_test)

y_pred = model.predict(x_ts_f)

i = input('Please Enter an Index from 0-10000 to perform a prediction: ')
i = int(i)

model_prediction = np.argmax(y_pred[i])

plt.matshow(x_test[i])
plt.show()

print(f'Prediction done by NN was: {model_prediction} and the hand-written digit at {i} index was: {plt.matshow(x_test[i])}')

y_predicted_labels = [np.argmax(j) for j in y_pred]
Con_Mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(Con_Mat, annot=True, fmt='d')
plt.xlabel('Predicted value')
plt.ylabel('Actual Value')
plt.show()

