#!/usr/bin/python
# coding:utf-8

from keras.models import Sequential
from keras.layers import Dense
import numpy as np


model = Sequential()
model.add(Dense(units=3, activation='relu', input_dim=1000))
# model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
x_train = np.ones((3, 1000))
y_train = np.dot(np.array([7, 8.2, -4.5]), np.ones((3, 1000))) + np.random.randn(1000)
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_train.T, y_train.T, batch_size=128)
classes = model.predict(x_train, batch_size=128)