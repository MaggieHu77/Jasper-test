#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
import numpy as np


data = tf.placeholder(tf.float32, shape=(4, 2))
label = tf.placeholder(tf.float32, shape=(4, 1))
with tf.variable_scope('layer1') as scope1:
    weight = tf.get_variable(name="weight", shape=(2, 3))
    bias = tf.get_variable(name="bias", shape=(3, ))
    # x = tf.nn.sigmoid(tf.matmul(data, weight) + bias)
    x = tf.nn.softplus(tf.matmul(data, weight) + bias)

with tf.variable_scope('layer2') as scope2:
    weight = tf.get_variable(name="weight", shape=(3, 1))
    bias = tf.get_variable(name="bias", shape=(1,))
    x = tf.matmul(x, weight) + bias
# preds = tf.nn.sigmoid(x)
preds = tf.nn.sigmoid(x)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=x))
learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_label = np.array([[0], [1], [1], [0]])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for step in range(5001):
        if step < 1500:
            lr = 1
        elif step < 3000:
            lr = 0.1
        else:
            lr = 0.01
        _, l, pred = session.run([optimizer, loss, preds],
                                 feed_dict={data: train_data,
                                            label: train_label,
                                            learning_rate: lr})
        if not step % 1000:
            print('Step:{} -> Loss:{} -> Predictions{}'.format(step, l, pred))


# if __name__ == "__main__":
