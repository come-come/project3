#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/19 16:54
@Author  : Junya Lu
@Site    : 
"""
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
rng = numpy.random

def tiduxiajiang():
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50
    # 数据集x
    train_X = numpy.asarray([3.3, 4.4, 5.5, 7.997, 5.654, .71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                             7.042, 10.791, 5.313, 9.27, 3.1])
    # 数据集y
    train_Y = numpy.asarray([1.7, 2.76, 3.366, 2.596, 2.53, 1.221, 1.694, 1.573, 3.465, 1.65, 2.09,
                             2.827, 3.19, 2.904, 2.42, 2.94, 1.3])
    n_samples = train_X.shape[0]

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    pred = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        # 训练数据
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

        print("优化完成!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        # 可视化显示
        plt.plot(train_X, train_Y, 'ro', label='Original data')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

def check_vocabulary():
    fr = open('G:\project3\\Data\\train\\genes\\vector_genes.txt', 'r')
    fr2 = open('G:\project3\\Data\\train\\terms\\vector_terms.txt' , 'r')
    dic1 = {}
    for line in fr.readlines():
        line_str = line.strip().split(' ')[0]
        dic1[line_str] = 0
    for line in fr2.readlines():
        line_str = line.strip().split(' ')[0]
        if line_str in dic1.keys():
            continue
        else:
            print (line_str)

if __name__ =='__main__':
    # check_vocabulary()
    a = tf.placeholder("float")
    b = tf.placeholder("float")
    y = tf.multiply(a, b)
    sess = tf.Session()
    print (sess.run(y, feed_dict={a:3, b:4}))
    sess.close()
