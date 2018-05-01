#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/28 14:40
@Author  : Junya Lu
@Site    : 
"""
from tensorflow.examples.tutorials.mnist import input_data # 数字识别
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print (mnist.train.images.shape, mnist.train.labels.shape)
print (mnist.test.images.shape, mnist.test.labels.shape)
print (mnist.validation.images.shape, mnist.validation.labels.shape)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b) #定义算法公式# 定义loss ， 并选定优化器：随机梯度下降SGD
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run() #TensorFlow全局参数初始化器
for i in range(1000): # 迭代执行训练操作train_step
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_:batch_ys})# 对模型的准确率进行验证 tf.argmax(y, 1) 从tensor中寻找最大值的序号。判断预测类别和真实类别是否一致
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))