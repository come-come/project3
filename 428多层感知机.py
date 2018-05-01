#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/28 11:22
@Author  : Junya Lu
@Site    : 
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


# 1. 定义算法公式： 神经网络forward时的计算
# 给隐含层的参数设置Variable 并进行初始化
in_units = 784 # 输入节点数
h1_units = 300 # 隐层节点数
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))  #初始化为截断的正态分布，标准差为0.1
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None,in_units])
keep_prob = tf.placeholder(tf.float32)

#实现一个激活函数为ReLU的隐含层 y = relu(wx+b)
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

#实现Dropout功能 即随机将一部分结点置为0
hidden1_dropput = tf.nn.dropout(hidden1, keep_prob) # keep_prob为保留数据不置为0的比例，训练时候应该小于1 防止过拟合，预测时候等于1

#输出层
y = tf.nn.softmax(tf.matmul(hidden1_dropput, w2) + b2)

# 2. 定义损失函数和自适应优化器Adagrad 学习率设置为0.3
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 3. 训练步骤
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 4.对模型的准确率进行评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast 将数据由bool转为float32形式
# eval() 其实就是tf.Tensor的Session.run() 的另外一种写法.
# eval()只能用于tf.Tensor类对象，也就是有输出的Operation。对于没有输出的Operation, 可以用.run()或者Session.run()。Session.run()没有这个限制
print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

