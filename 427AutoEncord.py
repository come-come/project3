#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/27 15:29
@Author  : Junya Lu
@Site    : 
"""
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in, fan_out, constant=1):
    '''
    Xavier 初始化器：根据输入结点数量和输出结点数量，自动调整合适的权重分布。让权重满足0均值，同时方差为2/(fan_in + fan_out).分布可以为均匀分布或者高斯分布。
    此处为均匀分布(low, high)。
    D(x) = (high-low)(high-low)/12 = 2/(fan_in + fan_out)
    初始化权重方法
    :输入结点数量 fan_in:
    :输出结点数量 fan_out:
    :param constant:
    :return:
    '''
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        '''

        :输入变量数 n_input:
        :隐层结点数 n_hidden:
        :隐层激活函数 transfer_function:
        :优化器 optimizer:
        :高斯降噪系数 scale:
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 定义自编码器的损失函数(平方误差)
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        '''
        初始化参数w1,w2,b1,b2
        :return:
        '''
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input]), dtype=tf.float32)
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)
        return  all_weights

    def partial_fit(self, X):
        '''
        用一个batch的数据进行训练，并返回当前的损失cost
        :一个batch的数据 X:
        :return:
        '''
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost
    def calc_total_cost(self, X):
        '''
        在自编码器训练完毕之后，在测试集上计算损失cost
        :测试集 X:
        :return:
        '''
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})
    def transform(self, X):
        '''
        返回编码器隐层的输出结果
        :param X:
        :return:
        '''
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})
    def generate(self, hidden=None):
        '''
        将隐层的输出结果作为输入给重构层
        :param hidden:
        :return:
        '''
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return  self.sess.run(self.reconstruction, feed_dict= {self.hidden: hidden})
    def reconstruction(self, X):
        '''
        整体运行一遍复原程序，包括提取高阶特征和通过高阶特征复原数据，包括transform和generate两块。
        输入为原始数据
        输出为重构后的数据
        :param X:
        :return:
        '''
        return  self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale:self.training_scale})
    def getWeight(self):
        '''
        获得隐含层的权重w1
        :return:
        '''
        return self.sess.run(self.weights['w1'])
    def getBiased(self):
        '''
        获得隐含层的偏置b1
        :return:
        '''
        return  self.sess.run(self.weights['b1'])


def standard_scale(X_train, X_test):
    '''
    对原始数据进行标准化（均值为0，标准差为1), 方法就是：先减去均值，再除以标准差。
    :param X_train:
    :param X_test:
    :return:
    '''
    preprocessor = prep.StandardScaler().fit(X_train) # 保证训练、测试数据使用相同的Scaler 保持数据的一致性
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test
def get_random_block_from_data(data, batch_size):
    '''
    获取一个batch数据
    :param data:
    :param batch_size:
    :return:
    '''
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index : (start_index+batch_size)]

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size
    if epoch%display_step == 0:
        print ('Epoch:', '%0.4d' %(epoch+1), 'cost=', '{:.9f}', format(avg_cost))
print ('Total cost:'+ str(autoencoder.calc_total_cost(X_test)))
