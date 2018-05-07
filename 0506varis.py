#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
@Time : 2018/5/6 20:27 
@Author : junya lu
@File : 0506varis.py 
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

decoder_embeddings = tf.Variable(tf.random_uniform([4, 5]))
decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, [0,1,2])

embedding = tf.Variable(np.identity(5,dtype=np.int32))
input_ids = tf.placeholder(dtype=tf.int32,shape=[None])
input_embedding = tf.nn.embedding_lookup(embedding,input_ids)



'''
tf.contrib.layers.embed_sequence
'''
vocab = [{'garbage':1},
         {'piles':2},
         {'in':3},
         {'the':4},
         {'city':5},
         {'is':6},
         {'clogged':7},
         {'with':8},
         {'vehicles':9}]

features = [[1, 2, 3, 4, 5], [5, 6, 7,8,6]]
encoder_embed_input = tf.contrib.layers.embed_sequence(ids=features, vocab_size=len(vocab), embed_dim=5)

sess.run(tf.initialize_all_variables())

print(sess.run(embedding))
print('tf.nn.embedding_lookup', sess.run(input_embedding,feed_dict={input_ids:[1,2,3,0,3,2,1]}))
print (sess.run(decoder_embeddings))
print ('tf.nn.embedding_lookup', sess.run(decoder_embed_input))

print ('tf.contrib.layers.embed_sequence: ', sess.run(encoder_embed_input))
