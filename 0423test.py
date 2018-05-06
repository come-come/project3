#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/23 10:58
@Author  : Junya Lu
@Site    :
"""
import numpy as np
import time
import tensorflow as tf
def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([character for line in data.split('\n') for character in line.strip().split(' ')]))
    print (set_words)
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 30
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text.split(' ')] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text.split(' ')))

with open('G:\project3\Data\\train\genes\genes_one_line_space.txt', 'r', encoding='gb18030', errors='ignore') as f:
    source_data = f.read()
with open('G:\project3\Data\\train\\terms\\terms.txt', 'r') as f:
    target_data = f.read()
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)
# 输入一个单词
input_word = 'common'
input_word = 'lujunya'
input_word = 'apolipoprotein A1 apolipoprotein A2 apolipoprotein C3'
text = source_to_seq(input_word)

checkpoint = "./trained_model.ckpt"
batch_size = 128
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word.split(' '))] * batch_size,
                                      source_sequence_length: [len(input_word.split(' '))] * batch_size})[0]

pad = source_letter_to_int["<PAD>"]

print('原始输入:', input_word)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))