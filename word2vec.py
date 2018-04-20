#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/17 14:25
@Author  : Junya Lu
@Site    : 
"""
import warnings
from gensim.models.word2vec import LineSentence
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import word2vec
from string import punctuation

# LineSentence('G:\project3\\Data\\train\\genes\\genes_one_line_space.txt')
# model = word2vec.Word2Vec(LineSentence('G:\project3\\Data\\train\\genes\\genes_one_line_space.txt'))
# model = word2vec.Word2Vec(LineSentence('G:\project3\\Data\\train\\terms\\terms.txt'))
model = word2vec.Word2Vec(LineSentence('G:\project3\\Data\\train\\all.txt'))
print ('the number of vocabulary', len(model.wv.vocab))
vocab = list(model.wv.vocab.keys())
print(vocab[:10])
model.save('G:\project3\\Data\\train\\w2v.model')
model.wv.save_word2vec_format('G:\project3\\Data\\train\\vector_all.txt')


# print (model.similarity('dogs', 'you'))
# print (model.similar_by_vector('dogs'))
# print (model['you'])