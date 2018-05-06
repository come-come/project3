#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/16 11:24
@Author  : Junya Lu
@Site    :
"""
import pandas as pd
import random
from matplotlib import pyplot as plt

def new_data():
    '''
    根据source sentence长度，选长度【3，53】的数据， 约78%
    根据target sentence长度，选长度【2，7】的数据，约88%
    两者取交集， 最后留下对应上的69.3%的数据 （11464 / 16541）
    :return:
    '''
    fr2 = open('G:\project3\\Data\\train\\genes\\genes_one_line_space.txt', 'r')
    fr = open('G:\project3\\Data\\train\\terms\\terms.txt', 'r')

    term_name_length = []
    gene_def_length = []
    dic_term = {}
    dic_gene = {}
    i = 0
    j = 0

    # terms
    for line in fr.readlines():
        sentence_length = line.strip().split(' ')
        term_name_length.append(len(sentence_length))
        dic_term[i] = line.strip()
        i += 1

    # genes
    for line in fr2.readlines():
        sentence_length = line.strip().split(' ')
        gene_def_length.append(len(sentence_length))
        dic_gene[j] = line.strip()
        j += 1

    print(dic_gene[1])
    print(dic_term[1])
    print('term', 'genes')
    print(max(term_name_length), max(gene_def_length))
    print(gene_def_length.index(max(gene_def_length)))
    print(min(term_name_length), min(gene_def_length))
    print(sum(term_name_length) / len(term_name_length), sum(gene_def_length) / len(gene_def_length))

    print(gene_def_length)

    data_size = 16541
    top_n = int(16541 * 0.15)
    sorted_list1 = sorted(term_name_length)
    print(sorted_list1)
    print(sorted_list1[top_n], sorted_list1[data_size - top_n])
    sorted_list2 = sorted(gene_def_length)
    print(sorted_list2[top_n], sorted_list2[data_size - top_n])

    list_1 = [i for i in gene_def_length if i > 2 and i < 54]
    list_2 = [i for i in term_name_length if i > 1 and i < 8]
    print(len(list_1), len(list_1) / data_size)
    print(len(list_2), len(list_2) / data_size)

    fw1 = open('genes0504.txt', 'w')
    fw2 = open('terms0504.txt', 'w')

    total = 0
    for key, value in dic_gene.items():
        length_gene = len(value.split(' '))
        length_term = len(dic_term[key].split(' '))
        if length_gene > 2 and length_gene < 54:
            if length_term > 1 and length_term < 8:
                fw1.write(value + '\n')
                fw2.write(dic_term[key] + '\n')
                total += 1
    fw1.close()
    fw2.close()
    print(total, total / data_size)




    # plt.hist(term_name_length)
    # plt.show()
    # plt.hist(gene_def_length)
    # plt.show()

def get_more_data():
    '''
    通过置换基因的顺序，得到多个sentence对应同一target
    :return:
    '''

    fr1 = open('genes0504.txt', 'r')
    fr2 = open('terms0504.txt', 'r')
    dic_gene = {}
    dic_term = {}
    i = 0
    j = 0
    for line in fr1.readlines():
        dic_gene[i] = line.strip()
        i += 1
    for line in fr2.readlines():
        dic_term[j] = line.strip()
        j += 1

    # termName--Genes
    fr_des = open('G:\project3\\Data\\description_def_lu.txt', 'r')
    dic = {}
    for line in fr_des.readlines():
        line = line.replace(',', '').replace('.', '') # remove , .
        line_str = line.strip().split('\t')
        term_name = line_str[1].strip()
        # term_name
        # term_name_length = len(term_name.split(' '))
        genes = line_str[3:]
        dic[term_name] = genes

    # 符合要求的进行数据扩增
    fw = open('source.txt', 'w')
    fw2 = open('target.txt', 'w')
    fw3 = open('source_target.txt', 'w')
    for key, value in dic_term.items():
        target = value
        source = dic[value]
        if len(source) < 2:
            fw.write(' '.join(source) + '\n')
            fw2.write(target + '\n')
            fw3.write(target + '\t' + ' '.join(source) + '\n')
        elif len(source) == 2:
            fw.write(source[0] + ' ' + source[1] + '\n')
            fw2.write(target + '\n')
            fw3.write(target + '\t' + source[0] + ' ' + source[1] + '\n')
            fw.write(source[1] + ' ' + source[0] + '\n')
            fw2.write(target + '\n')
            fw3.write(target + '\t' + source[1] + ' ' + source[0] + '\n')
        else:
            for t in range(1, len(source)):
                fw.write(' '.join(source) + '\n')
                fw2.write(target + '\n')
                fw3.write(target + '\t' + ' '.join(source) + '\n')
                random.shuffle(source)
    fw.close()
    fw2.close()
    fw3.close()


if __name__=='__main__':
    print ('begin')
    get_more_data()








