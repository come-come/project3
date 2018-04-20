#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/16 11:24
@Author  : Junya Lu
@Site    : 
"""
import pandas as pd
TermFile = 'G:\project3\Data\go-basic.obo' # GO描述文件
fr = open(TermFile, 'r')
fr2 = open('G:\project3\Data\\term_descriptions.txt', 'r') # 名称 第一列为 Go def

def add_columns():
    dic_id = {}
    dic_name = {}

    # 构造字典 def: id
    # 构造字典 def: name
    for line in fr.readlines():
        line_str = line.strip().split(':')
        if line_str[0] == 'id':
            term_id = line_str[1].strip() + ':' + line_str[2].strip()
        if line_str[0] == 'name':
            term_name = line_str[1].strip()
        if line_str[0] == 'def':
            term_def = line_str[1].strip().split('"')[1]
            dic_id[term_def] = term_id
            dic_name[term_def] = term_name
        else:
            continue

    fw = open('G:\project3\Data\\term_descriptions_new.txt', 'w')

    # 添加两列的信息
    for line in fr2.readlines():
        try:
            term_def = line.split('\t')[0]
            fw.write(dic_id[term_def] + '\t' + dic_name[term_def] + '\t' + line)
        except:
            print term_def
            print line
            continue

    fw.close()

def check_genes():
    # 检查注释基因是否正确
    # 生成对应注释基因的描述文件
    dic_gene_des = {}  # 基因的描述信息--基因名字
    gene_describe = 'G:\project3\Data\\Homo_sapiens.gene_info\\Homo_sapiens.gene_info'
    # gene_describe = 'G:\project3\Data\All_Mammalia.gene_info\All_Mammalia.gene_info' # 文件太大 跑不起来
    data = pd.read_table(gene_describe)

    for i in range(0, data.shape[0]):
        name = data.loc[i]['Symbol']
        describ = data.loc[i]['description']
        dic_gene_des[describ] = name
    print 'generate gene-describtion dictornary:', len(dic_gene_des)
    return dic_gene_des
def trans_des_to_gene(dic_gene_des):
    # 将基因的描述和基因关联起来
    fr = open('G:\project3\Data\\term_descriptions_new.txt', 'r') # 从第4列开始是基因的描述信息
    fw = open('G:\project3\Data\\term_descriptions_new_trans.txt', 'w')
    list = []
    for line in fr.readlines():
        line_str = line.strip().split('\t')
        fw.write(line_str[0] + '\t' + line_str[1] + '\t' + line_str[2])
        for x in range(3, len(line_str)):
            try:
                fw.write('\t' + dic_gene_des[line_str[x]])
            except:
                print line_str[x]
                list.append(line_str[x])
        fw.write('\n')
    print len(set(list))

    fw.close()
def number_genes():
    # GO term的信息
    TermFile = 'G:\project3\Data\go-basic.obo'  # GO描述文件
    fr = open(TermFile, 'r')
    dic_def = {} # 构造字典 id: def
    dic_name = {}   # 构造字典 def: name
    for line in fr.readlines():
        line_str = line.strip().split(':')
        if line_str[0] == 'id':
            term_id = line_str[1].strip() + ':' + line_str[2].strip()
        if line_str[0] == 'name':
            term_name = line_str[1].strip()
            dic_name[term_id] = term_name
        if line_str[0] == 'def':
            term_def = line_str[1].strip().split('"')[1]
            dic_def[term_id] = term_def
        else:
            continue
    print 'The number of term from obo file', len(dic_def)

    # 基因的信息
    dic_gene_des = {}
    gene_describe = 'G:\project3\Data\\Homo_sapiens.gene_info\\Homo_sapiens.gene_info'
    data = pd.read_table(gene_describe)
    for i in range(0, data.shape[0]):
        name = data.loc[i]['Symbol']
        describ = data.loc[i]['description']
        dic_gene_des[name] = describ
    print 'generate gene-describtion dictornary:', len(dic_gene_des)

    # GO 注释基因
    dic_term_gene ={}
    fr2 = open('G:\project3\\Data\goa_human.gaf\\goa_human.gaf', 'r')
    for line in fr2.readlines():
        line_str = line.strip().split('\t')
        gene = line_str[2]
        term = line_str[4]
        try:
            dic_term_gene[term].append(gene)
        except:
            dic_term_gene[term] = []
            dic_term_gene[term].append(gene)
    print 'The number of term from annotation file:', len(dic_term_gene)


    fw1 = open('G:\project3\\Data\description_gene_lu.txt', 'w')
    fw2 = open('G:\project3\\Data\description_def_lu.txt', 'w')

    for key, value in dic_term_gene.items():
        genes = list(set(value))
        try:
            # 生成文件
            gene_def = [dic_gene_des[i] for i in genes]
            fw1.write(key + '\t' + dic_name[key] + '\t' + dic_def[key] + '\t' + '\t'.join(genes) + '\n')
            fw2.write(key + '\t' + dic_name[key] + '\t' + dic_def[key] + '\t' + '\t'.join(gene_def) + '\n')
            # 生成训练数据和测试数据

        except:
            print i

    fw1.close()
    fw2.close()
def split_train_test():
    fr = open('G:\project3\\Data\description_def_lu.txt', 'r')
    fw = open('G:\project3\\Data\\train\\genes\\genes_one_line_space.txt', 'w')
    fw2 = open('G:\project3\\Data\\train\\terms\\terms.txt', 'w')
    for line in  fr.readlines():
        line = line.replace(',', '').replace('.', '') # remove , .
        line_str = line.strip().split('\t')
        fw2.write(line_str[1] + '\n')
        for i in range(3, len(line_str)):
            fw.write(line_str[i] + ' ')
        fw.write('\n')
    fw.close()
    fw2.close()

if __name__ == '__main__':
    # dic_gene_des = check_genes()
    # trans_des_to_gene(dic_gene_des)
    # print dic_gene_des['immunoglobulin heavy constant gamma 1 (G1m marker)']
    # number_genes()
    split_train_test()





