#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/16 11:24
@Author  : Junya Lu
@Site    : 
"""

TermFile = 'G:\project3\Data\go-basic.obo' # GO描述文件
fr = open(TermFile, 'r')
fr2 = open('G:\project3\Data\\term_descriptions.txt', 'r') # 名称 第一列为 Go def

dic_id = {}
dic_name= {}



# 构造字典 def: id
# 构造字典 def: name
for line in fr.readlines():
    line_str = line.strip().split(':')
    if line_str[0] == 'id':
        term_id = line_str[1].strip() + ':' +line_str[2].strip()
    if line_str[0] == 'name':
        term_name = line_str[1].strip()
    if line_str[0] == 'def':
        term_def = line_str[1].strip().split('"')[1]
        dic_id[term_def] = term_id
        dic_name[term_def] = term_name
    else:
        continue

fw = open('G:\project3\Data\\term_descriptions_new.txt', 'w')

for line in fr2.readlines():
    try:
        term_def = line.split('\t')[0]
        fw.write(dic_id[term_def] + '\t' + dic_name[term_def] + '\t' + line)
    except:
        print term_def
        print line
        continue


fw.close()
print len(dic_id)
print dic_id['The cell cycle process in which sister chromatids of a replicated chromosome are joined along the entire length of the chromosome during meiosis I.']
print dic_name['The cell cycle process in which sister chromatids of a replicated chromosome are joined along the entire length of the chromosome during meiosis I.']
