import logging
import argparse
import json
import os
import random
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pickle
from torch.utils.data import Dataset


def pos(num,position_num):
    if num < -position_num:
        return 0
    if -position_num <= num <= position_num:
        return num + position_num + 1
    if num > position_num:
        return (position_num + 1)*2

def position_padding(words,max_seq_len,position_num):
    words = [pos(word,position_num) for word in words]
    if len(words) >= max_seq_len:
        return words[:max_seq_len]
    words.extend([position_num*2+1]*(max_seq_len-len(words)))
    return words

def read_data(args,mode='train'):
    '''数据集读取'''
    sentences = []
    relations = []
    positions1 = []
    positions2 = []
    if mode == 'train':
        file = args.train_file
        num = args.train_data_num
    elif mode == 'valid':
        file = args.valid_file
        num = args.valid_data_num
    else:
        raise ValueError('mode must be in {train, valid}')
    logging.info(f"====load data from {file}====")
    with open (file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[:num]
        with tqdm(lines,desc=f'read {mode} data') as t:
            for line in t:
                dic = json.loads(line)
                sentence = []
                index1 = dic['text'].lower().index(dic['ent1'].lower())
                position1 = []
                index2 = dic['text'].lower().index(dic['ent2'].lower())
                position2 = []
                for i ,word in enumerate(dic['text']):
                    sentence.append(word)
                    position1.append(i-index1)
                    position2.append(i-index2)
                sentences.append(sentence)
                relations.append(dic['rel'])
                positions1.append(position1)
                positions2.append(position2)
        assert len(sentences) == len(positions1) == len(positions2) == len(relations),\
        logging.error('load data error')

    c = Counter(relations)
    for key ,value in c.items():
        logging.info("=====  关系：{:<20s}\t数量：{:<10d} =====".format(key, value))

    relation_list = list(set(c.keys()))  #构建关系表格

    if len(relation_list) > 0:
        with open(os.path.join(args.output_data_dir, "id2relation.pkl"), "rb") as f:
            id2relation = pickle.load(f)
            relation2id = {value: key for key, value in id2relation.items()}

    else:
        raise ValueError('data Wrong No relation')
    return sentences,relations,positions1,positions2,relation2id

def precess_data(args,mode='train'):
    samples = []
    sentences, relations, positions_1, positions_2, relation2id = read_data(args, mode)
    logging.info(f'========== constructing {mode} input data ...... ==========')
    tokenizer = BertTokenizer.from_pretrained(args.embedding_path,do_lower_case=args.do_lower_case)
    with tqdm(range(len(sentences)),desc=f'construct {mode} bert input data') as iter: #得到句子的数量
        for index in iter: #根据句子数量获取相应句子的索引
            sentence = ['[CLS]'] + sentences[index][:args.max_seq_len-2] + ['[SEP]'] #每一个句子补上cls和sep
            relation = relations[index]
            input_mask = [1] * len(sentence)
            input_segment = [0] * len(sentence)
            input_ids = tokenizer.convert_tokens_to_ids(sentence)
            relation_ids = [relation2id[relation]]
            position1 = position_padding(positions_1[index],args.max_seq_len,args.position_num)
            position2 = position_padding(positions_2[index], args.max_seq_len, args.position_num)
            while len(input_ids) < args.max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                input_segment.append(0)
            assert len(input_ids) == args.max_seq_len
            assert len(input_mask) == args.max_seq_len
            assert len(input_segment) == args.max_seq_len
            assert len(position1) == args.max_seq_len
            assert len(position2) == args.max_seq_len
            samples.append((input_ids, input_segment, input_mask, relation_ids, position1, position2))
    return samples

class ReDataset(Dataset):
    def __init__(self,args,mode='train'):
        self.args = args
        self.samples = precess_data(args,mode=mode)

    def __getitem__(self, idx):
        return torch.LongTensor(self.samples[idx][0]),\
               torch.LongTensor(self.samples[idx][1]),\
               torch.LongTensor(self.samples[idx][2]),\
               torch.LongTensor(self.samples[idx][3]),\
               torch.LongTensor(self.samples[idx][4]),\
               torch.LongTensor(self.samples[idx][5])

    def __len__(self):
        return len(self.samples)  #samples中有多少句话


def read_only_bert_data(args,mode='train'):  #对纯bert模型数据读取的处理函数
    '''数据集读取'''
    sentences = []
    relations = []
    if mode == 'train':
        file = args.train_file
        num = args.train_data_num
    elif mode == 'valid':
        file = args.valid_file
        num = args.valid_data_num
    else:
        raise ValueError('mode must be in {train, valid}')
    logging.info(f"====load data from {file}====")
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[:num]
        with tqdm(lines, desc=f'read {mode} data') as t:
            for line in t:
                dic = json.loads(line)
                sentence = []
                temp = dic['ent1']+dic['ent2']+dic['text']
                for i, word in enumerate(temp):
                    sentence.append(word)
                sentences.append(sentence)
                relations.append(dic['rel'])
        assert len(sentences) == len(relations), logging.error('load data error')

    c = Counter(relations)
    for key, value in c.items():
        logging.info("=====  关系：{:<20s}\t数量：{:<10d} =====".format(key, value))

    relation_list = list(set(c.keys()))  # 构建关系表格

    if len(relation_list) > 0:
        with open(os.path.join(args.output_data_dir, "id2relation.pkl"), "rb") as f:
            id2relation = pickle.load(f)
            relation2id = {value: key for key, value in id2relation.items()}

    else:
        raise ValueError('data Wrong No relation')
    return sentences, relations,relation2id

def precess_only_bert_data(args,mode='train'):  #纯bert模型的数据处理函数
    samples = []
    sentences, relations, relation2id = read_only_bert_data(args, mode)
    logging.info(f'========== constructing {mode} input data ...... ==========')
    tokenizer = BertTokenizer.from_pretrained(args.embedding_path,do_lower_case=args.do_lower_case)
    with tqdm(range(len(sentences)),desc=f'construct {mode} bert input data') as iter: #得到句子的数量
        for index in iter: #根据句子数量获取相应句子的索引
            sentence = sentences[index][:args.max_seq_len-2] #每一个句子补上cls和sep
            relation = relations[index]
            input_mask = [1] * (len(sentence)+2)
            input_segment = [0] * (len(sentence)+2)
            input_ids = tokenizer.encode(sentence)  #enocde自带cls和esp
            relation_ids = [relation2id[relation]]
            while len(input_ids) < args.max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                input_segment.append(0)
            assert len(input_ids) == args.max_seq_len
            assert len(input_mask) == args.max_seq_len
            assert len(input_segment) == args.max_seq_len
            samples.append((input_ids, input_segment, input_mask, relation_ids))
    return samples

class OnlyBertReDataset(Dataset):
    def __init__(self,args,mode='train'):
        self.args = args
        self.samples = precess_only_bert_data(args,mode=mode)

    def __getitem__(self, idx):
        return torch.LongTensor(self.samples[idx][0]),\
               torch.LongTensor(self.samples[idx][1]),\
               torch.LongTensor(self.samples[idx][2]),\
               torch.LongTensor(self.samples[idx][3])

    def __len__(self):
        return len(self.samples)  #samples中有多少句话








