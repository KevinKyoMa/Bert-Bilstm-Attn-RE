#!/usr/bin/env python
# coding=utf-8

import json
from collections import Counter
import time
import logging
import random
import os
import torch
import numpy as np
import pickle
logger = logging.getLogger(__name__)



def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list

class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""
    def __init__(self,golden_tags, predict_tags, remove_O=False):
        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()
        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

    def cal_precision(self):

        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / (self.predict_tags_counter[tag] + 0.0001)

        return precision_scores

    def cal_recall(self):

        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / (self.golden_tags_counter[tag] + 0.0001)
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self,args,epoch):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        row_format = '{:>9}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        if args.epoch-1 == epoch:
            logger.info(header_format.format('', *header))
            #打印每个标签的 精确率、召回率、f1分数
            for tag in self.tagset:
                logger.info(row_format.format(
                    tag,
                    self.precision_scores[tag],
                    self.recall_scores[tag],
                    self.f1_scores[tag],
                    self.golden_tags_counter[tag]
                ))

        # 计算并打印平均值
        avg_metrics = self._cal_weighted_average()
        if args.epoch-1 == epoch:
            logger.info(row_format.format(
                'avg/total',
                avg_metrics['precision'],
                avg_metrics['recall'],
                avg_metrics['f1_score'],
                len(self.golden_tags)
            ))
        return avg_metrics

    def count_correct_tags(self):
        """
        计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算
        """
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def _remove_Otags(self):
        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        logger.info("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """计算混淆矩阵"""
        logger.info("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size+1)
        logger.info(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            logger.info(row_format_.format(tag_list[i], *row))

def set_seed(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def del_file(path):
    """递归删除文件夹"""
    ls = os.listdir(path)
    for i in os.listdir(path):
        c_path = os.path.join(path, i)
        print(c_path)
        if os.path.isdir(c_path):
            del_file(c_path)
            os.rmdir(c_path)
        else:
            os.remove(c_path)

def get_re_label_dict(input_path, output_path):
    label = set()
    with open(input_path, 'r', encoding='utf-8') as f:
        temp = f.readlines()
        for line in temp:
            line = json.loads(line)
            label.add(line['rel'])
    relation2id = {v: k for k, v in enumerate(label)}
    id2relation = {value: key for key, value in relation2id.items()}
    with open(os.path.join(output_path, "id2relation.pkl"), "wb") as f:
        pickle.dump(id2relation, f)
    with open(os.path.join(output_path, "relation2id.txt"), "w", encoding='utf-8') as f:
        for k, v in relation2id.items():
            f.write(k + '\t' + str(v) + '\n')
    return id2relation


