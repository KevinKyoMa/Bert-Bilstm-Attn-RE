import argparse
import json
import os.path
from transformers import BertTokenizer
import torch.cuda
from data_process import read_data,position_padding,ReDataset,read_only_bert_data,precess_only_bert_data,OnlyBertReDataset
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
import time
import random
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.bert_bilstm_attention import Bert_bilstm_attn
from models.bert_re import BERT_Classifier
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn

def evaluate(args,dataloader,model,id2relation,criterion,epoch):
    model.eval()
    pred_relation = []
    original_relation = []
    eval_loss = 0.0
    with torch.no_grad():
        for input_ids,  input_segment, input_mask, relation_ids, position1, position2 in tqdm(dataloader, desc="Evaluating"):
            if torch.cuda.is_available():
                input_ids = input_ids.to(args.cuda)
                input_mask = input_mask.to(args.cuda)
                input_segment = input_segment.to(args.cuda)
                relation_ids = relation_ids.to(args.cuda).squeeze(dim=-1)
                position1 = position1.to(args.cuda)
                position2 = position2.to(args.cuda)
                output = model(input_ids, position1, position2, input_segment, input_mask)
            assert len(output) == len(relation_ids)
            loss = criterion(output, relation_ids)
            eval_loss += loss
            outputs = F.softmax(output,dim=-1)  #dim=-1时， 是对某一维度的行进行softmax运算
            pred_relation.extend([id2relation[i] for i in outputs.argmax(dim=1).detach().cpu().numpy().tolist()])
            original_relation.extend([id2relation[i] for i in relation_ids.view(-1).detach().cpu().numpy().tolist()])
        avg_eval_loss = eval_loss / len(dataloader)
        logger.info(f"eval_loss:{avg_eval_loss}")
        metrics = Metrics(original_relation, pred_relation)
        result = metrics.report_scores(args,epoch)
        return result,metrics

def only_bert_evaluate(args,dataloader,model,id2relation,criterion,epoch):
    model.eval()
    pred_relation = []
    original_relation = []
    eval_loss = 0.0
    with torch.no_grad():
        for input_ids, input_segment, input_mask, relation_ids, in tqdm(dataloader, desc="Evaluating"):
            if torch.cuda.is_available():
                input_ids = input_ids.to(args.cuda)
                input_mask = input_mask.to(args.cuda)
                input_segment = input_segment.to(args.cuda)
                relation_ids = relation_ids.to(args.cuda).squeeze(dim=-1)
                output = model(input_ids, input_segment, input_mask)
            assert len(output) == len(relation_ids)
            loss = criterion(output, relation_ids)
            eval_loss += loss
            outputs = F.softmax(output,dim=-1)  #dim=-1时， 是对某一维度的行进行softmax运算
            pred_relation.extend([id2relation[i] for i in outputs.argmax(dim=1).detach().cpu().numpy().tolist()])
            original_relation.extend([id2relation[i] for i in relation_ids.view(-1).detach().cpu().numpy().tolist()])
        avg_eval_loss = eval_loss / len(dataloader)
        logger.info(f"eval_loss:{avg_eval_loss}")
        metrics = Metrics(original_relation, pred_relation)
        result = metrics.report_scores(args,epoch)
        return result,metrics

def test(args,model,id2relation):
    if not os.path.exists(os.path.join(args.output_data_dir,'test')):
        os.makedirs(os.path.join(args.output_data_dir,'test'))
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.embedding_path)

    for file in os.listdir(args.test_dir):
        with open(os.path.join(args.test_file,file),'r',encoding='utf-8') as f,\
            open(os.path.join(args.output_dir,'test',file),'w',encoding='utf-8') as fw:
            temp = f.readlines()
            temp = temp[:5000]
            for line in temp:
                dic = json.load(line)
                index1 = dic['text'].lower().index(dic['ent1'].lower())
                position1 = []
                index2 = dic['text'].lower().index(dic['ent2'].lower())
                position2 = []
                for i , word in enumerate(dic['text']):
                    position1.append(i - index1)
                    position2.append(i - index2)
                assert len(dic['text']) == len(position1) == len(position2), print('read data error')
                tokens = ['[CLS]'] + list(dic['text']) + ['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1]*len(dic['text'])
                input_segment = [0]*len(dic['text'])
                position1 = position_padding(position1,len(tokens),args.position_num)
                position2 = position_padding(position2, len(tokens), args.position_num)

                input_ids = torch.LongTensor(input_ids).unsqueeze(0)
                input_mask = torch.LongTensor(input_mask).unsqueeze(0)
                input_segment = torch.LongTensor(input_segment).unsqueeze(0)
                position1 = torch.LongTensor(position1).unsqueeze(0)
                position2 = torch.LongTensor(position2).unsqueeze(0)

                if torch.cuda.is_available():
                    input_ids = input_ids.to('cuda')
                    input_mask = input_mask.to('cuda')
                    input_segment = input_segment.to('cuda')
                    position1 = position1.to('cuda')
                    position2 = position2.to('cuda')

                logits = model(input_ids,position1,position2,input_segment,input_mask)
                logits = F.softmax(logits)
                logit = logits.squeeze().detach().numpy()
                #去除维度为1的dim，将数据从网络中抽离，转位ndarry格式
                pred_index = logit.argmax()  #返回最大数的索引
                prob = logit[pred_index]  #预测为该种关系的可能性
                pred_label = id2relation[pred_index]
                if prob > 0.8:
                    fw.write(dic['ent1'] + '\t' + str(pred_label) + '\t' + dic['ent2'] + '\t' + dic['text'] + '\n')

def test_line(args,model,id2relation):
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.embedding_path)
    line = input('请输入句子:实体1，实体2，句子')
    line = line.strip().split()
    assert len(line) == 3,print('输入错误')
    index1 = line[2].index(line[0])
    position1 = []
    index2 = line[2].lower().index(line[1].lower())
    position2 = []
    for i,word in enumerate(line[2]):
        position1.append(i-index1)
        position2.append(i-index2)
    assert len(line[2]) == len(position1) == len(position2)
    tokens = ['[CLS]'] + list(line[2]) + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(tokens)
    input_segment = [0] * len(tokens)
    position1 = position_padding(position1, len(tokens), args.position_num)
    position2 = position_padding(position2, len(tokens), args.position_num)
    if torch.cuda.is_available():
        input_ids = torch.LongTensor(input_ids).to('cuda')
        input_ids = input_ids.reshape((1,-1))
        input_mask = torch.LongTensor(input_mask).to('cuda')
        input_mask = input_mask.reshape((1,-1))
        input_segment = torch.LongTensor(input_segment).to('cuda')
        input_segment = input_segment.reshape((1,-1))
        position1 = torch.LongTensor(position1).to('cuda')
        position1 = position1.reshape((1,-1))
        position2 = torch.LongTensor(position2).to('cuda')
        position2 = position2.reshape((1,-1))
    logits = model(input_ids, position1, position2, input_segment, input_mask)
    logits = F.softmax(logits, dim=-1)
    logit = logits.cpu().squeeze().detach().numpy()
    pred_index = logit.argmax()
    prob = logit[pred_index]
    pred_label = id2relation[pred_index]
    print("关系为： " + str(pred_label) + '\n' + "概率为：{}".format(prob))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='./data/train.json', type=str)
    parser.add_argument("--valid_file", default='./data/dev.json', type=str)
    parser.add_argument("--test_dir", default='./data/relation_input', type=str)
    parser.add_argument("--embedding_path", default='./pretrained/bert-base-chinese', type=str)
    parser.add_argument("--output_dir", default='./out_put', type=str)
    parser.add_argument("--save_model", default='./save_models', type=str)
    parser.add_argument("--use_bert_bilstm_attn", default=True)
    parser.add_argument("--do_train", default=True)
    parser.add_argument("--do_valid", default=True)
    parser.add_argument("--do_test", default=False)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--position_num", default=60, type=int)
    parser.add_argument("--position_dim", default=5, type=int)
    parser.add_argument("--rnn_dim", default=250, type=int)
    parser.add_argument("--epoch", default=15, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--bert_dim", default=768, type=int)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--dropout_prob", default=0.5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--label_balance", default=False)
    parser.add_argument("--clean", default=True, help="clean the output dir")
    parser.add_argument("--cuda", default='cuda:1')
    parser.add_argument("--train_data_num", default=45000)
    parser.add_argument("--valid_data_num", default=4000)
    parser.add_argument("--log_dir", default='./logs')
    args = parser.parse_args()

    set_seed(args.seed) #设置随机种子
    args.output_data_dir = args.output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if args.clean:
            if os.path.exists(args.output_dir):
                args.output_data_dir = args.output_dir
                print(f"清理输出文件夹：{args.output_dir} ")
                try:
                    del_file(args.output_dir)
                except Exception as e:
                    print(e)
                    print('please remove the files of output dir')
                    exit(-1)
        else:
            time_format = time.strftime('%Y-%m-%d-%H-%M-%S')
            args.output_data_dir = args.output_dir
            args.output_dir = os.path.join(args.output_dir, time_format)
            os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, "eval")):
        os.makedirs(os.path.join(args.output_dir, "eval"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)# Log等级总开关
    stream_handler = logging.StreamHandler()# 创建handler，用于输出到控制台、写入日志文件
    log_file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'train.log'), encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s : ")
    # 定义handler的输出格式
    stream_handler.setFormatter(formatter)
    log_file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(log_file_handler)

    if args.do_train:  #训练过程中加载关系表的判断
        if not os.path.exists(os.path.join(args.output_data_dir,"id2relation.pkl")):
            id2relation = get_re_label_dict(args.train_file, args.output_data_dir)
        else:
            with open(os.path.join(args.output_data_dir,"id2relation.pkl"),'rb') as f:
                id2relation = pickle.load(f)

    if args.do_test:   #测试过程中加载关系表的判断
        if not os.path.exists(os.path.join(args.output_data_dir,'id2relation.pkl')):
            print('no file id2label.pkl')
            exit(0)
        else:
            with open(os.path.join(args.output_data_dir, "id2relation.pkl"), "rb") as f:
                id2relation = pickle.load(f)

    args.relation_num = len(id2relation)  #自动获取关系的数量
    if args.do_train:
        if args.use_bert_bilstm_attn:
            valid_re_dataset = ReDataset(args,mode='valid')
            valid_re_dataloader = DataLoader(valid_re_dataset,shuffle=False,batch_size=args.batch_size)
        else:
            valid_re_dataset = OnlyBertReDataset(args, mode='valid')
            valid_re_dataloader = DataLoader(valid_re_dataset, shuffle=False, batch_size=args.batch_size)

    if args.do_train:
        writer = SummaryWriter(logdir=os.path.join(args.log_dir, "eval"), comment="Linear")
        #SummaryWriter类可以在指定文件夹生成一个事件文件，这个事件文件可以对TensorBoard解析
        if args.use_bert_bilstm_attn:
            model = Bert_bilstm_attn(args) #使用Bert_bilstm_attn训练
        else:
            model = BERT_Classifier(args)  #使用BERT_Classifier训练
        if torch.cuda.is_available():
            model.to(args.cuda)
        no_decay = ['bias','LayerNorm.weight'] # 不更新权重
        optimizer_ground_parameters=[
            {
                'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay':0.01
            },
            {
                'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }]
        if args.use_bert_bilstm_attn:
            train_re_dataset = ReDataset(args,mode='train')
        else:
            train_re_dataset = OnlyBertReDataset(args, mode='train')
        train_re_dataloader = DataLoader(train_re_dataset, shuffle=True, batch_size=args.batch_size)

        t_total = len(train_re_dataloader) // args.gradient_accumulation_steps * args.epoch

        optimizer = AdamW(optimizer_ground_parameters,lr=args.learning_rate,eps=args.adam_epsilon)  #一般使用adam
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.003, weight_decay=0)    #使用sgd可以采用较大的学习率
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        criterion = nn.CrossEntropyLoss()

        # 开始训练!
        logger.info("===== Running training  =====")
        logger.info("  Num Epochs = %d", args.epoch)

        global_step = 0
        best_f1 = 0.0
        for epoch in range(args.epoch):
            logger.info("====Train Epoch: %d ====", epoch)
            model.train()
            with tqdm(train_re_dataloader,desc=f"epoch {epoch}/{args.epoch}") as iteration:
                for batch in iteration:
                    if args.use_bert_bilstm_attn:
                        input_ids,input_segment,input_mask,relation_ids,position1, position2 = batch
                        if torch.cuda.is_available():
                            input_ids = input_ids.to(args.cuda)
                            input_segment = input_segment.to(args.cuda)
                            input_mask = input_mask.to(args.cuda)
                            relation_ids = relation_ids.to(args.cuda)
                            position1 = position1.to(args.cuda)
                            position2 = position2.to(args.cuda)
                        output = model(input_ids, position1, position2, input_segment, input_mask)
                    else:
                        input_ids, input_segment, input_mask, relation_ids = batch
                        if torch.cuda.is_available():
                            input_ids = input_ids.to(args.cuda)
                            input_segment = input_segment.to(args.cuda)
                            input_mask = input_mask.to(args.cuda)
                            relation_ids = relation_ids.to(args.cuda)
                        output = model(input_ids, input_segment, input_mask)
                    relation_ids = relation_ids.squeeze(dim=-1)
                    loss = criterion(output,relation_ids)
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                    writer.add_scalar('train/loss', loss, global_step)
                    if global_step % args.logging_steps == 0:
                        logger.info(f" train loss = {loss}")

            if args.do_valid:
                logger.info("=====  Running valid  =====")
                logger.info(f" Num examples = {len(valid_re_dataset)}")
                logger.info(f" Batch size = {args.batch_size}")
                if args.use_bert_bilstm_attn:
                    result,metrics = evaluate(args,valid_re_dataloader, model, id2relation,criterion,epoch)
                else:
                    result, metrics = only_bert_evaluate(args, valid_re_dataloader, model, id2relation, criterion, epoch)
                f1_score = result['f1_score']

                writer.add_scalar("Valid/precision", result['precision'], epoch)
                writer.add_scalar("Valid/recall", result['recall'], epoch)
                writer.add_scalar("Valid/f1_score", result['f1_score'], epoch)

                #save the best model
                if f1_score > best_f1:
                    logger.info(f"----------the best f1 is {f1_score}---------")
                    best_f1 = f1_score
                    torch.save(model,os.path.join(args.save_model,'model.bin'))
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

                if epoch == args.epoch - 1:
                    metrics.report_confusion_matrix()
        writer.close()

    if args.do_test:
        model =torch.load(os.path.join(args.save_model,'model.bin'))
        test_line(args,model,id2relation)
        test(args, model, id2relation)


if __name__ == "__main__":
    main()


#  tensorboard --logdir=D:\practice_code\RE\Bert-Bilstm-attn\logs\eval   #tensorboard代码

