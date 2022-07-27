import torch
from transformers import BertModel
from torch import nn
from typing import Text, Any, Dict, Optional, List, Union, Tuple, Type
import torch.nn.functional as F

class Bert_bilstm_attn(nn.Module):
    def __init__(self,args):
        super(Bert_bilstm_attn, self).__init__()
        self.batch = args.batch_size
        self.rnn_dim = args.rnn_dim
        self.hidden_size = args.rnn_dim*2
        self.relation_num = args.relation_num  #关系分类的数量

        self.pos_size = (args.position_num+1) *2 +1 # pos标记用到了 0 - args.position_num * 2 + 1个数据
        self.pos_dim = args.position_dim

        self.bert = BertModel.from_pretrained(args.embedding_path) #加载已经训练好的bert模型

        self.pos1_embedding = nn.Embedding(self.pos_size,self.pos_dim) # 实体1的embedding
        self.pos2_embedding = nn.Embedding(self.pos_size, self.pos_dim) # 实体2的embedding

        self.lstm = nn.LSTM(
            input_size=args.bert_dim + self.pos_dim * 2,
            hidden_size=self.rnn_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.dropoutlstm = nn.Dropout(args.dropout_prob)

        self.att_weight = nn.Parameter(torch.randn(self.hidden_size,1)) #[self.hidden_size,1]
        self.relation_bias = nn.Parameter(torch.randn(self.relation_num,1))  #[self.relation_num,1]
        self.relation_embedding = nn.Parameter(torch.randn(self.relation_num,self.hidden_size))

    def attn(self,H):
        M = torch.tanh(H)  # 非线性变换 size:(batch_size,seq_len, hidden_dim)
        a = F.softmax(torch.matmul(M,self.att_weight),dim=1)# a.Size : (batch_size,seq_len, 1),注意力权重矩阵
        a = torch.transpose(a,1,2)  # (batch_size,1, seq_len)
        return torch.bmm(a,H)  # (batch_size,1,hidden_dim) #权重矩阵对输入向量进行加权计算

    def forward(self,input_ids,pos1,pos2,token_type_ids=None,input_mask=None):
        batch_size = input_ids.shape[0]
        bert_output = self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=input_mask)
        embeds = torch.cat((bert_output[0],self.pos1_embedding(pos1),self.pos2_embedding(pos2)),dim=2)
        # batch_size, seq_len, 768 + 2 * pos_dim
        lstm_out, _ = self.lstm(embeds)
        # batch_size,seq_len,hidden_dim
        lstm_out = self.dropoutlstm(lstm_out)
        # batch_size,seq_len,hidden_dim
        att_out = torch.tanh(self.attn(lstm_out))
        # batch_size,1,hidden_dim
        att_out = torch.reshape(att_out,[batch_size,self.hidden_size])
        # batch_size, hidden_size
        att_out = torch.transpose(att_out,0,1)
        # hidden_size, batch_size
        out = torch.add(torch.matmul(self.relation_embedding,att_out),self.relation_bias)
        # relation_num, batch_size
        out = torch.transpose(out,0,1)
        # batch_size, relation_num
        return out





