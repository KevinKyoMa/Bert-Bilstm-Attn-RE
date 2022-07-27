from transformers import BertModel
import torch.nn as nn
import torch

class BERT_Classifier(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.pos_dim = args.position_dim
        self.encoder = BertModel.from_pretrained(args.embedding_path)
        self.dropout = nn.Dropout(args.dropout_prob,inplace=False)
        self.fc = nn.Linear(768, args.relation_num)

    def forward(self, input_ids,token_type_ids=None ,input_mask=None):
        x = self.encoder(input_ids, token_type_ids=None,attention_mask=input_mask)[0]
        x = x[:,0,:]
        # batch_size, seq_len, 768
        # batch_size, 768
        x = self.dropout(x)
        # batch_size, 778
        x = self.fc(x)
        # batch_size, relation_num
        return x