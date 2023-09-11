from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from config import DEVICE
from gnn_layer import GraphAttentionLayer


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert_encoder = BertEncoder(configs)
        self.gnn = GraphNN(configs)
        self.pred = Pre_Predictions(configs)
        self.batch_size = configs.batch_size

    def forward(self, query, query_mask, query_seg, query_len, clause_len, doc_len, adj, label):
        doc_sents_h = self.bert_encoder(query, query_mask, query_seg, query_len, clause_len, doc_len)
        doc_sents_h = self.gnn(doc_sents_h, doc_len, adj)
        pred = self.pred(doc_sents_h)
        return pred

    def loss_pre(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        criterion = nn.BCELoss()
        return criterion(pred, true)

class BertEncoder(nn.Module):
    def __init__(self, configs):
        super(BertEncoder, self).__init__()
        hidden_size = configs.feat_dim
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(768, 1)

    def forward(self, discourse, discourse_mask, segment_mask, query_len, clause_len, doc_len):
        hidden_states = self.bert(input_ids=discourse.to(DEVICE),
                                  attention_mask=discourse_mask.to(DEVICE),
                                  token_type_ids=segment_mask.to(DEVICE))[0]
        hidden_states, mask_doc = self.get_sentence_state(hidden_states, query_len, clause_len, doc_len)

        alpha = self.fc(hidden_states).squeeze(-1)  # bs, max_doc_len, max_seq_len
        mask_doc = 1 - mask_doc # bs, max_doc_len, max_seq_len
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2) # bs, max_doc_len, feat_dim

        return hidden_states.to(DEVICE)

    def get_sentence_state(self, hidden_states, query_lens, clause_lens, doc_len):
        # 对每个句子的token做注意力，得到每个句子的向量表示
        sentence_state_all = []
        mask_all = []
        max_clause_len = 0

        for clause_len in clause_lens: # 找出最长的一句话包含多少token
            for l in clause_len:
                max_clause_len = max(max_clause_len, l)

        max_doc_len = max(doc_len) # 最长的文档包含多少句子
        for i in range(hidden_states.size(0)):  # 对每个batch
            # 对文档sentence
            mask = []
            begin = 0
            sentence_state = []
            for clause_len in clause_lens[i]:
                sentence = hidden_states[i, begin: begin + clause_len]
                begin += clause_len
                if sentence.size(0) < max_clause_len:
                    sentence = torch.cat([sentence, torch.zeros((max_clause_len - clause_len, sentence.size(-1))).to(DEVICE)],
                                         dim=0)
                sentence_state.append(sentence.unsqueeze(0))
                mask.append([1] * clause_len + [0] * (max_clause_len - clause_len))
            sentence_state = torch.cat(sentence_state, dim=0).to(DEVICE)
            if sentence_state.size(0) < max_doc_len:
                mask.extend([[0] * max_clause_len] * (max_doc_len - sentence_state.size(0)))
                padding = torch.zeros(
                    (max_doc_len - sentence_state.size(0), sentence_state.size(-2), sentence_state.size(-1)))
                sentence_state = torch.cat([sentence_state, padding.to(DEVICE)], dim=0)
            sentence_state_all.append(sentence_state.unsqueeze(0))
            mask_all.append(mask)
        sentence_state_all = torch.cat(sentence_state_all, dim=0).to(DEVICE)
        mask_all = torch.tensor(mask_all).to(DEVICE)
        return sentence_state_all, mask_all


class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)
        return doc_sents_h

class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = 768
        self.num_classes = configs.num_classes
        
        # relevant/irrelevant prediction
        self.out_01x2 = nn.Linear(self.feat_dim * 6, 2)
        
        # true/false prediction
        self.out_0x1 = nn.Linear(self.feat_dim * 6, 2)
        
        # overall prediction
        self.out_0x1x2 = nn.Linear(self.feat_dim * 6, self.num_classes)
    
    # overall prediction
    def forward(self, doc_sents_h):
        # shape: bs, feat_dim, max_doc_len
        final_h = doc_sents_h.reshape([-1, self.feat_dim * 6])
        pred = self.out_0x1x2(final_h)  # bs, categories
        pred = F.softmax(pred, dim=1)
        return pred  # shape: bs, categories