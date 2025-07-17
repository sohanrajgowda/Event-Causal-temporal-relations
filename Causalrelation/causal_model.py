import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv
from transformers import BertModel

class HGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HGATLayer, self).__init__()
        self.hgat = HypergraphConv(in_channels, out_channels)

    def forward(self, x, hyperedge_index):
        return self.hgat(x, hyperedge_index)

class HGATCausalClassifier(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', hidden_dim=256, out_dim=2):
        super(HGATCausalClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.hgat = HGATLayer(768, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, input_ids, attention_mask, hyperedge_index):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_tokens = outputs.last_hidden_state.squeeze(0) # Use [CLS] token embeddings
        print(cls_tokens)
        
        hgat_out = self.hgat(cls_tokens, hyperedge_index)
        hgat_out = hgat_out.mean(dim=0)
        return self.classifier(hgat_out)
