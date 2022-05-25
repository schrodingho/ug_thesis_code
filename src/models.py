# model for undergraduate paper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MedRec(nn.Module):
    def __init__(self, vocab_size, enc, emb_dim=64, device=torch.device('cuda:0')):
        super(MedRec, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.4)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.enc=enc
        self.inter = nn.Parameter(torch.FloatTensor(1))##？？？？
        self.gate_trans=nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim,1),
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim+vocab_size[2], emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )
        self.init_weights()

    def forward(self, input, patient_step):
        i1_seq = []
        i2_seq = []
        i3=[]
        def mean_embedding(embedding):
            #返回(1,1,dim)形状的embedding
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)  #

        embeds = self.enc([patient_step]).t()
        embeds = torch.mul(embeds, (embeds > 0.5).float())
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)#得到一个时间点所有code的mean
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
            i3.append(embeds)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        )
        o2, h2 = self.encoders[1](
            i2_seq
        )

        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)
        query = queries[-1:] # (1,dim)最后一行

        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)]
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break#不要最后一个seq
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        if len(input) > 1:

            # query 进行linear维数变换与history_key相乘
            gate_h=torch.sigmoid(self.gate_trans(query))
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            # print("att_weight:",visit_weight)
            fact2 = visit_weight.mm(history_values) # (1, size)
            fact1=gate_h*fact2+(1-gate_h)*embeds
            # print("gate_para:",gate_h)
        else:

            fact1=embeds

        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1], dim=-1))
        return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)

