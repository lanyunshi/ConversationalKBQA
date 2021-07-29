from library.BertModel import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BertEncoderLayer(nn.Module):
    """Construct a 'Recurrent' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(BertEncoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.padding_id = 0
        self.device = device

    def forward(self, sequence):
        b, gn, sl = sequence.size()
        sequence = sequence.view(b*gn, sl) # (batch*gnum, glen)

        sequence_mask = torch.eq(sequence, self.padding_id).type(torch.FloatTensor)
        sequence_mask = sequence_mask.to(self.device) if self.device else sequence_mask
        sequence_mask = sequence_mask.unsqueeze(1).unsqueeze(2)
        sequence_mask = -1.e10 * sequence_mask

        sequence_emb = self.embeddings(sequence) # (batch, slen, hidden_size)
        sequence_enc = self.encoder(sequence_emb,
                                    sequence_mask,
                                    output_all_encoded_layers = False)[-1]
        sequence_pool = self.pooler(sequence_enc, mode = 'FirstPool').view(b, gn, self.hidden_size) # (batch+batch*gnum, hidden_size)

        return sequence_enc

class SimpleEmbLayer(nn.Module):
    """Construct a 'Recurrent' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(SimpleEmbLayer, self).__init__()
        self.hidden_size = 300 #config.hidden_size
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)
        self.padding_id = 0
        self.device = device

    def forward(self, sequence):
        if len(sequence.size()) == 3:
            b, gn, sl = sequence.size()
            sequence = sequence.view(b*gn, sl) # (batch*gnum, glen)

        sequence_enc = self.embeddings(sequence) # (batch, slen, hidden_size)
        #print('sequence_enc', sequence_enc[0, :3, 0])
        return sequence_enc

class CrossAtten(nn.Module):
    def __init__(self, config, device=None):
        super(CrossAtten, self).__init__()
        self.device = device
        self.hidden_size = config.hidden_size
        self.question_encoder = nn.LSTM(2*self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.subgraph_encoder = nn.LSTM(2*self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.sim = nn.Linear(4*self.hidden_size, 1)

    def forward(self, question_emb, subgraph_emb, mask):
        #print(subgraph_emb.size(), question_emb.size())
        b, ql, _ = question_emb.size()
        b, gn, gl, _ = subgraph_emb.size()
        subgraph_emb = subgraph_emb.view(b, gn*gl, -1)
        attention = torch.bmm(subgraph_emb, question_emb.transpose(2, 1)) # (batch, gnum*glen, hidden_size) * (batch, hidden_size, qlen) --> (batch, gnum*glen, qlen)
        mask_value = -1.e10 * mask
        attention = attention + mask_value # (batch, gnum, glen, qlen)
        #print('attention', mask[0, :3, :], attention[0, :3, :]); exit()

        atten_question = F.softmax(attention.view(b, gn*gl, ql), 2) # attentions along questions
        align_question = torch.bmm(atten_question, question_emb).view(b*gn, gl, self.hidden_size) # (batch, gnum*glen, hidden_size)
        atten_subgraph = F.softmax(attention.view(b*gn, gl, ql), 1) # attention along subgraphs
        align_subgraph = torch.bmm(atten_subgraph.transpose(2, 1), subgraph_emb.view(b*gn, gl, self.hidden_size)).view(b*gn, ql, self.hidden_size)

        compa_question = torch.max(self.question_encoder(torch.cat([align_question, subgraph_emb.view(b*gn, gl, self.hidden_size)], 2))[0], 1)[0] # (batch*gnum, 2*hidden_size)
        compa_subgraph = torch.max(self.subgraph_encoder(torch.cat([align_subgraph, question_emb.unsqueeze(1).repeat(1, gn, 1, 1).view(b*gn, ql, self.hidden_size)], 2))[0], 1)[0] # (batch*gnum, 2*hidden_size)
        # compa_question = torch.mean(self.question_encoder(torch.cat([align_question, subgraph_emb.view(b*gn, gl, self.hidden_size)], 2))[0], 1) # (batch*gnum, 2*hidden_size)
        # compa_subgraph = torch.mean(self.subgraph_encoder(torch.cat([align_subgraph, question_emb.unsqueeze(1).repeat(1, gn, 1, 1).view(b*gn, ql, self.hidden_size)], 2))[0], 1) # (batch*gnum, 2*hidden_size)
        sequence_sim = self.sim(torch.cat([compa_question, compa_subgraph], 1)).view(b, gn, 1) #F.cosine_similarity(compa_question, compa_subgraph).view(b, gn, 1)
        return sequence_sim

class LinkSpecificGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, link_num, bias=True):
        super(LinkSpecificGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.link_num = link_num
        self.link_weight = nn.Parameter(torch.FloatTensor(link_num, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.link_weight.size(1))
        self.link_weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, hidden):
        #print('forward')
        # nodes --> (node_num, hidden)
        # adjacent_matrix --> (node_num, node_num, link_num)
        # hidden --> (hidden, 1)
        #print('input size', input.size(), adj.size(), hidden.size())
        node_num, h = input.size()
        input = input.view(node_num, h)
        adj = adj.view(node_num*node_num, -1)
        hid = torch.ones_like(hidden.squeeze(0).unsqueeze(1))
        support = torch.mm(self.link_weight, hid) # (link_num, 1)
        #print(adj, support)
        support = torch.mm(adj, support).squeeze(1) #(node_num, h)
        support = F.softmax(support.view(node_num, node_num), 1) #(node_num, node_num)
        #print(support[0, :])
        output = torch.mm(support, input)
        return output

class LinkSpecificGraphConvolution_v2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, link_num, bias=True):
        super(LinkSpecificGraphConvolution_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.link_num = link_num
        self.link_weight = nn.Parameter(torch.FloatTensor(link_num, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.link_weight.size(1))
        self.link_weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, hidden):
        # nodes --> (node_num, hidden)
        # adjacent_matrix --> (node_num, node_num, link_num)
        # hidden --> (hidden, 1)
        #print('input size', input.size(), adj.size(), hidden.size())
        node_num, h = input.size()
        input = input.view(node_num, h)
        adj = adj.view(node_num*node_num, -1)
        support = torch.mm(self.link_weight, hidden.squeeze(0).unsqueeze(1)) # (link_num, 1)
        #print(adj, support)
        support = torch.mm(adj, support).squeeze(1) #(node_num, h)
        support = F.softmax(support.view(node_num, node_num), 1) #(node_num, node_num)
        #print(support[0, :])
        output = torch.mm(support, input)
        return output

class DynamicGCN(nn.Module):
    """Construct a 'Recurrent' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(DynamicGCN, self).__init__()
        #self.gcn = LinkSpecificGraphConvolution(config.hidden_size, config.hidden_size, 3)
        self.gcn = LinkSpecificGraphConvolution_v2(config.hidden_size, config.hidden_size, 3)

    def forward(self, nodes, adjacent_matrix, hidden):
        # nodes --> (node_num, hidden)
        # adjacent_matrix --> (node_num, node_num, link_num)
        #print('nodes', nodes.size(), adjacent_matrix.size())
        x = self.gcn(nodes, adjacent_matrix, hidden)
        #print('x', x.size())
        x = F.relu(x)
        return x

class SimplePooler(nn.Module):
    def __init__(self, config):
        super(SimplePooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, end_idx=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        b, ql, h = hidden_states.size()
        hidden_states = hidden_states.view(b, ql, 2, int(h / 2))
        # print('hidden_states', hidden_states.size())
        if end_idx is None:
            first_token_tensor = torch.cat([hidden_states[:, 0, 0, :], hidden_states[:, -1, 1, :]], -1)
        else:
            first_token_tensor = torch.cat([hidden_states[range(b), end_idx, 0, :], hidden_states[:, 0, 1, :]], -1)
        pooled_output = first_token_tensor
        #print('pooled_output', pooled_output.size())
        return pooled_output