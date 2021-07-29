import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from library.BertModel import *
from library.advanced_layer import BertEncoderLayer, CrossAtten, SimpleEmbLayer, DynamicGCN, SimplePooler

import numpy as np
import json
import zipfile
import os
import copy

eps = np.finfo(np.float32).eps.item()

class ModelConfig(object):
    """Configuration class to store the configuration of a 'Model'
    """
    def __init__(self,
                vocab_size_or_config_json_file,
                hidden_size = 200,
                dropout_prob = 0.1,
                initializer_range= 0.02):

        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.dropout_prob = dropout_prob
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """COnstruct a 'Config' from a Python dictionary of parameters."""
        config = ModelConfig(vocab_size_or_config_json_file = -1)
        for key, value in json_object.items():
            config.__dict__[key]=value
        return config
    @classmethod
    def from_json_file(cls, json_file):
        """Construct a 'Config' from a json file of parameters"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class SimpleRanker(nn.Module):
    """Construct a 'Recurrent' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(SimpleRanker, self).__init__()
        self.hidden_size = config.hidden_size
        self.use_bert = config.use_bert
        self.use_match = config.use_match
        if config.use_bert:
            self.encoder = BertEncoderLayer(config, device)
            self.pooler = BertPooler(config)
        else:
            self.embedder = SimpleEmbLayer(config, device)
            self.q_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.g_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.pooler = SimplePooler(config)
        if config.use_match:
            self.cross_atten = CrossAtten(config, device)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.decoder = nn.Linear(2, 1)
        self.device = device
        self.padding_id = 0
        self.cls_id = 101
        self.sep_id = 102

    def encode_qs(self, question, subgraph, answer):
        if self.use_bert:
            sequence = torch.cat([question, subgraph, answer], 1)
            sequence_enc = self.encoder(sequence) # (batch+batch*gnum, hidden_size)
            # print('sequence_enc.size()', sequence_enc.size()); exit()
            sn, _, _ = sequence_enc.size()
            gn = int((sn - 1) / 2)
            question_enc, subgraph_enc, answer_enc = sequence_enc[:1], sequence_enc[1:-gn], sequence_enc[-gn:]

            question = self.pooler(question_enc, 'MaxPool')
            subgraph = self.pooler(subgraph_enc, 'MaxPool').unsqueeze(0)
            answer = self.pooler(answer_enc, 'MaxPool')
        else:
            #print('torch.eq(question, self.padding_id)', torch.eq(question, self.padding_id))
            qidx = torch.sum(1 - torch.eq(question, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            question_enc = self.embedder(question)
            question_enc, _ = self.q_encoder(question_enc)
            #print('question_enc', question_enc.size())
            sequence = torch.cat([subgraph, answer], 1)
            cidx = torch.sum(1 - torch.eq(sequence, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            #print('qidx, sidx', qidx.size(), sidx.size()); exit()
            sequence_enc = self.embedder(sequence)
            sequence_enc, _ = self.g_encoder(sequence_enc)
            sn, _, _ = sequence_enc.size()
            gn = int(sn / 2)
            subgraph_enc, answer_enc = sequence_enc[:-gn], sequence_enc[-gn:]
            sidx, aidx = cidx[:-gn], cidx[-gn:]

            question = self.pooler(question_enc, qidx)
            #print('before question', subgraph.size())
            subgraph = self.pooler(subgraph_enc, sidx).unsqueeze(0)
            answer = self.pooler(answer_enc, aidx)
            #print('after sequence_enc', subgraph.size())
        #question, subgraph = question[:, -1, :], subgraph[:, -1, :].unsqueeze(0)
        return question_enc, subgraph_enc, answer_enc, question, subgraph, answer

    def forward(self, batch, time):
        question, subgraph, ts_score, answer = batch
        question = question[:, :1, :]
        #question_mask = torch.eq(question[:, 0, :], self.padding_id)
        #print(question.size(), subgraph.size(), answer.size())
        b, _, sl = question.size()

        #sequence_pool = self.dropout(sequence_pool)
        #print(sequence_enc.size()); exit()
        question_enc, subgraph_enc, _, question_vec, subgraph_vec, answer_vec = self.encode_qs(question, subgraph, answer)
        self.answer = answer_vec
        _, gn, _ = subgraph.size()

        if self.use_match:
            question_mask = 1 - torch.eq(question, self.padding_id).type(torch.FloatTensor).view(b, sl, 1)
            subgraph_mask = 1 - torch.eq(subgraph, self.padding_id).type(torch.FloatTensor).view(b, gn*sl, 1)
            mask = 1 - torch.bmm(subgraph_mask, question_mask.transpose(2, 1)) # (batch, gnum*glen, 1) * (batch, 1, qlen) --> (batch, gnum*glen, qlen)
            mask = mask.to(self.device) if self.device else mask

            sequence_sim = self.cross_atten(question_enc.contiguous().view(b, sl, -1), subgraph_enc.contiguous().view(b, gn, sl, -1), mask=mask) #(batch, sl, 1, hidden) (batch, sl, gn, hidden)
        else:
            sequence_sim = torch.bmm(subgraph_vec, question_vec.unsqueeze(0).transpose(2, 1)) # hidden[0]

        features = torch.cat([ts_score[:, :gn].view(b, gn, 1), sequence_sim], 2)
        features = self.dropout(features)
        #print(features)

        logits = self.decoder(features).view(b, gn)
        #print(logits)

        return logits, 0

class SimpleRecurrentRanker(nn.Module):
    """Construct a 'Recurrent' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(SimpleRecurrentRanker, self).__init__()
        self.hidden_size = config.hidden_size
        self.use_bert = config.use_bert
        self.use_te_graph = config.use_te_graph
        if config.use_bert:
            self.encoder = BertEncoderLayer(config, device)
            self.pooler = BertPooler(config)
        else:
            self.embedder = SimpleEmbLayer(config, device)
            self.q_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.g_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.pooler = SimplePooler(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.start_hidden = (torch.randn(1, self.hidden_size).to(device), torch.randn(1, self.hidden_size).to(device))
        self.topic_controller = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.te_predictor = nn.Sequential(nn.Linear(self.hidden_size, 4), nn.Tanh())
        self.decoder = nn.Linear(2, 1)
        self.device = device
        self.padding_id = 0
        self.cls_id = 101
        self.sep_id = 102

    def initial_hidden_state(self):
        self.hidden = [self.start_hidden]

    def update_answer(self, idx):
        #print('self.answer', self.answer.size(), idx)
        self.last_answer = self.answer[idx[0, 0], :].unsqueeze(0)

    def encode_qs(self, question, subgraph, answer):
        if self.use_bert:
            sequence = torch.cat([question, subgraph, answer], 1)
            sequence_enc = self.encoder(sequence) # (batch+batch*gnum, hidden_size)
            # print('sequence_enc.size()', sequence_enc.size()); exit()
            sn, _, _ = sequence_enc.size()
            gn = int((sn - 1) / 2)
            question_enc, subgraph_enc, answer_enc = sequence_enc[:1], sequence_enc[1:-gn], sequence_enc[-gn:]

            question = self.pooler(question_enc, 'MaxPool')
            subgraph = self.pooler(subgraph_enc, 'MaxPool').unsqueeze(0)
            answer = self.pooler(answer_enc, 'MaxPool')
        else:
            qidx = torch.sum(1 - torch.eq(question, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            question_enc = self.embedder(question)
            question_enc, _ = self.q_encoder(question_enc)
            #print('question_enc', question_enc.size())
            sequence = torch.cat([subgraph, answer], 1)
            cidx = torch.sum(1 - torch.eq(sequence, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            sequence_enc = self.embedder(sequence)
            sequence_enc, _ = self.g_encoder(sequence_enc)
            sn, _, _ = sequence_enc.size()
            gn = int(sn / 2)
            subgraph_enc, answer_enc = sequence_enc[:-gn], sequence_enc[-gn:]
            sidx, aidx = cidx[:-gn], cidx[-gn:]

            question = self.pooler(question_enc, qidx)
            #print('before question', question.size())
            subgraph = self.pooler(subgraph_enc, sidx).unsqueeze(0)
            answer = self.pooler(answer_enc, aidx)
            #print('after sequence_enc', question.size())
        #question, subgraph = question[:, -1, :], subgraph[:, -1, :].unsqueeze(0)
        return question_enc, subgraph_enc, answer_enc, question, subgraph, answer

    def self_attention(self, question, hidden, question_mask):
        # question ---> (batch_size, sl, dim)
        # hidden ---> (batch_size, 1, dim)
        #print('question.size(), hidden.size()', question.size(), hidden.size())
        attention = torch.bmm(question, hidden.unsqueeze(0).transpose(2, 1)) # (batch_size, sl, 1)
        question_mask = -1.e14 * question_mask.unsqueeze(2)
        attention = nn.Softmax(dim=1)(attention + question_mask)
        question = torch.sum(attention * question, 1)
        return question

    def forward_answer(self):
        hidden = self.topic_controller(self.dropout(self.last_answer), self.hidden[-1])
        self.hidden += [hidden]

    def forward(self, batch, time):
        question, subgraph, ts_score, answer = batch
        question_mask = torch.eq(question[:, 0, :], self.padding_id)
        # print(question.size(), subgraph.size())
        b, _, sl = question.size()
        question = question[:, :1, :]
        #print('ts_nb, ty_nb, su_nb', ts_nb.size(), ty_nb.size(), su_nb.size())

        #sequence_pool = self.dropout(sequence_pool)
        #print(sequence_enc.size()); exit()
        question_enc, subgraph_enc, _, question_vec, subgraph_vec, answer_vec = self.encode_qs(question, subgraph, answer)
        self.answer = answer_vec
        _, gn, _ = subgraph.size()

        if time > -1:
            #print('question_vec', question_vec.size())
            question_vec = self.dropout(question_vec)
            hidden = self.topic_controller(question_vec, self.hidden[-1])
            self.hidden += [hidden]
            #question = self.self_attention(question_enc, hidden[0], question_mask)
            #question = hidden[0]
        #print('subgraph, question', subgraph[0, :3, :3], question[0, :3, :3])

        sequence_sim = torch.bmm(subgraph_vec, hidden[0].unsqueeze(0).transpose(2, 1)) # hidden[0]

        features = torch.cat([ts_score[:, :gn].view(b, gn, 1), sequence_sim], 2)
        features = self.dropout(features)
        #print(features)

        logits = self.decoder(features).view(b, gn)
        #print(logits)

        return logits, 0


class RecurrentRanker(nn.Module):
    """Construct a 'Recurrent' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(RecurrentRanker, self).__init__()
        self.hidden_size = config.hidden_size
        self.use_bert = config.use_bert
        self.use_te_graph = config.use_te_graph
        if config.use_bert:
            self.embedder = SimpleEmbLayer(config, device)
            self.encoder = BertEncoderLayer(config, device)
            self.q_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.g_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.q_encoder2 = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.g_encoder2 = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.bert_pooler = BertPooler(config)
        else:
            self.embedder = SimpleEmbLayer(config, device)
            self.q_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.g_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.q_encoder2 = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
            self.g_encoder2 = nn.LSTM(self.hidden_size, int(self.hidden_size/2), batch_first=True, bidirectional=True)
        self.pooler = SimplePooler(config)
        if config.use_te_graph==1:
            self.te_graph = nn.ModuleList([copy.deepcopy(DynamicGCN(config)) for _ in range(10)])
            self.node_decoder = nn.Linear(2*self.hidden_size+1, 1) #
        self.node_enc = []
        self.dropout = nn.Dropout(config.dropout_prob)
        self.start_hidden = (torch.randn(1, self.hidden_size).to(device), torch.randn(1, self.hidden_size).to(device))
        self.topic_controller = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(2, 1)
        self.device = device
        self.padding_id = 0
        self.cls_id = 101
        self.sep_id = 102

    def initial_hidden_state(self):
        self.hidden = [self.start_hidden]

    def update_answer(self, idx):
        #print('self.answer', self.answer.size(), idx)
        self.last_answer = self.answer[idx[0, 0], :].unsqueeze(0)

    def encode_qs(self, question, subgraph, answer):
        if self.use_bert:
            sequence = torch.cat([question, subgraph, answer], 1)
            #print('sequence', sequence[0, :, :5]); exit()
            sequence_enc = self.encoder(sequence) # (batch+batch*gnum, hidden_size)
            # print('sequence_enc.size()', sequence_enc.size()); exit()
            sn, _, _ = sequence_enc.size()
            gn = int((sn - 1) / 2)
            question_emb, subgraph_emb, answer_emb = sequence_enc[:1], sequence_enc[1:-gn], sequence_enc[-gn:]

            qidx = torch.sum(1 - torch.eq(question, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            sequence_emb = self.embedder(sequence)
            question_emb2, subans_emb2 = sequence_emb[:1], sequence_emb[1:]
            question_enc2, _ = self.q_encoder(question_emb2)
            #print('question_enc', question_enc.size())
            subgraph_answer = torch.cat([subgraph, answer], 1)
            cidx = torch.sum(1 - torch.eq(subgraph_answer, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            subans_enc2, _ = self.g_encoder(subans_emb2)
            subgraph_enc2, answer_enc2 = subans_enc2[:-gn], subans_enc2[-gn:]
            sidx, aidx = cidx[:-gn], cidx[-gn:]
            question_vec = self.bert_pooler(question_emb)
            subgraph_vec = self.bert_pooler(subgraph_emb).unsqueeze(0)
            answer_vec = self.bert_pooler(answer_emb)

        else:
            qidx = torch.sum(1 - torch.eq(question, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            question_emb = self.embedder(question)
            question_enc, _ = self.q_encoder(question_emb)
            question_enc2, _ = self.q_encoder2(question_emb)
            #print('question_enc', question_enc.size())
            sequence = torch.cat([subgraph, answer], 1)
            cidx = torch.sum(1 - torch.eq(sequence, self.padding_id).type(torch.LongTensor), 2).squeeze(0) - 1
            sequence_emb = self.embedder(sequence)
            sequence_enc, _ = self.g_encoder(sequence_emb)
            sequence_enc2, _ = self.g_encoder2(sequence_emb)
            sn, _, _ = sequence_enc.size()
            gn = int(sn / 2)
            subgraph_enc, answer_enc = sequence_enc[:-gn], sequence_enc[-gn:]
            subgraph_enc2, answer_enc2 = sequence_enc2[:-gn], sequence_enc2[-gn:]
            sidx, aidx = cidx[:-gn], cidx[-gn:]
            question_vec = self.pooler(question_enc, qidx)
            subgraph_vec = self.pooler(subgraph_enc, sidx).unsqueeze(0)
            answer_vec = self.pooler(answer_enc, aidx)

        question_vec2 = self.pooler(question_enc2, qidx)
        subgraph_vec2 = self.pooler(subgraph_enc2, sidx).unsqueeze(0)
        answer_vec2 = self.pooler(answer_enc2, aidx)
        #question, subgraph = question[:, -1, :], subgraph[:, -1, :].unsqueeze(0)
        return question_vec, subgraph_vec, answer_vec, question_vec2, subgraph_vec2, answer_vec2

    def self_attention(self, question, hidden, question_mask):
        # question ---> (batch_size, sl, dim)
        # hidden ---> (batch_size, 1, dim)
        #print('question.size(), hidden.size()', question.size(), hidden.size())
        attention = torch.bmm(question, hidden.unsqueeze(0).transpose(2, 1)) # (batch_size, sl, 1)
        question_mask = -1.e14 * question_mask.unsqueeze(2)
        attention = nn.Softmax(dim=1)(attention + question_mask)
        question = torch.sum(attention * question, 1)
        return question

    def te_co_graph(self, te_batch, time, do_te_graph=1):
        '''Entity '''
        # node --> (batch_size, sl)
        #print('t_co_graph', time)
        if time == 1: self.node_enc = []; #print('renew node_enc')
        node, adjacent_matrix, node_attr, node_out, node_type = te_batch
        #print('self.embedder(node)', node.size())
        if len(node):
            n, _ = node.size()
            #current_node = torch.mean(self.embedder(node), 1)
            current_node = torch.normal(0, 1, size=(n, self.hidden_size)).to(self.device)
        if len(self.node_enc) != 0 and len(node):
            #print('self.node_enc[-1]', self.node_enc[-1].size())
            node = torch.cat([self.node_enc[-1], current_node])
        elif len(self.node_enc) != 0:
            node = self.node_enc[-1]
        else:
            node = current_node
        #print('node, adjacent_matrix', node.size(), adjacent_matrix.size())
        adjacent_matrix_mask = 1 - torch.eq(adjacent_matrix, self.padding_id).type(torch.FloatTensor).to(self.device)
        #print('adjacent_matrix_mask', adjacent_matrix_mask)
        #print('node_type', node_type)
        if do_te_graph==1:
            node_enc = self.te_graph[time](node, adjacent_matrix_mask, self.hidden[-1][0])
            node_attr_vec = self.embedder(node_attr)
            node_type_vec = self.embedder(node_type)
            node_out = node_out.unsqueeze(-1)
            # print('torch.cat([node_enc, node_attr], 1)', node_enc.size(), node_attr_vec.size())
            node_vec = torch.cat([node_enc, node_attr_vec, node_out], 1)  #

        node_vec = self.dropout(node_vec)
        node_vec = self.node_decoder(node_vec)
        #node = F.log_softmax(node_vec, dim=0).squeeze(1)
        node = node_vec.squeeze(1)
        self.node_enc += [node_enc.clone()]
        return node

    def forward_answer(self):
        hidden = self.topic_controller(self.last_answer, self.hidden[-1])
        self.hidden += [hidden]

    def train_next_te(self, gold_te, te_batch, time):
        # hidden = self.topic_controller(self.last_answer, self.hidden[-1])
        # self.hidden += [hidden]
        _gold_te = torch.tensor(gold_te, dtype=torch.float).to(self.device)

        _raw_te_logits = self.te_co_graph(te_batch, time, do_te_graph=self.use_te_graph)
        _te_logits = F.softmax(_raw_te_logits, dim=-1)
        #print('te_logits.size()', _te_logits, _gold_te)
        _te_loss = torch.sum((_te_logits - _gold_te)**2)
        self.te_logits = te_logits = _te_logits.cpu().data.numpy()
        max_te = np.argmax(te_logits)
        #print('te_logits, max_te', te_logits, gold_te, max_te)
        return _te_loss, gold_te[max_te], max_te

    def predict_next_te(self, gold_te, te_batch, time):
        _gold_te = torch.tensor(gold_te, dtype=torch.float).to(self.device)

        _raw_te_logits = self.te_co_graph(te_batch, time, do_te_graph=self.use_te_graph)
        _te_logits = F.softmax(_raw_te_logits, dim=-1)

        self.te_logits = te_logits = _te_logits.cpu().data.numpy()
        max_te = np.argmax(te_logits)

        return 0, gold_te[max_te], max_te

    def forward(self, batch, time):
        question, subgraph, ts_score, answer = batch
        question_mask = torch.eq(question[:, 0, :], self.padding_id)

        b, _, sl = question.size()
        question = question[:, :1, :]

        question_vec2, subgraph_vec2, _, question_vec, subgraph_vec, answer_vec = self.encode_qs(question, subgraph, answer)
        self.answer = answer_vec
        _, gn, _ = subgraph.size()

        if time > -1:
            hidden = self.topic_controller(question_vec, self.hidden[-1])
            self.hidden += [hidden]

        sequence_sim = torch.bmm(subgraph_vec2, question_vec2.unsqueeze(0).transpose(2, 1)) # hidden[0]

        features = torch.cat([ts_score[:, :gn].view(b, gn, 1)*100, sequence_sim], 2) # (batch, gnum, 2*hidden_size+2) su_nb.view(b, gn, 1)
        features = self.dropout(features)

        logits = self.decoder(features).view(b, gn)
        #print(logits)

        return logits, 0


class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, vocab, *input, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config
        self.vocab = vocab

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.Embedding)) and self.config.hidden_size==300 and module.weight.data.size(0) > 1000:
            if os.path.exists(self.config.Word2vec_path):
                embedding = np.load(self.config.Word2vec_path)
                module.weight.data = torch.tensor(embedding, dtype=torch.float)
                print('pretrained GloVe embeddings')
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                gloves = zipfile.ZipFile('/home/yslan/Word2Vec/glove.840B.300d.zip')
                seen = 0

                for glove in gloves.infolist():
                    with gloves.open(glove) as f:
                        for line in f:
                            if line != "":
                                splitline = line.split()
                                word = splitline[0].decode('utf-8')
                                embedding = splitline[1:]

                            if word in self.vocab and len(embedding) == 300:
                                temp = np.array([float(val) for val in embedding])
                                module.weight.data[self.vocab[word], :] = torch.tensor(temp, dtype=torch.float)
                                seen += 1

                print('pretrianed vocab %s among %s' %(seen, len(self.vocab)))
                np.save(self.config.Word2vec_path, module.weight.data.numpy())
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class Policy(PreTrainedModel):
    def __init__(self, config, vocab, device):
        super(Policy, self).__init__(config, vocab, device=None)
        self.device = device
        if config.method == "SimpleRecurrentRanker":
            self.ranker = SimpleRecurrentRanker(config, device=device)
        elif config.method == "SimpleRanker":
            self.ranker = SimpleRanker(config, device=device)
        elif config.method == "RecurrentRanker":
            self.ranker = RecurrentRanker(config, device=device)
        self.method = config.method
        self.apply(self.init_bert_weights)

        self.gamma = config.gamma
        self.top_enc = None

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor()).to(self.device) if self.device else Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x, chunk_size = 15):
        gn = x[1].size(1)
        #print('gn', gn)
        if gn > chunk_size:
            chunk, logits, losses = int(np.ceil(gn*1./chunk_size)), [], 0
            #print('chunk', chunk)
            for i in range(chunk):
                new_x = tuple([xx[:, i*chunk_size: (i+1)*chunk_size] for xx in x])
                #print(new_x)
                logit, loss = self.ranker(new_x)
                logits += [logit]
                losses += loss
            logits = torch.cat(logits, 1)
            #print('logits', logits.size())
        else:
            logits, losses = self.ranker(x)
        return logits, losses

    def reset(self):
        self.policy_history = Variable(torch.Tensor()).to(self.device) if self.device else Variable(torch.Tensor())
        self.reward_episode= []
