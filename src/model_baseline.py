import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.utils import reformat
from module import Encoder, Attention, Embedding, BertEmbedding, GraphEncoder, NoLinear


class Model(nn.Module):
    def __init__(self, config, vocab):
        super(Model, self).__init__()
        self.emb = config.emb
        self.use_dep = config.use_dep

        # emb
        if self.emb == 'bert':
            self.embedding = BertEmbedding(config)
        else:
            self.embedding = Embedding(config, vocab)

        word_input_size = config.word_dims * 2 if self.use_dep else config.word_dims
        self.word_encoder = Encoder(word_input_size, config, inputs='word')
        self.word_rep_size = config.word_hidden_size * 2

        # word
        if self.use_dep:
            graph_input_size = config.word_dims
            self.dep_encoder = GraphEncoder(graph_input_size, config.graph_num_layers, vocab.dep_size)

        self.word_attention = Attention(self.word_rep_size)

        self.tag_proj = NoLinear(self.word_rep_size, vocab.label_size)

        # criterion
        weight = 1 / torch.FloatTensor(vocab.label_weights)
        weight = weight / torch.sum(weight)

        if config.use_cuda:
            weight = weight.to(config.device)

        self.criterion = nn.CrossEntropyLoss(weight)

        if config.use_cuda:
            self.to(config.device)

        logging.info(self)
        logging.info('Build baseline model with {} embedding, with dep: {}.'.format(config.emb, config.use_dep))
        
        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        para_need_grad = list(filter(lambda p: p.requires_grad, self.parameters()))
        para_need_grad_num = sum([np.prod(p.size()) for p in para_need_grad])
        logging.info('Model param num: %.2f M, need grad: %.2f M.' % (para_num / 1e6, para_need_grad_num / 1e6))

        
    def forward(self, batch_inputs):
        inputs, dep_graphs, sent_labels = batch_inputs

        # word embed
        if self.emb == 'bert':
            tokens, token_masks, token_lens = inputs
            # word_embed: batch_size x max_sent_len x word_dims
            # word_masks: batch_size x max_sent_len
            word_embed, word_masks = self.embedding(tokens, token_masks, token_lens)
        else:
            words, extwords, word_masks = inputs
            word_embed = self.embedding(words, extwords)

        # word input and enc
        if self.use_dep:
            sent_lens = word_masks.sum(-1).int().tolist()
            word_embed_lst = [embed.squeeze(0) for embed in torch.split(word_embed, 1, dim=0)]
            for i, sent_len in enumerate(sent_lens):
                word_embed_lst[i] = word_embed_lst[i][:sent_len]

            word_embed_dep = self.dep_encoder(word_embed_lst, dep_graphs)  # batch_size x max_sent_len x word_dims
            word_embed_dep = self.pad_batch(word_embed_dep, sent_lens)
            word_input = torch.cat([word_embed, word_embed_dep], dim=-1)
        else:
            word_input = word_embed

        word_hiddens = self.word_encoder(word_input, word_masks)  # batch_size x max_sent_len x word_rep_size

        # sent embed
        sent_embed, word_attns = self.word_attention(word_hiddens, word_masks)  # batch_size x word_rep_size

        if not self.training:
            attns = {'word_attns': word_attns.tolist()}
        else:
            attns = {'word_attns': None}

        # output
        sent_output = self.tag_proj(sent_embed)

        loss = self.criterion(sent_output, sent_labels) if self.training else None
        y_preds = torch.max(sent_output, dim=-1)[1].tolist()
        y_labels = sent_labels.tolist()

        return loss, y_preds, y_labels, attns

    def pad_batch(self, batch_inputs, batch_lens):
        '''
            split batch_inputs and pad them to max_len

            # Parameters
                batch_inputs: total_num x hidden_size
                batch_lens: list of num fer batch

            # Returns:
                batch_outputs: b x max_len x hidden_size
        '''
        assert batch_inputs.shape[0] == sum(batch_lens)
        batch_size, max_len = len(batch_lens), max(batch_lens)
        batch_split = list(torch.split(batch_inputs, batch_lens))

        for i in range(batch_size):
            len_ = batch_lens[i]
            if len_ < max_len:
                batch_split[i] = F.pad(batch_split[i], (0, 0, 0, max_len - len_))

        batch_outputs = torch.stack(batch_split)
        return batch_outputs

    def merge_attens(self, word_attns, sent_scores, batch_masks):
        # word_attns: b x max_doc_len x max_sent_len
        # batch_masks: b x max_doc_len x max_sent_len
        # sent_scores: b x max_doc_len

        sent_attens = []
        word_attns = []
        # [[sent0_len, sent2_len, ...], ...]
        batch_lens = torch.sum(batch_masks, 2).int().tolist()  # b x max_doc_len

        word_attns = word_attns.tolist()
        sent_scores = sent_scores.tolist()
        for i, sent_lens in enumerate(batch_lens):
            word_attns_ = []
            sent_lens = list(filter(lambda x: x > 0, sent_lens))
            doc_len = len(sent_lens)
            for j, sent_len in enumerate(sent_lens):
                scores = word_attns[i][j][:sent_len]
                norm_scores = [reformat(score / sum(scores), 2) for score in scores]
                word_attns_.append(norm_scores)
            scores = sent_scores[i][:doc_len]
            norm_scores = [reformat(score / sum(scores), 2) for score in scores]
            sent_attens.append(norm_scores)
            word_attns.append(word_attns_)

        return word_attns, sent_attens

    def merge_sent_attens(self, sent_scores, batch_masks):
        # word_attns: b x max_doc_len x max_sent_len
        # batch_masks: b x max_doc_len x max_sent_len
        # sent_scores: b x max_doc_len

        sent_attens = []
        word_attns = None
        # [[sent0_len, sent2_len, ...], ...]
        batch_lens = torch.sum(batch_masks, 2).tolist()  # b x max_doc_len

        sent_scores = sent_scores.tolist()
        for i, sent_lens in enumerate(batch_lens):
            sent_lens = list(filter(lambda x: x > 0, sent_lens))
            doc_len = len(sent_lens)

            scores = sent_scores[i][:doc_len]
            norm_scores = [reformat(score / sum(scores), 2) for score in scores]
            sent_attens.append(norm_scores)

        return word_attns, sent_attens
