import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.utils import reformat
from module import Encoder, Attention, Embedding, BertEmbedding, GraphEncoder, NoLinear, CRF


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

        # word
        if self.use_dep:
            graph_input_size = config.word_dims
            self.dep_encoder = GraphEncoder(graph_input_size, config.graph_num_layers, vocab.dep_size)

        word_input_size = config.word_dims * 2 if self.use_dep else config.word_dims
        self.word_encoder = Encoder(word_input_size, config, inputs='word')

        self.word_rep_size = config.word_hidden_size * 2
        self.word_attention = Attention(self.word_rep_size)

        # sent
        sent_input_size = self.word_rep_size
        self.sent_encoder = Encoder(sent_input_size, config, inputs='sent')
        self.sent_rep_size = config.sent_hidden_size * 2

        self.tag_proj = NoLinear(self.sent_rep_size, vocab.label_size)

        # criterion
        weight = 1 / torch.FloatTensor(vocab.label_weights)
        weight = weight / torch.sum(weight)

        if config.use_cuda:
            weight = weight.to(config.device)

        self.criterion = nn.CrossEntropyLoss(weight)

        # crf
        self.crf = CRF(vocab.label_size)

        if config.use_cuda:
            self.to(config.device)

        logging.info(self)
        logging.info('Build model with {} embedding, with dep: {}.'.format(config.emb, config.use_dep))
        
        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        para_need_grad = list(filter(lambda p: p.requires_grad, self.parameters()))
        para_need_grad_num = sum([np.prod(p.size()) for p in para_need_grad])
        logging.info('Model param num: %.2f M, need grad: %.2f M.' % (para_num / 1e6, para_need_grad_num / 1e6))

        
    def forward(self, batch_inputs, use_crf):
        inputs, dep_graphs, doc_lens, sent_labels = batch_inputs

        # word embed
        if self.emb == 'bert':
            tokens, token_masks, token_lens = inputs
            # word_embed: total_sent_num x max_sent_len x word_dims
            # word_masks: total_sent_num x max_sent_len
            word_embed, word_masks = self.embedding(tokens, token_masks, token_lens)
        else:
            words, extwords, word_masks = inputs
            word_embed = self.embedding(words, extwords)

        max_sent_len = word_embed.shape[1]
        batch_size, max_doc_len = len(doc_lens), max(doc_lens)

        # word input and enc
        if self.use_dep:
            sent_lens = word_masks.sum(-1).int().tolist()
            word_embed_lst = [embed.squeeze(0) for embed in torch.split(word_embed, 1, dim=0)]
            for i, sent_len in enumerate(sent_lens):
                word_embed_lst[i] = word_embed_lst[i][:sent_len]

            word_embed_dep = self.dep_encoder(word_embed_lst, dep_graphs)  # total_sent_num x max_sent_len x word_dims
            word_embed_dep = self.pad_batch(word_embed_dep, sent_lens)
            word_input = torch.cat([word_embed, word_embed_dep], dim=-1)
        else:
            word_input = word_embed

        word_hiddens = self.word_encoder(word_input, word_masks)  # total_sent_num x max_sent_len x word_rep_size

        sent_embed, word_attns = self.word_attention(word_hiddens, word_masks)  # total_sent_num x word_rep_size
    
        sent_input = sent_embed
        if max_doc_len == min(doc_lens):
            sent_input = sent_input.view(batch_size, max_doc_len, -1)  # b x max_doc_len x sent_input_size
            word_masks = word_masks.view(batch_size, max_doc_len, max_sent_len)  # b x max_doc_len x max_sent_len
            if not self.training:
                word_attns = word_attns.view(batch_size, max_doc_len, max_sent_len)  # b x max_doc_len x max_sent_len
        else:
            sent_input = self.pad_batch(sent_input, doc_lens)
            word_masks = self.pad_batch(word_masks, doc_lens)
            if not self.training:
                word_attns = self.pad_batch(word_attns, doc_lens)

        if not self.training:
            attns = {'word_attns': word_attns.tolist()}
        else:
            attns = {'word_attns': None}

        # sent enc
        sent_masks = word_masks.bool().any(2).float()  # b x max_doc_len
        sent_hiddens = self.sent_encoder(sent_input, sent_masks)  # b x max_doc_len x doc_rep_size

        # output
        sent_output = self.tag_proj(sent_hiddens)

        y_preds = []
        if use_crf:
            loss, sent_preds = self.crf(sent_output, sent_masks, sent_labels)  # b x num_labels
            y_preds = sent_preds
        else:
            loss = self.criterion(sent_output.view(-1, sent_output.shape[-1]), sent_labels.view(-1)) if self.training else None
            sent_preds = torch.max(sent_output, dim=-1)[1].tolist()

        y_labels = []
        for i, doc_len in enumerate(doc_lens):
            y_labels.append(sent_labels[i][:doc_len].tolist())
            if not use_crf:
                y_preds.append(sent_preds[i][:doc_len])

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
