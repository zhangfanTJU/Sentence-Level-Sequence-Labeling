import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from module.layers import drop_input_independent, NoLinear, set_requires_grad

from transformers import BertTokenizer, BertModel


class Embedding(nn.Module):
    def __init__(self, config, vocab):
        super(Embedding, self).__init__()
        self.dropout_embed = config.dropout_embed

        word_embed = np.zeros((vocab.word_size, config.word_dims), dtype=np.float32)
        self.word_embed = nn.Embedding(vocab.word_size, config.word_dims, padding_idx=0)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_embed))

        extword_embed = vocab.load_pretrained_embs(config.glove_path)
        extword_size, word_dims = extword_embed.shape
        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

    def forward(self, word_ids, extword_ids):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len

        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = drop_input_independent(batch_embed, self.dropout_embed)  # sen_num x sent_len x embed_dim

        return batch_embed


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(config.dropout_mlp)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path)

        if config.num > 0:
            set_requires_grad(self.bert, False)

            for i in range(1, config.num + 1):
                set_requires_grad(self.bert.encoder.layer[-i], True)

        self.proj = NoLinear(self.bert.config.hidden_size, config.word_dims)

    def retokenize(self, words):
        tokens = ["[CLS]"]
        lens = []
        for word in words:
            tokens_ = self.tokenizer.tokenize(word)
            lens.append(len(tokens_))
            tokens.extend(tokens_)
        tokens.append("[SEP]")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        assert len(token_ids) <= 512
        return token_ids, lens

    def merge_embedding(self, hiddens, lens, max_sent_len, merge='mean'):
        # hiddens: bert_len x 768
        # lens: list of word_len
        # lens exclude cls and sep

        bert_len, sent_len = sum(lens), len(lens)
        hiddens = hiddens[:bert_len]
        if bert_len != sent_len:  # no subword
            hiddens_split = list(torch.split(hiddens, lens, dim=0))
            if merge == 'mean':
                hiddens_split = list(map(lambda x: torch.mean(x, dim=0), hiddens_split))
            elif merge == 'sum':
                hiddens_split = list(map(lambda x: torch.sum(x, dim=0), hiddens_split))

            hiddens = torch.stack(hiddens_split)  # sent_len x 768

        if sent_len < max_sent_len:
            hiddens = F.pad(hiddens, (0, 0, 0, max_sent_len - sent_len))  # max_sent_len x 768

        return hiddens

    def forward(self, input_ids, attention_mask, token_lens):
        # input_ids: sen_num x bert_len
        # sent_lens: sen_num x list of word_len

        hiddens, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # sen_num x bert_len x 768
        hiddens = hiddens[:, 1:, :]  # remove cls and sep
        hiddens = self.proj(hiddens)  # sen_num x max_sent_len x word dims

        sent_lens = [len(lens) for lens in token_lens]
        max_sent_len = max(sent_lens)
        word_masks = torch.zeros((hiddens.shape[0], max_sent_len), dtype=torch.float32)  # sen_num x sent_len

        sent_hiddens = list(torch.split(hiddens, 1, dim=0))
        for i in range(len(sent_hiddens)):
            sent_hiddens[i] = self.merge_embedding(sent_hiddens[i].squeeze(0), token_lens[i], max_sent_len, merge='mean')  # max_sent_len x 768
            for j in range(sent_lens[i]):
                word_masks[i][j] = 1

        hiddens = torch.stack(sent_hiddens)  # sen_num x max_sent_len x 768

        if self.training:
            hiddens = self.dropout(hiddens)

        word_masks = word_masks.to(hiddens.device)
        return hiddens, word_masks
