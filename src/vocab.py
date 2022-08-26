import json
import logging
import numpy as np
from collections import Counter


class Vocab():
    def __init__(self, file):
        self.min_count = 1
        self.pad = 0
        self.unk = 1
        self._id2word = ['<pad>', '<unk>']
        self._id2extword = ['<pad>', '<unk>']

        self._id2label = [
            'Other', 'Collect_Personal_Information', 'Data_Retention_Period', 'Data_Processing_Purposes', 'Contact_Details',
            'Right_to_Access', 'Right_to_Rectify_or_Erase', 'Right_to_Restrict_Processing', 'Right_to_Object_to_Processing',
            'Right_to_Data_Portability', 'Right_to_Lodge_a_Complaint'
        ]
        self.label_weights = []

        self._id2dep = [
            'case', 'punct', 'det', 'conj', 'obj', 'amod', 'nsubj', 'obl', 'compound', 'cc', 'nmod', 'nmod:poss', 'mark', 'advmod',
            'aux', 'dep', 'advcl', 'xcomp', 'aux:pass', 'nsubj:pass', 'acl', 'ccomp', 'cop', 'acl:relcl', 'fixed', 'parataxis',
            'appos', 'nummod', 'discourse', 'compound:prt', 'iobj', 'obl:npmod', 'csubj', 'expl', 'det:predet', 'cc:preconj',
            'obl:tmod', 'csubj:pass', 'orphan', 'goeswith'
        ]

        self.build_vocab(file)

        logging.info("Build vocab: words %d, labels %d, dep %d." %
                     (self.word_size, self.label_size, self.dep_size))

    def build_vocab(self, file):
        word_counter = Counter()
        label_counter = Counter()
        # dep_counter = Counter()

        data = json.load(open(file, 'r'))
        for doc_data in data:
            for label, words, graph in doc_data[:-1]:
                label_counter[int(label)] += 1
                for word in words:
                    word_counter[word] += 1
            #     for dep in graph[1]:
            #         dep_counter[dep] += 1

        for word, count in word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        for label in range(self.label_size):
            self.label_weights.append(label_counter[label])

        # for dep, _ in dep_counter.most_common():
        #     self._id2dep.append(dep)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)
        self._dep2id = reverse(self._id2dep)

    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            word_count = len(lines)
            values = lines[0].split()
            embedding_dim = len(values) - 1

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        logging.info("Load extword embed: words %d, dims %d." % (self.extword_size, embedding_dim))

        return embeddings

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    def dep2id(self, xs):
        if isinstance(xs, list):
            return [self._dep2id.get(x) for x in xs]
        return self._dep2id.get(xs)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x) for x in xs]
        return self._label2id.get(xs)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)

    @property
    def dep_size(self):
        return len(self._id2dep)
