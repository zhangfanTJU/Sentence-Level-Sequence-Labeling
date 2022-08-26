import json
import pickle
import logging
import numpy as np
from pathlib import Path
from module import Embedding


def get_examples(file_name, embedding, vocab):
    mode = file_name.split('/')[-1][:-5]  # train validation test

    dir = './data/examples/' + mode
    if isinstance(embedding, Embedding):
        cache_name = dir + '.glove.pickle'
    else:
        cache_name = dir + '.bert.pickle'

    if Path(cache_name).exists():
        examples = pickle.load(open(cache_name, 'rb'))
        logging.info('Data from cache file: %s, total %d docs.' % (cache_name, len(examples)))
        return examples

    examples = []
    data = json.load(open(file_name, 'r'))

    for doc_data in data:
        doc = []
        doc_len = 0
        for label, words, dep_graph in doc_data:
            doc_len += len(words)
            dep_edges_index, dep_edges_type = dep_graph
            dep_edges_type_id = vocab.dep2id(dep_edges_type)
            dep_graph = (dep_edges_index, dep_edges_type_id)

            if isinstance(embedding, Embedding):
                word_ids = vocab.word2id(words)
                extword_ids = vocab.extword2id(words)
                doc.append([int(label), word_ids, extword_ids, dep_graph])
            else:
                token_ids, lens = embedding.retokenize(words)
                doc.append([int(label), token_ids, lens, dep_graph])

        examples.append(doc)

    logging.info('Data from file: %s, total %d docs.' % (file_name, len(examples)))

    pickle.dump(examples, open(cache_name, 'wb'))
    logging.info('Cache Data to file: %s, total %d docs.' % (cache_name, len(examples)))
    return examples


def get_examples_baseline(file_name, embedding, vocab):
    mode = file_name.split('/')[-1][:-5]  # train validation test

    dir = './data/examples-baseline/' + mode
    if isinstance(embedding, Embedding):
        cache_name = dir + '.glove.pickle'
    else:
        cache_name = dir + '.bert.pickle'

    if Path(cache_name).exists():
        examples = pickle.load(open(cache_name, 'rb'))
        logging.info('Data from cache file: %s, total %d sentences.' % (cache_name, len(examples)))
        return examples

    examples = []
    data = json.load(open(file_name, 'r'))

    for doc_data in data:
        for label, words, dep_graph in doc_data:
            dep_edges_index, dep_edges_type = dep_graph
            dep_edges_type_id = vocab.dep2id(dep_edges_type)
            dep_graph = (dep_edges_index, dep_edges_type_id)

            if isinstance(embedding, Embedding):
                word_ids = vocab.word2id(words)
                extword_ids = vocab.extword2id(words)
                examples.append([int(label), word_ids, extword_ids, dep_graph])
            else:
                token_ids, lens = embedding.retokenize(words)
                examples.append([int(label), token_ids, lens, dep_graph])

    logging.info('Data from file: %s, total %d sentences.' % (file_name, len(examples)))

    pickle.dump(examples, open(cache_name, 'wb'))
    logging.info('Cache Data to file: %s, total %d sentences.' % (cache_name, len(examples)))
    return examples


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


def data_iter(data, batch_size, config, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    indices = list(range(len(data)))

    if shuffle:
        np.random.shuffle(indices)
        data = [data[i] for i in indices]

    config.sorted_indices = indices
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch