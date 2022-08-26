import pandas as pd
import json
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
from collections import Counter

nlp = StanfordCoreNLP('./emb/stanford_corenlp')
np.random.seed(888)


def build_dep_graph(text, words):
    props = {'annotators': 'depparse', 'outputFormat': 'json', 'pipelineLanguage': 'en'}
    post_len = 0
    edges_type = []
    src_idx = []
    tgt_idx = []

    dic = json.loads(nlp.annotate(text, properties=props))
    for sent in dic['sentences']:
        for dep in sent['basicDependencies']:
            # head -> tail, tail修饰head
            # src -> tgt, flow: target to source
            # governor -> dependent
            label, src, tgt = dep['dep'], dep['governor'], dep['dependent']
            src_word, tgt_word = dep['governorGloss'], dep['dependentGloss']
            if src != 0:
                src_idx.append(src + post_len - 1)
                tgt_idx.append(tgt + post_len - 1)
                edges_type.append(label)
                assert src_word == words[src + post_len - 1]
                assert tgt_word == words[tgt + post_len - 1]
            else:
                assert tgt_word == words[tgt + post_len - 1]

        post_len += len(sent['basicDependencies'])

    edges_index = [src_idx, tgt_idx]
    return (edges_index, edges_type), post_len


def collect_data_label():
    label_counter = Counter()
    file_name = './data/data.tsv'
    f = pd.read_csv(file_name, sep='\t', encoding='UTF-8')
    labels = f['label'].tolist()
    for label in labels:
        label_counter[label] += 1

    for label in range(11):
        print(label, label_counter[label])
    print(sum(label_counter.values()))


def convert_data():
    file_name = './data/data.tsv'
    f = pd.read_csv(file_name, sep='\t', encoding='UTF-8')
    texts = f['sentence'].map(lambda x: x.lower().strip()).tolist()
    labels = f['label'].tolist()
    fns = f['filename'].tolist()

    print(len(texts), len(set(fns)))

    data = {}
    for text, label, fn in zip(texts, labels, fns):
        if fn not in data:
            data[fn] = []
        data[fn].append(str(label) + '\t' + text)

    examples = []
    for doc in data.values():
        example = []
        for line in doc:
            label, text_ = line.split('\t')
            words = nlp.word_tokenize(text_)[:128]
            for i in range(len(words)):
                words[i] = ''.join(words[i].split())
            sent_len = len(words)
            text = ' '.join(words)
            dep_graph, total_len = build_dep_graph(text, words)
            assert sent_len == total_len
            example.append([label, words, dep_graph])
        examples.append(example)

    json.dump(examples, open('./data/preprocess/data.json', 'w'))
    print(len(examples))


def split_data():
    def idx2data(fold, dtype, idxs):
        np.random.shuffle(idxs)
        data = [examples[idx] for idx in idxs]
        json.dump(data, open('./data/fold/' + dtype + '_' + str(fold) + '.json', 'w'))
        print(fold, dtype, len(idxs))

    examples = []
    others = []
    lengths = {}
    idx = 0

    data = json.load(open('./data/preprocess/data.json', 'r'))
    all_data = data[48:] + data[24:48] + data[:23]
    for data in all_data:
        don_len = 0
        for sent in data:
            don_len += len(sent[1])
        c = don_len // 2000
        if c > 3:
            others.append(idx)
        else:
            if c not in lengths:
                lengths[c] = []
            lengths[c].append(idx)
        examples.append(data)
        idx += 1

    print(len(examples))
    print(len(others))
    np.random.shuffle(others)

    fold_idxs = {}
    for key, idx_lst in lengths.items():
        print(key, len(idx_lst))
        np.random.shuffle(idx_lst)
        num = len(idx_lst)
        index = list(range(10)) * (num // 10)
        if key == 2:
            index += [7, 8, 9]
        elif key == 0:
            index += list(range(0, num % 10 * 2, 2))
        elif key == 3:
            index += list(range(1, num % 10 * 2, 2))

        assert len(index) == num

        for i in range(num):
            fold_idx = index[i]
            data_idx = idx_lst[i]
            if fold_idx not in fold_idxs:
                fold_idxs[fold_idx] = []
            fold_idxs[fold_idx].append(data_idx)

    print(list(map(lambda x: len(x), fold_idxs.values())))

    for i in range(0, 10, 2):
        fold = i // 2
        test_idx = fold_idxs[i] + fold_idxs[i + 1]
        if fold < len(others):
            test_idx.append(others[fold])
        idx2data(fold, 'test', test_idx)

        val_idx = fold_idxs[(i + 2) % 10]
        idx2data(fold, 'val', val_idx)

        train_idx = []
        for j in range(10):
            if j in [i, i + 1, (i + 2) % 10]:
                continue
            train_idx += fold_idxs[j]
        idx2data(fold, 'train', train_idx)

        assert len(set(train_idx + val_idx + test_idx)) == len(train_idx + val_idx + test_idx)


if __name__ == "__main__":
    convert_data()
    split_data()
