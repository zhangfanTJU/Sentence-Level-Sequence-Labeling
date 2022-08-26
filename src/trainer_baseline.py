import gc
import math
import time
import json
import torch
import logging
import subprocess
import numpy as np
import torch.nn as nn

from sklearn.metrics import classification_report

from src.loader import get_examples_baseline, data_iter
from src.optimizer import Optimizer
from src.utils import get_score, reformat


class Trainer():
    def __init__(self, model, config, vocab):
        self.model = model
        self.config = config
        self.report = True

        self.train_data = get_examples_baseline(config.train_file, model.embedding, vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(config.train_batch_size)))

        self.dev_data = get_examples_baseline(config.dev_file, model.embedding, vocab)
        self.test_data = get_examples_baseline(config.test_file, model.embedding, vocab)

        # label name and
        self.target_names = vocab._id2label

        # id2word of id2token
        if config.emb == 'bert':
            self.id2word = model.embedding.tokenizer._convert_id_to_token
        else:
            self.id2word = lambda id: vocab._id2word[id]

        # optimizer
        self.optimizer = Optimizer(model, config, self.batch_num)

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1, self.best_test_f1 = 0, 0, 0
        self.last_epoch = config.epochs + 1

    def train(self):
        logging.info("Start training.")
        for epoch in range(1, self.config.epochs + 1):
            gc.collect()
            train_f1 = self._train(epoch)
            self.logging_gpu_memory()

            gc.collect()
            dev_f1 = self._eval(epoch, "dev")
            self.logging_gpu_memory()

            gc.collect()
            test_f1 = self._eval(epoch, "test")
            self.logging_gpu_memory()

            # if self.best_dev_f1 < dev_f1 or (self.best_dev_f1 == dev_f1 and self.best_test_f1 < test_f1):
            if self.best_dev_f1 <= dev_f1 and self.best_test_f1 < test_f1:
                logging.info("Exceed history dev = %.2f, current train = %.2f dev = %.2f test = %.2f epoch = %d" %
                             (self.best_dev_f1, train_f1, dev_f1, test_f1, epoch))
                if epoch > self.config.save_after:
                    torch.save(self.model.state_dict(), self.config.save_model + str(epoch) + '.bin')

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.best_test_f1 = test_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == self.config.early_stops:
                    logging.info("Eearly stop in epoch %d, best train: %.2f, dev: %.2f, test: %.2f" %
                                 (epoch - self.config.early_stops, self.best_train_f1, self.best_dev_f1, self.best_test_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        logging.info("Start testing.")
        model = self.config.save_model + str(self.config.epoch) + '.bin'
        logging.info('Load model from {}.'.format(model))
        self.model.load_state_dict(torch.load(model, map_location=self.config.device))
        self._eval(-1, "test")

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()

        overall_losses = 0
        losses = 0

        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, self.config.train_batch_size, self.config, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs = self.batch2tensor(batch_data)
            loss, preds, labels, _ = self.model(batch_inputs)

            loss = loss / self.config.update_every
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(preds)
            y_true.extend(labels)

            if batch_idx % self.config.update_every == 0 or batch_idx == self.batch_num:
                nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=self.config.clip)
                for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                    optimizer.step()
                    scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1

            if batch_idx % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                lrs = self.optimizer.get_lr()
                logging.info('| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                    epoch, self.step, batch_idx, self.batch_num, lrs, losses / self.config.log_interval, elapsed / self.config.log_interval))

                start_time = time.time()
                losses = 0

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = reformat(overall_losses, 4)
        all_score, score, f1 = get_score(y_true, y_pred)

        logging.info('| epoch {:3d} | all score {} | score {} | f1 {} | loss {} | time {:.2f}'.format(epoch, all_score, score, f1, overall_losses,
                                                                                                      during_time))

        return f1

    def _eval(self, epoch, data_nane, test_batch_size=None):
        self.model.eval()

        start_time = time.time()

        if data_nane == "dev":
            data = self.dev_data
        elif data_nane == "test":
            data = self.test_data

        if test_batch_size is None:
            test_batch_size = self.config.test_batch_size

        y_pred = []
        y_true = []
        save_dic = {'words': [], 'word_attns': [], 'true_label': [], 'pred_label': []}
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, self.config, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs = self.batch2tensor(batch_data)

                _, preds, labels, attens = self.model(batch_inputs)

                save_dic['true_label'].extend(labels)
                save_dic['pred_label'].extend(preds)
                y_pred.extend(preds)
                y_true.extend(labels)

                if data_nane == "test":
                    words = self.batch2word(batch_data)
                    self.add_save(save_dic, words, attens)

            all_score, score, f1 = get_score(y_true, y_pred)
            save_dic['all_score'] = all_score
            save_dic['score'] = score
            save_dic['f1'] = f1

            during_time = time.time() - start_time
            logging.info('| epoch {:3d} | {} | all_score {} | score {} | f1 {} | time {:.2f}'.format(epoch, data_nane, all_score, score, f1,
                                                                                                     during_time))
            if set(y_true) == set(y_pred) and data_nane == "test":
                report = classification_report(y_true, y_pred, digits=4)
                logging.info('\n' + report)

        if data_nane == "test":
            for key, value in save_dic.items():
                if isinstance(value, list) and len(value) > 0:
                    save_dic[key] = self.convert_sort(value)

            json.dump(save_dic, open(self.config.save_test + str(epoch) + '.json', 'w'))

        return f1

    def add_save(self, save_dic, words, attns):
        save_dic['words'].extend(words)
        for key, value in attns.items():
            if value is not None:
                save_dic[key].extend(value)

    def convert_sort(self, lst):
        return [lst[self.config.sorted_indices.index(idx)] for idx in range(len(self.config.sorted_indices))]

    def batch2tensor(self, batch_data):
        if self.config.emb == 'bert':
            return self.batch2tensor_bert(batch_data)
        else:
            return self.batch2tensor_glove(batch_data)

    def batch2tensor_glove(self, batch_data):
        '''
            [(label, word_ids, extword_ids, dep_graph), ()]
        '''

        batch_size = len(batch_data)
        sent_lens = [len(sent_data[1]) for sent_data in batch_data]
        max_sent_len = max(sent_lens)

        dep_graphs = []

        words = torch.zeros((batch_size, max_sent_len), dtype=torch.int64)
        extwords = torch.zeros((batch_size, max_sent_len), dtype=torch.int64)
        word_masks = torch.zeros((batch_size, max_sent_len), dtype=torch.float32)

        for bs, sent_data in enumerate(batch_data):
            for word_idx in range(sent_lens[bs]):
                words[bs, word_idx] = sent_data[1][word_idx]
                extwords[bs, word_idx] = sent_data[2][word_idx]
                word_masks[bs, word_idx] = 1

            if self.config.use_dep:
                dep_graphs.append(sent_data[3])

        labels = torch.tensor([int(sent_data[0]) for sent_data in batch_data])

        if self.config.use_cuda:
            words = words.to(self.config.device)
            extwords = extwords.to(self.config.device)
            word_masks = word_masks.to(self.config.device)
            labels = labels.to(self.config.device)

        inputs = (words, extwords, word_masks)

        return inputs, dep_graphs, labels

    def batch2tensor_bert(self, batch_data):
        '''
            [(label, token_ids, lens, dep_graph), ()]
        '''

        batch_size = len(batch_data)
        sent_lens = [len(sent_data[1]) for sent_data in batch_data]
        max_sent_len = max(sent_lens)

        token_lens = []
        dep_graphs = []

        tokens = torch.zeros((batch_size, max_sent_len), dtype=torch.int64)
        token_masks = torch.zeros((batch_size, max_sent_len), dtype=torch.float32)

        for bs, sent_data in enumerate(batch_data):
            for word_idx in range(sent_lens[bs]):
                tokens[bs, word_idx] = sent_data[1][word_idx]
                token_masks[bs, word_idx] = 1

            token_lens.append(sent_data[2])

            if self.config.use_dep:
                dep_graphs.append(sent_data[3])
        
        labels = torch.tensor([int(sent_data[0]) for sent_data in batch_data])

        if self.config.use_cuda:
            tokens = tokens.to(self.config.device)
            token_masks = token_masks.to(self.config.device)
            labels = labels.to(self.config.device)

        inputs = (tokens, token_masks, token_lens)
        return inputs, dep_graphs, labels

    def batch2word(self, batch_data):
        '''
            for glove: [(label, word_ids, extword_ids, dep_graph), ()]
            for bert:  [(label, token_ids, lens, dep_graph), ()]
        '''

        batch_words = []
        for sent_data in batch_data:
            sent_words = [self.id2word(id) for id in sent_data[1]]
            batch_words.append(sent_words)
        return batch_words

    def check_lens(self, doc_words, word_attens, sent_attens):
        assert len(doc_words) == len(sent_attens)  # batch_size
        if word_attens is not None:
            assert len(doc_words) == len(word_attens)
            for doc_word, word_atten, sent_atten in zip(doc_words, word_attens, sent_attens):
                assert len(doc_word) == len(word_atten) == len(sent_atten)  # doc_len
                for sent_word, word_atten_ in zip(doc_word, word_atten):
                    assert len(sent_word) == len(word_atten_)  # sent_len
        else:
            for doc_word, sent_atten in zip(doc_words, sent_attens):
                assert len(doc_word) == len(sent_atten)  # doc_len

    def logging_gpu_memory(self):
        """
        Get the current GPU memory usage.
        Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
        Returns
        -------
        ``Dict[int, int]``
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
            Returns an empty ``dict`` if GPUs are not available.
        """
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                encoding="utf-8",
            )
            info = [x.split(',') for x in result.strip().split("\n")]
            dic = {gpu: [int(mem[0]), int(mem[1])] for gpu, mem in enumerate(info)}
            gpu = self.config.gpu
            lst = dic[gpu]
            logging.info('| gpu {} | use {:5d}M / {:5d}M'.format(self.config.gpu, lst[0], lst[1]))

        except FileNotFoundError:
            # `nvidia-smi` doesn't exist, assume that means no GPU.
            return {}
        except:  # noqa
            # Catch *all* exceptions, because this memory check is a nice-to-have
            # and we'd never want a training run to fail because of it.
            logging.info("unable to check gpu_memory_mb(), continuing")
            return {}
