import os
import json
import torch
import random
import logging
import numpy as np
from configparser import ConfigParser


class Config(object):
    def __init__(self, args, baseline=False):
        for key in dir(args):
            if not key.startswith('_'):
                setattr(self, key, getattr(args, key))

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

        config = ConfigParser()
        if baseline:
            cfg_file = './config/' + args.emb + '.baseline.cfg'
        else:
            cfg_file = './config/' + args.emb + '.cfg'
        config.read(cfg_file)

        self._config = config
        self.train_file = './data/fold/train_' + str(args.fold) + '.json'
        self.dev_file = './data/fold/val_' + str(args.fold) + '.json'
        self.test_file = './data/fold/test_' + str(args.fold) + '.json'

        if baseline:
            save_dir = './save-baseline/' + args.emb
        else:
            save_dir = './save/' + args.emb

        if self.use_dep:
            self.save_dir = save_dir + '/dep/fold_' + str(args.fold) + '/' + args.exp
        else:
            self.save_dir = save_dir + '/base/fold_' + str(args.fold) + '/' + args.exp

        self.save_model = self.save_dir + '/module_'
        self.save_config = self.save_dir + '/config.cfg'
        self.save_test = self.save_dir + '/test_'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            config.write(open(self.save_config, 'w'))

        _config = {}
        for key in dir(self):
            if not key.startswith('_'):
                _config[key] = getattr(self, key)

        logging.info(json.dumps(_config, indent=1))

        logging.info('Load config file: {}, seed: {}.'.format(cfg_file, args.seed))

    @property
    def glove_path(self):
        return self._config.get('Data', 'glove_path')

    @property
    def bert_path(self):
        return self._config.get('Data', 'bert_path')

    @property
    def word_dims(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def dropout_embed(self):
        return self._config.getfloat('Network', 'dropout_embed')

    @property
    def dropout_mlp(self):
        return self._config.getfloat('Network', 'dropout_mlp')

    @property
    def sent_num_layers(self):
        return self._config.getint('Network', 'sent_num_layers')

    @property
    def word_num_layers(self):
        return self._config.getint('Network', 'word_num_layers')

    @property
    def graph_num_layers(self):
        return self._config.getint('Network', 'graph_num_layers')

    @property
    def word_hidden_size(self):
        return self._config.getint('Network', 'word_hidden_size')

    @property
    def sent_hidden_size(self):
        return self._config.getint('Network', 'sent_hidden_size')

    @property
    def dropout_input(self):
        return self._config.getfloat('Network', 'dropout_input')

    @property
    def dropout_hidden(self):
        return self._config.getfloat('Network', 'dropout_hidden')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def bert_lr(self):
        return self._config.getfloat('Optimizer', 'bert_lr')

    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')

    @property
    def decay_steps(self):
        return self._config.getint('Optimizer', 'decay_steps')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def threads(self):
        return self._config.getint('Run', 'threads')

    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def log_interval(self):
        return self._config.getint('Run', 'log_interval')

    @property
    def early_stops(self):
        return self._config.getint('Run', 'early_stops')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')
