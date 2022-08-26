import sys
import torch
import pickle
import logging
import argparse
from pathlib import Path
from src.vocab import Vocab
from src.model import Model
from src.config import Config
from src.trainer import Trainer

sys.path.extend(["../../", "../", "./"])
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--emb', default='bert', choices=['glove', 'bert'], help='word embedding')
    argparser.add_argument('--num', default=6, type=int, help='finetune layer nums')
    argparser.add_argument('--crf_start_epoch', '--start', default=5, type=int, help='crf start epoch')
    argparser.add_argument('--use_dep', action='store_true', help='use dependency parsing')
    argparser.add_argument('--use_dis', action='store_true', help='use discourse parsing')
    argparser.add_argument('--sent_rep', default='avg', choices=['attn', 'avg'], help='pooling for sentence representation')
    argparser.add_argument('--span_rep', default='add', choices=['attn', 'concat', 'sub', 'add', 'mean'], help='pooling for span representation')
    argparser.add_argument('--seed', default=888, type=int, help='seed')
    argparser.add_argument('--exp', default='128', type=str, help='exp name')
    argparser.add_argument('--gpu', default=2, type=int, help='gpu id')
    argparser.add_argument('--fold', default=0, type=int, help='fold for test')
    argparser.add_argument('--epoch', default=0, type=int, help='best epoch for test')
    args = argparser.parse_args()

    config = Config(args)
    torch.set_num_threads(config.threads)

    # set cuda
    config.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if config.use_cuda:
        torch.cuda.set_device(args.gpu)
        config.device = torch.device("cuda", args.gpu)
    else:
        config.device = torch.device("cpu")
    logging.info("Use cuda: %s, gpu id: %d.", config.use_cuda, args.gpu)

    # vocab
    cache_name = "./save/vocab/" + str(args.fold) + ".pickle"
    if Path(cache_name).exists():
        vocab = pickle.load(open(cache_name, 'rb'))
        logging.info('Load vocab from ' + cache_name + ', words %d, labels %d, dep %d, dis %d.' %
                     (vocab.word_size, vocab.label_size, vocab.dep_size, vocab.dis_size))
    else:
        vocab = Vocab(config.train_file)
        pickle.dump(vocab, open(cache_name, 'wb'))
        logging.info('Cache vocab to ' + cache_name)

    # model
    model = Model(config, vocab)

    # trainer
    trainer = Trainer(model, config, vocab)
    trainer.test()

    logging.info('Done.')
