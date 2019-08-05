# encoding=utf-8
import os
cwd = os.getcwd()


class HyperParam:
    def __init__(self,
                 mode,
                 vocab=14000):
        """
    The HyperParam of the model, you can use 'test', 'small' or 'large' to 
    specify model size.
    Args:
        mode: use 'test', 'small' or 'large' to use different model size.
        vocab: Number of tokens in the embedding.
    """
        self.model_summary_dir = cwd + "/model_summary"
        self.model_weights_dir = cwd + "/model_weights"
        self.model_checkpoint_dir = cwd + "/model_checkpoint"
        try:
            os.makedirs(self.model_weights_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_checkpoint_dir)
        except OSError:
            pass
        try:
            os.makedirs(self.model_summary_dir)
        except OSError:
            pass

        self.vocabulary_size = vocab

        if mode == 'test':
            self.test()
        if mode == 'small':
            self.small()
        if mode == 'large':
            self.large()

    def test(self,
             embedding_size=8,
             batch_size=8,
             heads_num=3,
             max_seq_len=10,
             input_chanels=2,
             filter_kinds=3,
             filters_size=(3,4,5),
             filter_nums=10,
             classes_nums=2,
             epoch_num=5,
             epoch=1,
             lr=0.02,
             dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.heads_num = heads_num
        self.max_seq_len = max_seq_len
        self.input_chanels = input_chanels
        self.filter_kinds = filter_kinds
        self.filters_size = filters_size
        self.filter_nums = filter_nums
        self.classes_nums = classes_nums
        self.epoch = epoch
        self.epoch_num = epoch_num
        self.dropout = dropout
        self.lr = lr

    def small(self,
             embedding_size=16,
             batch_size=16,
             heads_num=3,
             max_seq_len=25,
             input_chanels=2,
             filter_kinds=3,
             filters_size=(3,4,5),
             filter_nums=25,
             classes_nums=5,
             epoch_num=5,
             epoch=1,
             lr=2,
             dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.heads_num = heads_num
        self.max_seq_len = max_seq_len
        self.input_chanels = input_chanels
        self.filter_kinds = filter_kinds
        self.filters_size = filters_size
        self.filter_nums = filter_nums
        self.classes_nums = classes_nums
        self.epoch = epoch
        self.epoch_num = epoch_num
        self.dropout = dropout
        self.lr = lr

    def large(self,
              embedding_size=100,
              batch_size=32,
              heads_num=3,
              max_seq_len=100,
              input_chanels=2,
              filter_kinds=3,
              filters_size=(3,4,5),
              filter_nums=100,
              classes_nums=10,
              epoch_num=5,
              epoch=1,
              lr=0.001,
              dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.heads_num = heads_num
        self.max_seq_len = max_seq_len
        self.input_chanels = input_chanels
        self.filter_kinds = filter_kinds
        self.filters_size = filters_size
        self.filter_nums = filter_nums
        self.classes_nums = classes_nums
        self.epoch = epoch
        self.epoch_num = epoch_num
        self.dropout = dropout
        self.lr = lr
