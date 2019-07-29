# encoding=utf-8
import os
cwd = os.getcwd()


class HyperParam:
    def __init__(self,
                 mode,
                 vocab=14000):

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
             input_chanels=2,
             output_chanels=10,
             epoch_num=5,
             epoch=1,
             lr=2,
             dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.heads_num = heads_num
        self.input_chanels = input_chanels
        self.output_chanels = output_chanels
        self.epoch = epoch
        self.epoch_num = epoch_num
        self.dropout = dropout
        self.lr = lr

    def small(self,
             embedding_size=16,
             batch_size=16,
             heads_num=3,
             input_chanels=2,
             output_chanels=25,
             epoch_num=5,
             epoch=1,
             lr=2,
             dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.heads_num = heads_num
        self.input_chanels = input_chanels
        self.output_chanels = output_chanels
        self.epoch = epoch
        self.epoch_num = epoch_num
        self.dropout = dropout
        self.lr = lr

    def large(self,
              embedding_size=100,
              batch_size=32,
              heads_num=3,
              input_chanels=2,
              output_chanels=100,
              epoch_num=5,
              epoch=1,
              lr=2,
              dropout=0.4):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.heads_num = heads_num
        self.input_chanels = input_chanels
        self.output_chanels = output_chanels
        self.epoch = epoch
        self.epoch_num = epoch_num
        self.dropout = dropout
        self.lr = lr
