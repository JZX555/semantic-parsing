import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import hyper_parameter

class WordEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, name="embedding"):
        """
    Specify characteristic parameters of embedding layer.
    Args:
        vocab_size: Number of tokens in the embedding.
        embedding_size: the dimension of token.
    """
        super(WordEmbedding, self).__init__(name='word_embedding')
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        #self.pad_id = pad_id

    def build(self, input_shape):
        self.shared_weights = self.add_variable(
            shape=[self.vocab_size, self.embedding_size],
            dtype="float32",
            name="shared_weights",
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self.embedding_size**-0.5))
        super(WordEmbedding, self).build(input_shape)

    def call(self, inputs):
        mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.float32)
        embeddings = tf.gather(self.shared_weights, inputs)
        # embeddings = tf.nn.embedding_lookup(self.shared_weights, inputs)
        embeddings *= tf.expand_dims(mask, -1)
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.embedding_size**0.5

        return embeddings

class TextParsing(tf.keras.Model):
    def __init__(self,
             embedding_size,
             max_seq_len,
             filter_kinds,
             filters_size,
             filter_nums,
             classes_nums,
             dropout,
             word_embedding):
        """
    The model for semantic parsing.
    Args:
        ...
    Input:
        A batch of tokens with shape [batch_size, token length].
    Output:
        The classification results after SoftMax.
    """
        super(TextParsing, self).__init__(name = 'semantic_parsing')

        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len

        self.filter_kinds = filter_kinds
        self.filters_size = filters_size
        self.filter_nums = filter_nums

        self.classes_nums = classes_nums
        self.dropout = dropout
        self.word_embedding = word_embedding

        self.kernel_initializer = "ones"
        self.bias_initializer = "zeros"

    def build(self, input_shape):
        self.convs = []
        self.pools = []
        
        self.fc_kernel = self.add_variable(
            shape=[self.filter_nums * self.filter_kinds, self.classes_nums],
            name="fc_kernel",
            dtype=tf.float32,
            initializer=self.kernel_initializer)
        
        self.fc_bias = self.add_variable(
            shape=(self.classes_nums),
            name="fc_bias",
            dtype=tf.float32,
            initializer=self.bias_initializer)

        for i in range(self.filter_kinds):
            conv2D = tf.keras.layers.Conv2D(
                self.filter_nums,
                (self.filters_size[i], self.embedding_size),
                padding='VALID',
                activation='relu',
                name='cnn_filter_{0}'.format(i)
            )
            #pool = tf.keras.layers.MaxPooling2D((self.max_seq_len - self.filters_size[i] + 1, 1))
            pool = tf.keras.layers.MaxPooling2D((input_shape[-1] - self.filters_size[i] + 1, 1))

            self.convs.append(conv2D)
            self.pools.append(pool)

        #super(TextParsing, self).build(input_shape)

    def call(self, inputs):
        embeded = self.word_embedding(inputs)
        embeded =tf.expand_dims(embeded, -1)

        #print(np.shape(embeded))

        pool_output = []

        for i in range(self.filter_kinds):
            feature = self.convs[i](embeded)
            pooled = self.pools[i](feature)
            #print(np.shape(pooled))

            pool_output.append(pooled)
        
        fc = tf.concat(pool_output, -1)
        fc = tf.reshape(fc, [-1, self.filter_nums * self.filter_kinds])
        #print(np.shape(fc))

        droped = tf.nn.dropout(fc, rate=self.dropout)

        logits = tf.matmul(droped, self.fc_kernel) + self.fc_bias

        #projection = tf.nn.softmax(logits)

        return logits
    
    def get_predict_with_logits(self, logits):
        return tf.argmax(logits, axis=-1)

    def get_loss(self, src, tgt):
        loss = tf.nn.softmax_cross_entropy_with_logits(tgt, src)
        return loss

    def get_accuracy(self, logits, label):
        predict = self.get_predict_with_logits(logits)
        ground_truth = tf.argmax(label, axis=-1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, ground_truth), tf.float32))

        print(predict)
        print(ground_truth)
        print(accuracy)

        return accuracy

if __name__ == "__main__":
    hp = hyper_parameter.HyperParam("test")
    word_embedding = WordEmbedding(hp.vocabulary_size, hp.embedding_size)
    print('initial test model')
    test = TextParsing(hp.embedding_size, hp.max_seq_len, hp.filter_kinds,
                        hp.filters_size, hp.filter_nums, hp.classes_nums,
                        hp.dropout, word_embedding)

    test_case = tf.constant(np.ones((16, hp.max_seq_len)), dtype=tf.int32)
    test_label = tf.constant(np.ones((16, 2)), dtype=tf.int32)
    print(test_case)

    out = test(test_case)
    print(out)

    loss = test.get_loss(out, test_label)
    print(loss)