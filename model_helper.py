import tensorflow as tf

class WordEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, pad_id, name="embedding"):
        """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      num_units: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
        super(WordEmbedding, self).__init__(name='word_embedding')
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pad_id = pad_id

    def build(self, input_shape):
        self.shared_weights = self.add_variable(
            shape=[self.vocab_size, self.embedding_size],
            dtype="float32",
            name="shared_weights",
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self.num_units**-0.5))
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
    def init(self,
             vocabulary_size,
             embedding_size,
             max_seq_len,
             filter_kinds,
             filters_size,
             filter_nums,
             dropout,
             word_embedding):
        super(TextParsing, self).__init__(name = 'semantic_parsing')

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len

        self.filter_kinds = filter_kinds
        self.filters_size = filters_size
        self.filter_nums = filter_nums

        self.dropout = dropout
        self.word_embedding = word_embedding

    def build(self):
        self.convs = []
        self.pools = []

        for i in range(self.filter_kinds):
            conv2D = tf.keras.layers.Conv2D(
                self.filter_nums,
                (self.filters_size[i], self.embedding_size),
                padding='VALID',
                activation='relu',
                name='cnn_filter_{0}'.format(i)
            )
            pool = tf.keras.layers.MaxPooling2D(self.max_seq_len - self.filters_size[i] + 1, 1)

            self.convs.append(conv2D)
            self.pools.append(pool)

    def call(self, inputs):
        embeded = self.word_embedding(inputs)
        embeded =tf.expand_dims(embeded, -1)

        pools = []

        for i in range(self.filter_kinds):
            feature = self.convs[i](embeded)
            pooled = self.pools[i](feature)

            pools.append(pooled)
        
        fc = tf.concat(pools, -1)
        fc = tf.reshape(fc, [-1, self.filter_nums * self.filter_kinds])

        droped = tf.nn.dropout(fc, rate=self.dropout)