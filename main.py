import tensorflow as tf
import numpy as np

import model_helper
import hyper_parameter
import data_helper

import time

print(tf.__version__)
print(np.__version__)

DATA_PATH = './data/rt-polarity'
TEST_PATH = './data/text_case'

def main():
    hp = hyper_parameter.HyperParam("test")
    word_embedding = model_helper.WordEmbedding(hp.vocabulary_size, hp.embedding_size)
    print('initial train model')
    model = model_helper.TextParsing(hp.embedding_size, hp.max_seq_len, hp.filter_kinds,
                            hp.filters_size, hp.filter_nums, hp.classes_nums,
                            hp.dropout, word_embedding)
    print('initial dataset')
    dataset, tokenizer = data_helper.generator_batch_dataset(TEST_PATH, hp.batch_size)
    print('initial optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)

    for epoch in range(hp.epoch_num):
        start = time.time()
        total_loss = 0
        
        for (batch, (x, y)) in enumerate(dataset):   
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = model.get_loss(logits, y)

            batch_loss = loss / int(y.shape[0])
            total_loss += batch_loss       
            variables = model.variables
            gradients = tape.gradient(loss, variables)        
            optimizer.apply_gradients(zip(gradients, variables)) 

            if batch % 1 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                        batch,
                                                        batch_loss))

if __name__ == '__main__':
    main()