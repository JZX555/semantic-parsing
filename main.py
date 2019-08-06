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

def main(path):
    hp = hyper_parameter.HyperParam("large")
    word_embedding = model_helper.WordEmbedding(hp.vocabulary_size, hp.embedding_size)
    print('initial train model')
    model = model_helper.TextParsing(hp.embedding_size, hp.max_seq_len, hp.filter_kinds,
                            hp.filters_size, hp.filter_nums, hp.classes_nums,
                            hp.dropout, word_embedding)
    print('initial dataset')
    dataset, tokenizer = data_helper.generator_batch_dataset(path, hp.batch_size)
    for (batch, (x, y)) in enumerate(dataset):
        max_seq_len = np.shape(x)[-1]
        print('max_seq_len={}'.format(max_seq_len))
        break

    test_dataset, tokenizer = data_helper.generator_batch_dataset(TEST_PATH, 
                                                                  hp.batch_size, 
                                                                  tokenizer=tokenizer, 
                                                                  max_seq_len=max_seq_len)
    print('initial optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)

    for epoch in range(hp.epoch_num):
        start = time.time()
        total_loss = 0
        accuracy = 0.0
        
        for (batch, (x, y)) in enumerate(dataset):   
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = model.get_loss(logits, y)

            batch_loss = tf.reduce_mean(loss)
            total_loss += batch_loss       
            variables = model.variables
            gradients = tape.gradient(loss, variables)        
            optimizer.apply_gradients(zip(gradients, variables)) 

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                        batch,
                                                        batch_loss))
        
        print('train epoch:{} has completed, total loss is: {} , use: {}s'.format(epoch, total_loss, time.time() - start))

        for (test_batch, (test_x, test_y)) in enumerate(test_dataset):
            test_logits = model(test_x)
            accuracy += model.get_accuracy(test_logits, test_y)
        accuracy = accuracy / (test_batch + 1)

        print('the accuracy on test dataset is: {}'.format(accuracy))

if __name__ == '__main__':
    main(DATA_PATH)