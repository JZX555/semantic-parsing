import tensorflow as tf
import numpy as np

import model_helper
import hyper_parameter
import data_helper

from matplotlib import pyplot as plt
import time

print(tf.__version__)
print(np.__version__)

DATA_PATH = './data/rt-polarity'
TEST_PATH = './data/text_case'
PRE_TRAINNING = 'D:/dataset/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'

PLOT = True

def main(path):
    hp = hyper_parameter.HyperParam("large")
    print('initial dataset')
    dataset, tokenizer = data_helper.generator_batch_dataset(path, hp.batch_size, hp.max_seq_len)
    for (batch, (x, y)) in enumerate(dataset):
        max_seq_len = np.shape(x)[-1]
        print('max_seq_len={}'.format(max_seq_len))
        break

    test_dataset, tokenizer = data_helper.generator_batch_dataset(TEST_PATH, 
                                                                  hp.batch_size, 
                                                                  hp.max_seq_len,
                                                                  tokenizer=tokenizer)
    # print(tokenizer.to_json())


    print('initial train model')
    word_embedding = model_helper.WordEmbedding(hp.vocabulary_size, hp.embedding_size, 
                            tokenizer, PRE_TRAINNING, pre_training=False)
    model = model_helper.TextParsing(hp.embedding_size, hp.max_seq_len, hp.filter_kinds,
                            hp.filters_size, hp.filter_nums, hp.classes_nums,
                            hp.dropout, hp.regular_constrains, word_embedding)

    print('initial optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)

    print('begin training:')
    accuracy = []
    for epoch in range(hp.epoch_num):
        start = time.time()
        total_loss = 0
        
        for (batch, (x, y)) in enumerate(dataset):   
            with tf.GradientTape() as tape:
                logits = model(x)
                batch_loss = model.get_loss(logits, y, regular=True)

            total_loss += batch_loss       
            variables = model.variables
            gradients = tape.gradient(batch_loss, variables)        
            optimizer.apply_gradients(zip(gradients, variables))

            if(PLOT):
                batch_accuracys = []
                for (test_batch, (test_x, test_y)) in enumerate(test_dataset):
                    test_logits = model(test_x)
                    batch_accuracys.append(model.get_accuracy(test_logits, test_y))
                batch_accuracy = tf.reduce_mean(batch_accuracys)
                accuracy.append(batch_accuracy)

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                        batch,
                                                        batch_loss))
        
        print('train epoch:{} has completed, total loss is: {} , use: {}s'.format(epoch, total_loss, time.time() - start))

        epoch_accuracys = []
        for (test_batch, (test_x, test_y)) in enumerate(test_dataset):
            test_logits = model(test_x)
            epoch_accuracys.append(model.get_accuracy(test_logits, test_y))
        epoch_accuracy = tf.reduce_mean(epoch_accuracys)

        print('the accuracy on test dataset is: {}'.format(epoch_accuracy))

    if(PLOT):
        plt.plot(range(len(accuracy)), accuracy)
        plt.show()
    
if __name__ == '__main__':
    main(DATA_PATH)