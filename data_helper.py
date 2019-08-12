import tensorflow as tf
import numpy as np
import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

def tokenize_and_padding(text, 
                         tokenizer=None):
    if tokenizer == None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ')

    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    text = tf.keras.preprocessing.sequence.pad_sequences(text, padding='post')

    return text, tokenizer

def process_text(data_path,
                 max_seq_len,
                 tokenizer=None):
    pos_data = open(data_path + '.pos', "r", encoding='UTF-8').readlines()
    pos_data = [clean_str(text.strip()) for text in pos_data]
    pos_label = [[0, 1] for _ in pos_data]

    neg_data = open(data_path + '.neg', "r", encoding='UTF-8').readlines()
    neg_data = [clean_str(text.strip()) for text in neg_data]
    neg_label = [[1, 0] for _ in neg_data]

    text = pos_data + neg_data
    text, tokenizer = tokenize_and_padding(text, tokenizer)
    text = [tmp[:max_seq_len] for tmp in text]
    label = np.concatenate([pos_label, neg_label], 0)

    return text, label, tokenizer

def generator_batch_dataset(data_path, batch_size, max_seq_len, tokenizer=None):
    text, label, tokenizer = process_text(data_path, max_seq_len, tokenizer)
    buffer_size = len(text)
    print(np.shape(text))

    dataset = tf.data.Dataset.from_tensor_slices((text, label)).shuffle(buffer_size)
    if(max_seq_len == None):
        dataset = dataset.batch(batch_size, drop_remainder = True)
    else:
        dataset = dataset.padded_batch(batch_size, 
                                       padded_shapes=
                                           ([max_seq_len],
                                           [None])
                                       )

    return dataset, tokenizer

def get_vocab_from_tokenizer(tokenizer, vocab_size):
    dictory = tokenizer.word_index
    with open('vocab.txt',"w") as vocab:
        for (word, index) in dictory.items():
            vocab.write(word + '\n')
        
        print('generator vocab success')
        



if __name__ == "__main__":
    dataset, tokenizer = generator_batch_dataset('./data/rt-polarity', 8)
    print(dataset)
    print(tokenizer.sequences_to_texts([[10]]))
    # get_vocab_from_tokenizer(tokenizer, "")
