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

def tokenize_and_padding(data, 
                         tokenizer=None):
    if tokenizer == None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    data = tf.keras.preprocessing.sequence.pad_sequences(data, padding='post')

    return data, tokenizer

def process_text(data_path):
    pos_data = open(data_path + '.pos', "r", encoding='UTF-8').readlines()
    pos_data = [clean_str(text.strip()) for text in pos_data]
    pos_label = [[0, 1] for _ in pos_data]

    neg_data = open(data_path + '.neg', "r", encoding='UTF-8').readlines()
    neg_data = [clean_str(text.strip()) for text in neg_data]
    neg_label = [[1, 0] for _ in neg_data]

    data = pos_data + neg_data
    data, tokenizer = tokenize_and_padding(data)
    label = np.concatenate([pos_label, neg_label], 0)

    return data, label, tokenizer

if __name__ == "__main__":
    data, label, tokenizer = process_text('./data/text_case')
    print(data)
    print(label)