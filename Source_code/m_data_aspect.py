import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
# data
url_van_hanh_data = "/media/dykl/Store/model_aspect_03_2022/van_hanh/dataset/chuan_hoa/lan_2/full.xls"
url_dong_co_data = "/media/dykl/Store/model_aspect_03_2022/dong_co/dataset/chuan_hoa/lan_2/full.xls"
url_noi_that_data = "/media/dykl/Store/model_aspect_03_2022/noi_that/dataset/chuan_hoa/lan_2/full.xls"
url_gia_ban_data = "/media/dykl/Store/model_aspect_03_2022/gia_ban/dataset/chuan_hoa/lan_2/full.xls"
url_an_toan_data = "/media/dykl/Store/model_aspect_03_2022/an_toan/dataset/chuan_hoa/lan_1/full.xls"
url_ngoai_that_data = "/media/dykl/Store/model_aspect_03_2022/ngoai_that/dataset/chuan_hoa/lan_2/full.xls"
url_tong_the_data = "/media/dykl/Store/model_aspect_03_2022/tong_the/dataset/chuan_hoa/lan_2/full.xls"
url_raw_van_hanh_data = "/media/dykl/Store/model_aspect_03_2022/van_hanh/dataset/chua_chuan_hoa/lan_2/full.xls"
url_raw_dong_co_data = "/media/dykl/Store/model_aspect_03_2022/dong_co/dataset/chua_chuan_hoa/lan_2/full.xls"
url_raw_noi_that_data = "/media/dykl/Store/model_aspect_03_2022/noi_that/dataset/chua_chuan_hoa/lan_2/full.xls"
url_raw_ngoai_that_data = "/media/dykl/Store/model_aspect_03_2022/ngoai_that/dataset/chua_chuan_hoa/lan_2/full.xls"
url_raw_gia_ban_data = "/media/dykl/Store/model_aspect_03_2022/gia_ban/dataset/chua_chuan_hoa/lan_2/full.xls"
url_raw_an_toan_data = "/media/dykl/Store/model_aspect_03_2022/an_toan/dataset/chua_chuan_hoa/lan_2/full.xls"
url_raw_tong_the_data = "/media/dykl/Store/model_aspect_03_2022/tong_the/dataset/chua_chuan_hoa/lan_2/full.xls"
url_van_hanh_data_temp = "/media/dykl/Data/model_aspect_03_2022/van_hanh/dataset/chuan_hoa/temp/lan_1/full.xls"
# word2vec
url_word2vec_full = "/home/dykl/DeepLearning/CNN/word2vec/word2vec_full_data_200e_new290720.model"
url_word2vec_van_hanh = "/media/dykl/Store/model_aspect_03_2022/van_hanh/dataset/chuan_hoa/lan_2/vah_word_size_300.model"
url_word2vec_dong_co = "/media/dykl/Store/model_aspect_03_2022/dong_co/dataset/chuan_hoa/lan_2/doc_word_size_300.model"
url_word2vec_noi_that = "/media/dykl/Store/model_aspect_03_2022/noi_that/dataset/chuan_hoa/lan_2/not_word_size_300.model"
url_word2vec_ngoai_that = "/media/dykl/Store/model_aspect_03_2022/ngoai_that/dataset/chuan_hoa/lan_2/ngt_word_size_300.model"
url_word2vec_gia_ban = "/media/dykl/Store/model_aspect_03_2022/gia_ban/dataset/chuan_hoa/lan_2/gib_word_size_300.model"
url_word2vec_an_toan = "/media/dykl/Store/model_aspect_03_2022/an_toan/dataset/chuan_hoa/lan_1/ant_word_size_300.model"
url_word2vec_tong_the = "/media/dykl/Store/model_aspect_03_2022/tong_the/dataset/chuan_hoa/lan_2/tot_word_size_300.model"
url_raw_word2vec_van_hanh = "/media/dykl/Store/model_aspect_03_2022/van_hanh/dataset/chua_chuan_hoa/lan_2/vah_word_size_300.model"
url_raw_word2vec_dong_co = "/media/dykl/Store/model_aspect_03_2022/dong_co/dataset/chua_chuan_hoa/lan_2/doc_word_size_300.model"
url_raw_word2vec_noi_that = "/media/dykl/Store/model_aspect_03_2022/noi_that/dataset/chua_chuan_hoa/lan_2/not_word_size_300.model"
url_raw_word2vec_ngoai_that = "/media/dykl/Store/model_aspect_03_2022/ngoai_that/dataset/chua_chuan_hoa/lan_2/ngt_word_size_300.model"
url_raw_word2vec_gia_ban = "/media/dykl/Store/model_aspect_03_2022/gia_ban/dataset/chua_chuan_hoa/lan_2/gib_word_size_300.model"
url_raw_word2vec_an_toan = "/media/dykl/Store/model_aspect_03_2022/an_toan/dataset/chua_chuan_hoa/lan_2/ant_word_size_300.model"
url_raw_word2vec_tong_the = "/media/dykl/Store/model_aspect_03_2022/tong_the/dataset/chua_chuan_hoa/lan_2/tot_word_size_300.model"
url_raw_word2vec_van_hanh_temp = "/media/dykl/Data/model_aspect_03_2022/van_hanh/dataset/chuan_hoa/temp/lan_1/vah_word_size_300.model"

# params
EMBEDDING_DIM = 300
filter_sizes = [3, 4, 5]
num_filters = 298
max_length = 300
drop = 0.2
NUM_WORDS = 50000
pad = ['post', 'pre']



def load_data_forlstm():
    train_data = pd.read_excel(url_van_hanh_data, 'Sheet1')
    train_len = len(train_data)

    dic = {'vah': 0, 'other': 1}
    dense_num = len(dic)
    labels = train_data.polarity.apply(lambda x: dic[x])

    test_data = train_data[34086:35392]  # atempt 1: 34075:35394 | atempt 2: 34086:35392
    train_data = train_data.drop(test_data.index)
    val_data = train_data.sample(frac=0.2, random_state=42)
    train_data = train_data.drop(val_data.index)
    texts = train_data.text

    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
    tokenizer.fit_on_texts(texts)
    print("tokenizer Text:")
    print(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid = tokenizer.texts_to_sequences(val_data.text)
    sequences_test = tokenizer.texts_to_sequences(test_data.text)
    word_index = tokenizer.word_index
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[1])
    X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1], padding=pad[1])
    X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1], padding=pad[0])
    y_train = tf.keras.utils.to_categorical(np.asarray(labels[train_data.index]))
    y_val = tf.keras.utils.to_categorical(np.asarray(labels[val_data.index]))
    y_test = tf.keras.utils.to_categorical(np.asarray(labels[test_data.index]))

    word_vectors = KeyedVectors.load(url_word2vec_van_hanh, mmap='r')

    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

    from keras.layers import Embedding
    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=X_train.shape[1],
                                trainable=True)

    return X_train, y_train, X_test, y_test, X_val, y_val, vocabulary_size, embedding_layer, dense_num


def load_data():
    train_data = pd.read_excel(url_van_hanh_data, 'Sheet1')

    dic = {'vah': 0, 'other': 1}
    dense_num = len(dic)
    labels = train_data.polarity.apply(lambda x: dic[x])

    test_data = train_data[34086:35392] # atempt 1: 34075:35394 | atempt 2: 34086:35392
    train_data = train_data.drop(test_data.index)
    val_data = train_data.sample(frac=0.2, random_state=42)
    train_data = train_data.drop(val_data.index)
    texts = train_data.text

    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid = tokenizer.texts_to_sequences(val_data.text)
    sequences_test = tokenizer.texts_to_sequences(test_data.text)
    word_index = tokenizer.word_index

    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])
    X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1], padding=pad[0])
    X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1], padding=pad[0])
    y_train = tf.keras.utils.to_categorical(np.asarray(labels[train_data.index]))
    y_val = tf.keras.utils.to_categorical(np.asarray(labels[val_data.index]))
    y_test = tf.keras.utils.to_categorical(np.asarray(labels[test_data.index]))

    word_vectors = KeyedVectors.load(url_word2vec_van_hanh, mmap='r')

    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)

    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

    del (word_vectors)

    from keras.layers import Embedding
    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)

    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)

    return X_train, y_train, X_test, y_test, X_val, y_val, vocabulary_size, embedding_layer, dense_num
