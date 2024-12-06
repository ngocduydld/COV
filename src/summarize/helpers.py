import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from gensim.models.keyedvectors import KeyedVectors
import data_url as df

# params
EMBEDDING_DIM = 300
filter_sizes = [3, 4, 5]
num_filters = 298
max_length = 300
drop = 0.2
NUM_WORDS = 50000
L2 = 0.0004
pad = ['post', 'pre']


def load_corpus():
    stopword_data = pd.read_excel(df.url_stopword_data, 'Sheet1')
    for i in range(0, len(stopword_data)):
        df.stopword.append(stopword_data.text[i])

    opinion_data = pd.read_excel(df.url_opinion_data, 'Data')
    for i in range(0, len(opinion_data)):
        df.opinion_list.append(opinion_data.opinion[i])

    # phan tich khia canh
    ant_aspect_data = pd.read_excel(df.url_ant_aspect_data, 'Sheet1')
    for i in range(0, len(ant_aspect_data)):
        df.ant_aspect_term.append(ant_aspect_data.aspect_item[i])
        df.ant_similar.append(ant_aspect_data.similar_item[i])

    doc_aspect_data = pd.read_excel(df.url_doc_aspect_data, 'Sheet1')
    for i in range(0, len(doc_aspect_data)):
        df.doc_aspect_term.append(doc_aspect_data.aspect_item[i])
        df.doc_similar.append(doc_aspect_data.similar_item[i])

    gib_aspect_data = pd.read_excel(df.url_gib_aspect_data, 'Sheet1')
    for i in range(0, len(gib_aspect_data)):
        df.gib_aspect_term.append(gib_aspect_data.aspect_item[i])
        df.gib_similar.append(gib_aspect_data.similar_item[i])

    ngt_aspect_data = pd.read_excel(df.url_ngt_aspect_data, 'Sheet1')
    for i in range(0, len(ngt_aspect_data)):
        df.ngt_aspect_term.append(ngt_aspect_data.aspect_item[i])
        df.ngt_similar.append(ngt_aspect_data.similar_item[i])

    not_aspect_data = pd.read_excel(df.url_not_aspect_data, 'Sheet1')
    for i in range(0, len(not_aspect_data)):
        df.not_aspect_term.append(not_aspect_data.aspect_item[i])
        df.not_similar.append(not_aspect_data.similar_item[i])

    tot_aspect_data = pd.read_excel(df.url_tot_aspect_data, 'Sheet1')
    for i in range(0, len(tot_aspect_data)):
        df.tot_aspect_term.append(tot_aspect_data.aspect_item[i])
        df.tot_similar.append(tot_aspect_data.similar_item[i])

    vah_aspect_data = pd.read_excel(df.url_vah_aspect_data, 'Sheet1')
    for i in range(0, len(vah_aspect_data)):
        df.vah_aspect_term.append(vah_aspect_data.aspect_item[i])
        df.vah_similar.append(vah_aspect_data.similar_item[i])


def opinion_standardize(text):
    str = ""
    pattern = '\d+[\.,]\d+'
    # text = text.replace("\n", ",")
    text = " ".join(text.split())
    text = text.replace("(.)", ".")
    text = text.replace("(...)", ".")
    text = text.replace("...", ".")
    text = text.replace("..", ".")
    text = text.replace("!", ".")
    text = text.replace("?", ".")
    text = text.replace(".)", ").")
    text = text.replace(":", ",")
    text = text.replace(";", ".")
    text = text.replace("\"", "")
    # loai bo dau khogn dung den
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("*", " ")
    text = text.replace(",", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("|", " ")
    text = text.replace("km/h", " km h ")
    text = text.replace("km", " km ")
    text = text.replace("rpm", " rpm ")
    text = text.replace("usd", " usd ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("vòng/phút", " vòng phút ")

    strSub = re.findall(pattern, text)
    if len(strSub) > 0:
        for item in strSub:
            str = item
            str = str.replace(".", ",")
            text = text.replace(item, str)

    return text


def load_ant_aspect_model():
    json_file = open(df.ant_json_file_bilstm, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.ant_weight_file_bilstm)
    return model


def load_doc_aspect_model():
    json_file = open(df.doc_json_file_bilstm, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.doc_weight_file_bilstm)
    return model


def load_gib_aspect_model():
    json_file = open(df.gib_json_file_bilstm, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.gib_weight_file_bilstm)
    return model


def load_ngt_aspect_model():
    json_file = open(df.ngt_json_file_bilstm, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.ngt_weight_file_bilstm)
    return model


def load_not_aspect_model():
    json_file = open(df.not_json_file_bilstm, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.not_weight_file_bilstm)
    return model


def load_tot_aspect_model():
    json_file = open(df.tot_json_file_bilstm, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.tot_weight_file_bilstm)
    return model


def load_vah_aspect_model():
    json_file = open(df.vah_json_file_bilstm, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.vah_weight_file_bilstm)
    return model


def load_aspect_model(str_json, str_weight):
    json_file = open(str_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(str_weight)
    return model


def load_full_data_an_toan():
    train_data = pd.read_excel(df.url_an_toan_data, 'Sheet1')
    test_data = train_data[34161:35463]  # lan 1: 34166:35465 | lan 2: 34161:35463
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_full_data_dong_co():
    train_data = pd.read_excel(df.url_dong_co_data, 'Sheet1')
    test_data = train_data[34133:35464]  # lan 1: 34143:35464 | lan 2: 34133:35464
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_full_data_gia_ban():
    train_data = pd.read_excel(df.url_gia_ban_data, 'Sheet1')
    test_data = train_data[34062:35463]  # lan_1: 34062:35463 | lan 2: 34062:35463
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_full_data_ngoai_that():
    train_data = pd.read_excel(df.url_ngoai_that_data, 'Sheet1')
    test_data = train_data[34016:35463]  # lan 1: 34016:35463 | lan 2: 34016:35463
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_full_data_noi_that():
    train_data = pd.read_excel(df.url_noi_that_data, 'Sheet1')
    test_data = train_data[34143:35463]  # lan 1: 34162:35463 | lan 2: 34143:35463
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_full_data_tong_the():
    train_data = pd.read_excel(df.url_tong_the_data, 'Sheet1')
    test_data = train_data[33702:35463]  # lan 1: 33702:35463 | lan 2: 33702:35463
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_full_data_van_hanh():
    train_data = pd.read_excel(df.url_van_hanh_data, 'Sheet1')
    test_data = train_data[34075:35394]  # lan 2: 34086:35392 | lan 1: 34075:35394
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_full_data(text, str_data, str_wv):
    text_temp = text
    train_data = pd.read_excel(str_data, 'Sheet1')

    test_data = train_data[34162:35463] # lan 1: 34162:35463 | lan 2: 34143:35463
    train_data = train_data.drop(test_data.index)

    texts = train_data.text

    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
    tokenizer.fit_on_texts(texts)

    sequences_train = tokenizer.texts_to_sequences(texts)
    eval_text = tokenizer.texts_to_sequences(text_temp)
    word_index = tokenizer.word_index

    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])
    text_test = pad_sequences(eval_text, maxlen=X_train.shape[1], padding=pad[0])

    word_vectors = KeyedVectors.load(str_wv, mmap='r')

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

    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)

    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)

    return text_test


def chunk_original(str, arrPolarityTerm):
    # Maximum_Matching
    vTerm = []
    strRemain = ""
    start = 0
    isTerm = False
    isStop = False

    str = str.lower()
    str = str.lstrip(" ").rstrip(" ")
    WordList = str.split(" ")
    stop = len(WordList)

    while (isStop == False and stop >= 0):
        for num in range(start, stop):
            strRemain = strRemain + WordList[num] + " "

        strRemain = strRemain.lstrip(" ").rstrip(" ").lower()
        isTerm = False
        for cha in range(0, len(arrPolarityTerm)):
            arr = arrPolarityTerm[cha]
            if (arr == strRemain):
                vTerm.append(strRemain)
                isTerm = True
                if (start == 0):
                    isStop = True
                else:
                    stop = start
                    start = 0

        if (isTerm == False):
            if (start == stop):
                vTerm.append(strRemain)
                stop = stop - 1
                start = 0
            else:
                start += 1
        strRemain = ""
    strRemain = ""
    for stt in range(0, len(vTerm)):
        strRemain = strRemain + " " + vTerm[stt]

    return strRemain


def chunk_feature(str, arrPolarityTerm):
    # Maximum_Matching
    vTerm = []
    strRemain = ""
    start = 0
    isTerm = False
    isStop = False

    str = str.lower()
    str = str.lstrip(" ").rstrip(" ")
    WordList = str.split(" ")
    stop = len(WordList)

    while (isStop == False and stop >= 0):
        for num in range(start, stop):
            strRemain = strRemain + WordList[num] + " "

        strRemain = strRemain.lstrip(" ").rstrip(" ").lower()
        isTerm = False
        for cha in range(0, len(arrPolarityTerm)):
            arr = arrPolarityTerm[cha]
            if (arr == strRemain):
                vTerm.append(strRemain)
                isTerm = True
                if (start == 0):
                    isStop = True
                else:
                    stop = start
                    start = 0

        if (isTerm == False):
            if (start == stop):
                stop = stop - 1
                start = 0
            else:
                start += 1

        strRemain = ""
    strRemain = ""
    for stt in range(0, len(vTerm)):
        strRemain = strRemain + " " + vTerm[stt]

    return vTerm


def aspect_standardized(str, arrSimilarTerm):
    # Maximum_Matching
    vTerm = []
    strRemain = ""
    start = 0
    isTerm = False
    isStop = False

    str = str.lower()
    str = str.lstrip(" ").rstrip(" ")
    WordList = str.split(" ")
    stop = len(WordList)

    while (isStop == False and stop > 0):
        for num in range(start, stop):
            strRemain = strRemain + WordList[num] + " "

        strRemain = strRemain.lstrip(" ").rstrip(" ").lower()
        isTerm = False
        for cha in range(0, len(arrSimilarTerm)):
            arr = arrSimilarTerm[cha]
            if (arr == strRemain):
                vTerm.append(strRemain)
                isTerm = True
                if (start == 0):
                    isStop = True
                else:
                    stop = start
                    start = 0

        if (isTerm == False):
            if (start == stop - 1):
                vTerm.append((strRemain))
                stop = stop - 1
                start = 0
            else:
                start += 1

        strRemain = ""
    strRemain = ""
    for stt in range(0, len(vTerm)):
        strRemain = strRemain + " " + vTerm[stt]
    return vTerm

def ant_predict(str, tok_sam_ant, sample_seq_ant, ant_model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # ant
    if len(str_temp)>1:
        sentences.append(str_temp)
        ant_text = tok_sam_ant.texts_to_sequences(sentences)
        ant_seq = pad_sequences(ant_text, maxlen=sample_seq_ant.shape[1], padding='post')
        pred_ant = ant_model.predict(ant_seq)
        temp_aspect_detect = df.ant_labels[np.argmax(pred_ant)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def doc_predict(str, tok_sam_doc, sample_seq_doc, doc_model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # ant
    if len(str_temp)>1:
        sentences.append(str_temp)
        doc_text = tok_sam_doc.texts_to_sequences(sentences)
        doc_seq = pad_sequences(doc_text, maxlen=sample_seq_doc.shape[1], padding='post')
        pred_doc = doc_model.predict(doc_seq)
        temp_aspect_detect = df.doc_labels[np.argmax(pred_doc)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def gib_predict(str, tok_sam_gib, sample_seq_gib, gib_model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # ant
    if len(str_temp)>1:
        sentences.append(str_temp)
        gib_text = tok_sam_gib.texts_to_sequences(sentences)
        gib_seq = pad_sequences(gib_text, maxlen=sample_seq_gib.shape[1], padding='post')
        pred_gib = gib_model.predict(gib_seq)
        temp_aspect_detect = df.gib_labels[np.argmax(pred_gib)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def ngt_predict(str, tok_sam_ngt, sample_seq_ngt, ngt_model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # ngt
    if len(str_temp)>1:
        sentences.append(str_temp)
        ngt_text = tok_sam_ngt.texts_to_sequences(sentences)
        ngt_seq = pad_sequences(ngt_text, maxlen=sample_seq_ngt.shape[1], padding='post')
        pred_ngt = ngt_model.predict(ngt_seq)
        temp_aspect_detect = df.ngt_labels[np.argmax(pred_ngt)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def not_predict(str, tok_sam_not, sample_seq_not, not_model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # not
    if len(str_temp)>1:
        sentences.append(str_temp)
        not_text = tok_sam_not.texts_to_sequences(sentences)
        not_seq = pad_sequences(not_text, maxlen=sample_seq_not.shape[1], padding='post')
        pred_not = not_model.predict(not_seq)
        temp_aspect_detect = df.not_labels[np.argmax(pred_not)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def vah_predict(str, tok_sam_vah, sample_seq_vah, vah_model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # vah
    if len(str_temp)>1:
        sentences.append(str_temp)
        vah_text = tok_sam_vah.texts_to_sequences(sentences)
        vah_seq = pad_sequences(vah_text, maxlen=sample_seq_vah.shape[1], padding='post')
        pred_vah = vah_model.predict(vah_seq)
        temp_aspect_detect = df.vah_labels[np.argmax(pred_vah)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def tot_predict(str, tok_sam_tot, sample_seq_tot, tot_model):
    str_temp = " ".join(str.split())
    sentences = []
    aspect_detect = []
    # tot
    if len(str_temp)>1:
        sentences.append(str_temp)
        tot_text = tok_sam_tot.texts_to_sequences(sentences)
        tot_seq = pad_sequences(tot_text, maxlen=sample_seq_tot.shape[1], padding='post')
        pred_tot = tot_model.predict(tot_seq)
        temp_aspect_detect = df.tot_labels[np.argmax(pred_tot)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def sentiment_predict(str, tok_sam_sent, sample_seq_sent, sent_model):
    sentences = []
    aspect_detect = []
    if len(str)>1:
        str_temp = str.replace('.', '')
        sentences.append(str_temp)
        print(sentences)
        sent_text = tok_sam_sent.texts_to_sequences(sentences)
        sent_seq = pad_sequences(sent_text, maxlen=sample_seq_sent.shape[1], padding='post')
        pred_sentiment = sent_model.predict(sent_seq)
        temp_aspect_detect = df.sentiment_labels[np.argmax(pred_sentiment)]
        aspect_detect.append(temp_aspect_detect)
    return aspect_detect


def load_full_data_sentiment():
    train_data = pd.read_excel(df.url_sentiment_doc_data, 'Sheet1')
    test_data = train_data[37270:38171]
    train_data = train_data.drop(test_data.index)
    texts = train_data.text
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                                                      lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])

    return tokenizer, X_train


def load_sentiment_doc_model():
    json_file = open(df.sentiment_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(df.sentiment_weight_file)
    return model
