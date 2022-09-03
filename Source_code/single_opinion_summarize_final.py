import numpy as np
import helpers as hf
import data_url as df

# labels of aspects
not_labels = ['not', 'none']
tot_labels = ['tot', 'none']
gib_labels = ['gib', 'none']
ngt_labels = ['ngt', 'none']
vah_labels = ['vah', 'none']
ant_labels = ['ant', 'none']
doc_labels = ['doc', 'none']
sentiment_labels = ['pos', 'neu', 'neg']
sentence = ""
ant_text = ""
doc_text = ""
gib_text = ""
ngt_text = ""
not_text = ""
tot_text = ""
vah_text = ""
str_summarize_final = ""

tok_sam_ant, sample_seq_ant = hf.load_full_data_an_toan()
tok_sam_doc, sample_seq_doc = hf.load_full_data_dong_co()
tok_sam_gib, sample_seq_gib = hf.load_full_data_gia_ban()
tok_sam_ngt, sample_seq_ngt = hf.load_full_data_ngoai_that()
tok_sam_not, sample_seq_not = hf.load_full_data_noi_that()
tok_sam_tot, sample_seq_tot = hf.load_full_data_tong_the()
tok_sam_vah, sample_seq_vah = hf.load_full_data_van_hanh()
tok_sam_sent, sample_seq_sent = hf.load_full_data_sentiment()
ant_model = hf.load_ant_aspect_model()
doc_model = hf.load_doc_aspect_model()
gib_model = hf.load_gib_aspect_model()
ngt_model = hf.load_ngt_aspect_model()
not_model = hf.load_not_aspect_model()
tot_model = hf.load_tot_aspect_model()
vah_model = hf.load_vah_aspect_model()
sentiment_model = hf.load_sentiment_doc_model()


def aspect_standardized_for_text(aspect_token, aspect_arr, similar_arr):
    for num in range(0, len(aspect_token)):
        for stt in range(0, len(similar_arr)):
            if aspect_token[num] == similar_arr[stt]:
                aspect_token[num] = aspect_arr[stt]
    # remove stopwprd
    for k in range(0, len(aspect_token)):
        for m in range(0, len(df.stopword)):
            if aspect_token[k] == df.stopword[m]:
                aspect_token[k] = ''

    strtext = ""
    for j in range(0, len(aspect_token)):
        if aspect_token[j] != '':
            strtext = aspect_token[j] + ' ' + strtext

    strtext = " ".join(strtext.split())  # bỏ khoảng trắng thừa
    return strtext


def standardized(str_text, str_aspect_term, str_similar):
    str_text = hf.aspect_standardized(str_text, str_similar) # phân đoạn từ
    str_text = aspect_standardized_for_text(str_text, str_aspect_term, str_similar)  # xong chuan hoa khia canh
    # Loai bo stopword
    return str_text


def building_aspect_text(str_sentence): # tạo văn bản theo khía cạnh
    is_aspect = 0
    # safety
    global ant_text
    if len(str_sentence) > 1:
        aspect_text = standardized(str_sentence, df.ant_aspect_term, df.ant_similar)  # chuan hoa khia canh cho tung cau
        aspect_pred = hf.ant_predict(aspect_text, tok_sam_ant, sample_seq_ant, ant_model)
        if aspect_pred == ['ant']:
            is_aspect = 1
            if ant_text == '':
                ant_text = aspect_text
            else:
                ant_text = ant_text + '. ' + aspect_text
    # engine
    global doc_text
    if len(str_sentence) > 1:
        aspect_text = standardized(str_sentence, df.doc_aspect_term,
                                   df.doc_similar)  # chuan hoa khia canh cho tung cau
        aspect_pred = hf.doc_predict(aspect_text, tok_sam_doc, sample_seq_doc, doc_model)
        if aspect_pred == ['doc']:
            is_aspect = 1
            if doc_text == '':
                doc_text = aspect_text
            else:
                doc_text = doc_text + '. ' + aspect_text
    # price
    global gib_text
    if len(str_sentence) > 1:
        aspect_text = standardized(str_sentence, df.gib_aspect_term,
                                   df.gib_similar)  # chuan hoa khia canh cho tung cau
        aspect_pred = hf.gib_predict(aspect_text, tok_sam_gib, sample_seq_gib, gib_model)
        if aspect_pred == ['gib']:
            is_aspect = 1
            if gib_text == '':
                gib_text = aspect_text
            else:
                gib_text = gib_text + '. ' + aspect_text
    # exterior
    global ngt_text
    if len(str_sentence) > 1:
        aspect_text = standardized(str_sentence, df.ngt_aspect_term,
                                   df.ngt_similar)  # chuan hoa khia canh cho tung cau
        aspect_pred = hf.ngt_predict(aspect_text, tok_sam_ngt, sample_seq_ngt, ngt_model)
        if aspect_pred == ['ngt']:
            is_aspect = 1
            if ngt_text == '':
                ngt_text = aspect_text
            else:
                ngt_text = ngt_text + '. ' + aspect_text
    # interior
    global not_text
    if len(str_sentence) > 1:
        aspect_text = standardized(str_sentence, df.not_aspect_term,
                                   df.not_similar)  # chuan hoa khia canh cho tung cau
        aspect_pred = hf.not_predict(aspect_text, tok_sam_not, sample_seq_not, not_model)
        if aspect_pred == ['not']:
            is_aspect = 1
            if not_text == '':
                not_text = aspect_text
            else:
                not_text = not_text + '. ' + aspect_text
    # performance
    global vah_text
    if len(str_sentence) > 1:
        aspect_text = standardized(str_sentence, df.vah_aspect_term,
                                   df.vah_similar)  # chuan hoa khia canh cho tung cau
        aspect_pred = hf.vah_predict(aspect_text, tok_sam_vah, sample_seq_vah, vah_model)
        if aspect_pred == ['vah']:
            is_aspect = 1
            if vah_text == '':
                vah_text = aspect_text
            else:
                vah_text = vah_text + '. ' + aspect_text
    # overall
    global tot_text
    if len(str_sentence) > 1:
        aspect_text = standardized(str_sentence, df.tot_aspect_term,
                                   df.tot_similar)  # chuan hoa khia canh cho tung cau
        aspect_pred = hf.tot_predict(aspect_text, tok_sam_tot, sample_seq_tot, tot_model)
        if aspect_pred == ['tot']:
            if tot_text == '':
                tot_text = aspect_text
            else:
                tot_text = tot_text + '. ' + aspect_text


def sentiment_analysis():
    # safety
    global ant_sentiment_pos, ant_sentiment_neg, ant_text, str_summarize_final
    str_summarize_final = ""
    if len(ant_text) > 1:
        pred_ant_sentiment = hf.sentiment_predict(ant_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_ant_sentiment == ['post']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Safety is positive."
            else:
                str_summarize_final = str_summarize_final + " Safety is positive."
        elif pred_ant_sentiment == ['neg']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Safety is negative."
            else:
                str_summarize_final = str_summarize_final + " Safety is negative."
        elif pred_ant_sentiment == ['neu']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Safety is neutral."
            else:
                str_summarize_final = str_summarize_final + " Safety is neutral."
    # engine
    global doc_sentiment_pos, doc_sentiment_neg, doc_text
    if len(doc_text) > 1:
        pred_doc_sentiment = hf.sentiment_predict(doc_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_doc_sentiment == ['post']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Engine is positive."
            else:
                str_summarize_final = str_summarize_final + " Engine is positive."
        elif pred_doc_sentiment == ['neg']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Engine is negative."
            else:
                str_summarize_final = str_summarize_final + " Engine is negative."
        elif pred_doc_sentiment == ['neu']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Engine is neutral."
            else:
                str_summarize_final = str_summarize_final + " Engine is neutral."
    # price
    global gib_sentiment_pos, gib_sentiment_neg, gib_text
    if len(gib_text) > 1:
        pred_gib_sentiment = hf.sentiment_predict(gib_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_gib_sentiment == ['post']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Price is positive."
            else:
                str_summarize_final = str_summarize_final + " Price is positive."
        elif pred_gib_sentiment == ['neg']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Price is negative."
            else:
                str_summarize_final = str_summarize_final + " Price is negative."
        elif pred_gib_sentiment == ['neu']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Price is neutral."
            else:
                str_summarize_final = str_summarize_final + " Price is neutral."
    # exterior
    global ngt_sentiment_pos, ngt_sentiment_neg, ngt_text
    if len(ngt_text) > 1:
        pred_ngt_sentiment = hf.sentiment_predict(ngt_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_ngt_sentiment == ['post']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Exterior is positive."
            else:
                str_summarize_final = str_summarize_final + " Exterior is positive."
        elif pred_ngt_sentiment == ['neg']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Exterior is negative."
            else:
                str_summarize_final = str_summarize_final + " Exterior is negative."
        elif pred_ngt_sentiment == ['neu']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Exterior is neutral."
            else:
                str_summarize_final = str_summarize_final + " Exterior is neutral."
    # interior
    global not_sentiment_pos, not_sentiment_neg, not_text
    if len(not_text) > 1:
        pred_not_sentiment = hf.sentiment_predict(not_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_not_sentiment == ['post']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Interior is positive."
            else:
                str_summarize_final = str_summarize_final + " Interior is positive."
        elif pred_not_sentiment == ['neg']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Interior is negative."
            else:
                str_summarize_final = str_summarize_final + " Interior is negative."
        elif pred_not_sentiment == ['neu']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Interior is neutral."
            else:
                str_summarize_final = str_summarize_final + " Interior is neutral."
    # overall
    global tot_sentiment_pos, tot_sentiment_neg, tot_text
    if len(tot_text) > 1:
        pred_tot_sentiment = hf.sentiment_predict(tot_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_tot_sentiment == ['post']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Overall is positive."
            else:
                str_summarize_final = str_summarize_final + " Overall is positive."
        elif pred_tot_sentiment == ['neg']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Overall is negative."
            else:
                str_summarize_final = str_summarize_final + " Overall is negative."
        elif pred_tot_sentiment == ['neu']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Overall is neutral."
            else:
                str_summarize_final = str_summarize_final + " Overall is neutral."
    # performance
    global vah_sentiment_pos, vah_sentiment_neg, vah_text
    if len(vah_text) > 1:
        pred_vah_sentiment = hf.sentiment_predict(vah_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_vah_sentiment == ['post']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Performance is positive."
            else:
                str_summarize_final = str_summarize_final + " Performance is positive."
        elif pred_vah_sentiment == ['neg']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Performance is negative."
            else:
                str_summarize_final = str_summarize_final + " Performance is negative."
        elif pred_vah_sentiment == ['neu']:
            if len(str_summarize_final) == 0:
                str_summarize_final = "Performance is neutral."
            else:
                str_summarize_final = str_summarize_final + " Performance is neutral."


hf.load_corpus()
while 1:
    evalSentence = input()
    if evalSentence:
        evalSentence = evalSentence.lower()
    else:
        break
    # chuan hoa khia canh
    sent_text = evalSentence
    str_temp = sent_text.lower()
    opinion_temp = hf.opinion_standardize(str_temp)  # chuan theo dau cau
    sent_of_opinion_list = opinion_temp.split('.')  # tach cau
    ant_text = ""
    doc_text = ""
    gib_text = ""
    ngt_text = ""
    not_text = ""
    tot_text = ""
    vah_text = ""
    for str_sentence in sent_of_opinion_list:  # duyet theo tung cau
        if len(str_sentence) >= 1:
            str_temp = str_sentence.lower()
            building_aspect_text(str_temp)
    sentiment_analysis()

    print(str_summarize_final)

    del evalSentence
