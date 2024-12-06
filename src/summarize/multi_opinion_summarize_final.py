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
ant_sentiment_pos = 0
ant_sentiment_neu = 0
ant_sentiment_neg = 0
doc_sentiment_pos = 0
doc_sentiment_neu = 0
doc_sentiment_neg = 0
gib_sentiment_pos = 0
gib_sentiment_neu = 0
gib_sentiment_neg = 0
ngt_sentiment_pos = 0
ngt_sentiment_neu = 0
ngt_sentiment_neg = 0
not_sentiment_pos = 0
not_sentiment_neu = 0
not_sentiment_neg = 0
tot_sentiment_pos = 0
tot_sentiment_neu = 0
tot_sentiment_neg = 0
vah_sentiment_pos = 0
vah_sentiment_neu = 0
vah_sentiment_neg = 0

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

    strtext = " ".join(strtext.split())  # remove space
    return strtext


def standardized(str_text, str_aspect_term, str_similar):
    str_text = hf.aspect_standardized(str_text, str_similar) # word chunking
    str_text = aspect_standardized_for_text(str_text, str_aspect_term, str_similar)  # xong chuan hoa khia canh
    return str_text


def building_aspect_text(str_opinion): # tạo văn bản theo khía cạnh
    is_aspect = 0
    # Neu co the chuan hoa theo khia canh nao
    opinion_temp = hf.opinion_standardize(str_opinion)  # chuan theo dau cau
    sent_of_opinion_list = opinion_temp.split('.')  # tach cau
    # safety
    global ant_text
    for ant_of_opinion in sent_of_opinion_list:
        if len(ant_of_opinion) > 1:
            aspect_text = standardized(ant_of_opinion, df.ant_aspect_term, df.ant_similar)  # chuan hoa khia canh cho tung cau
            aspect_pred = hf.ant_predict(aspect_text, tok_sam_ant, sample_seq_ant, ant_model)
            if aspect_pred == ['ant']:
                is_aspect = 1
                if ant_text == '':
                    ant_text = aspect_text
                else:
                    ant_text = ant_text + '. ' + aspect_text
    # engine
    global doc_text
    for doc_of_opinion in sent_of_opinion_list:
        if len(doc_of_opinion) > 1:
            aspect_text = standardized(doc_of_opinion, df.doc_aspect_term,
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
    for gib_of_opinion in sent_of_opinion_list:
        if len(gib_of_opinion) > 1:
            aspect_text = standardized(gib_of_opinion, df.gib_aspect_term,
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
    for ngt_of_opinion in sent_of_opinion_list:
        if len(ngt_of_opinion) > 1:
            aspect_text = standardized(ngt_of_opinion, df.ngt_aspect_term,
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
    for not_of_opinion in sent_of_opinion_list:
        if len(not_of_opinion) > 1:
            aspect_text = standardized(not_of_opinion, df.not_aspect_term,
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
    for vah_of_opinion in sent_of_opinion_list:
        if len(vah_of_opinion) > 1:
            aspect_text = standardized(vah_of_opinion, df.vah_aspect_term,
                                       df.vah_similar)  # chuan hoa khia canh cho tung cau
            aspect_pred = hf.vah_predict(aspect_text, tok_sam_vah, sample_seq_vah, vah_model)
            if aspect_pred == ['vah']:
                is_aspect = 1
                if vah_text == '':
                    vah_text = aspect_text
                else:
                    vah_text = vah_text + '. ' + aspect_text
    # overal
    global tot_text
    if is_aspect == 0:
        for tot_of_opinion in sent_of_opinion_list:
            if len(tot_of_opinion) > 1:
                aspect_text = standardized(tot_of_opinion, df.tot_aspect_term,
                                           df.tot_similar)  # chuan hoa khia canh cho tung cau
                aspect_pred = hf.tot_predict(aspect_text, tok_sam_tot, sample_seq_tot, tot_model)
                if aspect_pred == ['tot']:
                    if tot_text == '':
                        tot_text = aspect_text
                    else:
                        tot_text = tot_text + '. ' + aspect_text


def sentiment_analysis():
    # safety
    global ant_sentiment_pos, ant_sentiment_neg, ant_sentiment_neu, ant_text
    if len(ant_text) > 1:
        pred_ant_sentiment = hf.sentiment_predict(ant_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_ant_sentiment == ['post']:
            ant_sentiment_pos = ant_sentiment_pos + 1
        elif pred_ant_sentiment == ['neg']:
            ant_sentiment_neg = ant_sentiment_neg + 1
        elif pred_ant_sentiment == ['neu']:
            ant_sentiment_neu = ant_sentiment_neu + 1
    # engine
    global doc_sentiment_pos, doc_sentiment_neg, doc_sentiment_neu, doc_text
    if len(doc_text) > 1:
        pred_doc_sentiment = hf.sentiment_predict(doc_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_doc_sentiment == ['post']:
            doc_sentiment_pos = doc_sentiment_pos + 1
        elif pred_doc_sentiment == ['neg']:
            doc_sentiment_neg = doc_sentiment_neg + 1
        elif pred_doc_sentiment == ['neu']:
            doc_sentiment_neu = doc_sentiment_neu + 1
    # price
    global gib_sentiment_pos, gib_sentiment_neg, gib_sentiment_neu, gib_text
    if len(gib_text) > 1:
        pred_gib_sentiment = hf.sentiment_predict(gib_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_gib_sentiment == ['post']:
            gib_sentiment_pos = gib_sentiment_pos + 1
        elif pred_gib_sentiment == ['neg']:
            gib_sentiment_neg = gib_sentiment_neg + 1
        elif pred_gib_sentiment == ['neu']:
            gib_sentiment_neu = gib_sentiment_neu + 1
    # exterior
    global ngt_sentiment_pos, ngt_sentiment_neg, ngt_sentiment_neu, ngt_text
    if len(ngt_text) > 1:
        pred_ngt_sentiment = hf.sentiment_predict(ngt_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_ngt_sentiment == ['post']:
            ngt_sentiment_pos = ngt_sentiment_pos + 1
        elif pred_ngt_sentiment == ['neg']:
            ngt_sentiment_neg = ngt_sentiment_neg + 1
        elif pred_ngt_sentiment == ['neu']:
            ngt_sentiment_neu = ngt_sentiment_neu + 1
    # interior
    global not_sentiment_pos, not_sentiment_neg, not_sentiment_neu, not_text
    if len(not_text) > 1:
        pred_not_sentiment = hf.sentiment_predict(not_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_not_sentiment == ['post']:
            not_sentiment_pos = not_sentiment_pos + 1
        elif pred_not_sentiment == ['neg']:
            not_sentiment_neg = not_sentiment_neg + 1
        elif pred_not_sentiment == ['neu']:
            not_sentiment_neu = not_sentiment_neu + 1
    # overal
    global tot_sentiment_pos, tot_sentiment_neg, tot_sentiment_neu, tot_text
    if len(tot_text) > 1:
        pred_tot_sentiment = hf.sentiment_predict(tot_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_tot_sentiment == ['post']:
            tot_sentiment_pos = tot_sentiment_pos + 1
        elif pred_tot_sentiment == ['neg']:
            tot_sentiment_neg = tot_sentiment_neg + 1
        elif pred_tot_sentiment == ['neu']:
            tot_sentiment_neu = tot_sentiment_neu + 1
    # performance
    global vah_sentiment_pos, vah_sentiment_neg, vah_sentiment_neu, vah_text
    if len(vah_text) > 1:
        pred_vah_sentiment = hf.sentiment_predict(vah_text, tok_sam_sent, sample_seq_sent, sentiment_model)
        if pred_vah_sentiment == ['post']:
            vah_sentiment_pos = vah_sentiment_pos + 1
        elif pred_vah_sentiment == ['neg']:
            vah_sentiment_neg = vah_sentiment_neg + 1
        elif pred_vah_sentiment == ['neu']:
            vah_sentiment_neu = vah_sentiment_neu + 1


hf.load_corpus()
for opinion in df.opinion_list:
    ant_text = ""
    doc_text = ""
    gib_text = ""
    ngt_text = ""
    not_text = ""
    tot_text = ""
    vah_text = ""
    opinion_temp = hf.opinion_standardize(opinion)  # chuan theo dau cau
    sent_of_opinion_list = opinion_temp.split('.')  # tach cau
    for sent_of_opinion in sent_of_opinion_list:
        str_temp = sent_of_opinion.lower()
        building_aspect_text(str_temp)
    sentiment_analysis()

print("The number of opinions: "+str(len(df.opinion_list)))
str_final = "Safety: "+str(ant_sentiment_pos+ant_sentiment_neu+ant_sentiment_neg)+" opinions - "
str_final=str_final+str(round((ant_sentiment_pos*100)/(ant_sentiment_pos+ant_sentiment_neu+ant_sentiment_neg),2))+" % positive, "
str_final=str_final+str(round((ant_sentiment_neg*100)/(ant_sentiment_pos+ant_sentiment_neu+ant_sentiment_neg),2))+" % negative. "
str_final=str_final+"\nEngine: "+str(doc_sentiment_pos+doc_sentiment_neu+doc_sentiment_neg)+" opinions - "
str_final=str_final+str(round((doc_sentiment_pos*100)/(doc_sentiment_pos+doc_sentiment_neu+doc_sentiment_neg),2))+" % positive, "
str_final=str_final+str(round((doc_sentiment_neg*100)/(doc_sentiment_pos+doc_sentiment_neu+doc_sentiment_neg),2))+" % negative. "
str_final=str_final+"\nPrice: "+str(gib_sentiment_pos+gib_sentiment_neu+gib_sentiment_neg)+" opinions - "
str_final=str_final+str(round((gib_sentiment_pos*100)/(gib_sentiment_pos+gib_sentiment_neu+gib_sentiment_neg),2))+" % positive, "
str_final=str_final+str(round((gib_sentiment_neg*100)/(gib_sentiment_pos+gib_sentiment_neu+gib_sentiment_neg),2))+" % negative. "
str_final=str_final+"\nExterior: "+str(ngt_sentiment_pos+ngt_sentiment_neu+ngt_sentiment_neg)+" opinions - "
str_final=str_final+str(round((ngt_sentiment_pos*100)/(ngt_sentiment_pos+ngt_sentiment_neu+ngt_sentiment_neg),2))+" % positive, "
str_final=str_final+str(round((ngt_sentiment_neg*100)/(ngt_sentiment_pos+ngt_sentiment_neu+ngt_sentiment_neg),2))+" % negative. "
str_final=str_final+"\nInterior: "+str(not_sentiment_pos+not_sentiment_neu+not_sentiment_neg)+" opinions - "
str_final=str_final+str(round((not_sentiment_pos*100)/(not_sentiment_pos+not_sentiment_neu+not_sentiment_neg),2))+" % positive, "
str_final=str_final+str(round((not_sentiment_neg*100)/(not_sentiment_pos+not_sentiment_neu+not_sentiment_neg),2))+" % negative. "
str_final=str_final+"\nPerformance: "+str(vah_sentiment_pos+vah_sentiment_neu+vah_sentiment_neg)+" opinions - "
str_final=str_final+str(round((vah_sentiment_pos*100)/(vah_sentiment_pos+vah_sentiment_neu+vah_sentiment_neg),2))+" % positive, "
str_final=str_final+str(round((vah_sentiment_neg*100)/(vah_sentiment_pos+vah_sentiment_neu+vah_sentiment_neg),2))+" % negative. "
str_final=str_final+"\nOveral: "+str(tot_sentiment_pos+tot_sentiment_neu+tot_sentiment_neg)+" opinions - "
str_final=str_final+str(round((tot_sentiment_pos*100)/(tot_sentiment_pos+tot_sentiment_neu+tot_sentiment_neg),2))+" % positive, "
str_final=str_final+str(round((tot_sentiment_neg*100)/(tot_sentiment_pos+tot_sentiment_neu+tot_sentiment_neg),2))+" % negative. "
print(str_final)
