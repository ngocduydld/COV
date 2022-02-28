def aspect_standardized(word_of_sent, aspect_arr, similar_arr):
    # programming language: Python    
    # aspect_arr: aspect list
    # similar_arr: standard aspect equivalent term list
    # The elements of aspect_arr and similar_arr, respectively index, are 
    # related to each other as in the ontology.
    # word_of_sent: word list in the sentence
    # output: Sentences have been aspect standardized.
    for num in range(0, len(word_of_sent)):
        for stt in range(0, len(similar_arr)):
            if word_of_sent[num] == similar_arr[stt]:
                word_of_sent[num] = aspect_arr[stt]
		aspect_term = aspect_arr[stt]
    strtext = ""
    for j in range(0, len(word_of_sent)):
        strtext = word_of_sent[j] + ' ' + strtext

    strtext = " ".join(strtext.split())
    return strtext


def generalization_sentence(word_of_sent, w_term, aspect_arr, similar_arr, sentiment_arr):
    # programming language: Python    
    # w_term: the aspect of the sentence
    # aspect_arr: aspect list
    # similar_arr: standard aspect equivalent term list
    # sentiment_arr: sentiment term list
    # The elements of aspect_arr and similar_arr, respectively index, are
    # related to each other as in the ontology.
    # output: The sentence has been generalized.
    strtext = ""
    asp_text = ""
    for num in range(0, len(word_of_sent)):
        for stt in range(0, len(similar_arr)):
            if word_of_sent[num] == similar_arr[stt] and w_term == similar_arr[stt]:
                s_term = sentiment_arr[stt]
                asp_text = aspect_arr[stt]
                for st in range(0, len(word_of_sent)):
                    if w_term == word_of_sent[st]:
                        strtext = asp_text + " " + s_term
    return strtext
