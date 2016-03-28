# encoding: utf-8

import pickle
import numpy as np
import gensim, logging
import re

def sentence_construct(review):
    """construct the sentence for word2vec

    :review: TODO
    :returns: TODO

    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentence = []
    for i in review:
        sentence.append(i['text'])
    model = gensim.models.Word2Vec(sentence, min_count=5, workers = 8)
    return model

def combiedatafram(bussiness, review, combine_rating):
    """make a dataframe

    :bussiness: TODO
    :review: TODO
    :combine_ra: TODO
    :returns: TODO

    """
    pass


def readfile(buzi, revi, revicom):
    """read the primary process file

    :buzi: bussiness file name
    :revi: review file name
    :revicom: review combine name for markov model
    :returns: combination dataframe

    """
    with open(buzi, 'rb') as f:
        bussiness = pickle.load(f)
    with open(revi, 'rb') as f :
        review = pickle.load(f)
    combine_rating = np.load(revicom)
    return bussiness, review, combine_rating

if __name__ == "__main__":
    bussiness, review, combine_rating = readfile('bussiness_id.txt', 'review.txt', 'review_combined.npy')
    n = len(review)
    # Seperate review text as 10 parts
    model = sentence_construct(review)
    model.save('mymodel')


