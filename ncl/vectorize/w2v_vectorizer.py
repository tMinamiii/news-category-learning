import os

from gensim.models import doc2vec

import utils as u
from vectorize import news_tokenizer


def calc_length(filetype):
    chunks = u.find_and_load_news(filetype)
    total = len(chunks)
    return total


def sentences():
    wakati_gen = news_tokenizer.read_wakati()
    for category, tokens in wakati_gen:
        yield doc2vec.LabeledSentence(tokens,
                                      tags=[category])


def main(filetype):
    dirname = './data/vector/d2v'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    print('Calculating length')
    length = calc_length(filetype)

    print('Building wakati')
    data = sentences()
    print('Building model')
    model = doc2vec.Doc2Vec(data, dm=0, alpha=0.025, min_alpha=0.025,
                            size=200, window=15,
                            sample=1e-6, min_count=1)

    print('Training model epoch')
    for epoch in range(5000):
        model.train(data, total_examples=length, epochs=model.iter)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    model.save('./data/vector/d2v/category.model')
