import os
import random
from gensim.models import doc2vec

from vectorize import news_tokenizer


def calc_length(filetype):
    wakati_gen = list(news_tokenizer.read_wakati())
    return len(wakati_gen)


def sentences():
    # wakati_gen = news_tokenizer.read_wakati()
    wakati_gen = list(news_tokenizer.read_wakati())
    random.shuffle(wakati_gen)
    for category, tokens in wakati_gen:
        yield doc2vec.LabeledSentence(tokens,
                                      tags=[category])


def main(filetype):
    dirname = './data/vector/d2v'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    print('Calculating length')
    import logging
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print('Building wakati')
    data = sentences()
    print('Building model')
    model = doc2vec.Doc2Vec(data, size=100,
                            alpha=0.0025,
                            min_alpha=0.000001,
                            window=15, min_count=1)
    word = 'microsoft'
    print(model.wv.most_similar(word))
    print('Training model epoch')
    training = 10
    for epoch in range(training):
        data = sentences()
        model.train(data, total_examples=model.corpus_count, epochs=model.iter)
        if epoch % 1 == 0:
            print(model.wv.most_similar(word))
    model.save('./data/vector/d2v/category.model')
