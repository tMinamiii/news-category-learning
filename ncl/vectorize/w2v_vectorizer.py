import os

from gensim.models import doc2vec

import utils as u
from vectorize.news_tokenizer import YahooNewsTokenizer


def sentences(filetype):
    chunks = u.find_and_load_news('csv')
    for ck in chunks:
        category = ck['category']
        ynt = YahooNewsTokenizer()
        sanitize = ynt.sanitize(ck['manuscript'])
        tokens = ynt.tokenize(sanitize)
        yield doc2vec.LabeledSentence(tokens,
                                      tags=[category])


def main(filetype):
    dirname = './data/vector/d2v'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    data = sentences(filetype)
    model = doc2vec.Doc2Vec(data, dm=0, alpha=0.025, min_alpha=0.025,
                            size=200, window=15,
                            sample=1e-6, min_count=1)
    length = sum([len(s) for s in sentences()])
    for epoch in range(50):
        model.train(data, total_examples=length, epochs=model.iter)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    model.save('./data/vector/d2v/category.model')
