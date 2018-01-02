import glob
import os

from gensim.models import doc2vec

import utils as u


def sentences(path, category):
    with open(path, 'r') as f:
        one_sentence = f.readline().split(' ')
        yield doc2vec.LabeledSentence(one_sentence, tags=[category])


def main():
    dirname = './data/vector/d2v'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    read_paths = glob.glob('./data/wakati/*.wakati')
    for path in read_paths:
        category = u.extract_category(path)
        if category not in u.CATEGORIES:
            continue
        print('training ' + category)
        data = sentences(path, category)
        model = doc2vec.Doc2Vec(data, dm=0, alpha=0.025, min_alpha=0.025,
                                size=200, window=15,
                                sample=1e-6, min_count=1)
        length = sum([len(s) for s in sentences(path, category)])
        for epoch in range(50):
            model.train(data, total_examples=length, epochs=model.iter)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        model.save('./data/vector/d2v/{}.model'.format(category))
