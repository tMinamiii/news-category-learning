import os
import random
from gensim.models import doc2vec

from vectorize import news_tokenizer
import utils as u


def calc_length(filetype):
    wakati_gen = list(news_tokenizer.read_wakati())
    return len(wakati_gen)


def sentences(wakati_list):
    for category, tokens in wakati_list:
        yield doc2vec.LabeledSentence(tokens,
                                      tags=[category])


def divid_data():
    wakati_list = list(news_tokenizer.read_wakati())
    random.shuffle(wakati_list)
    train_length = int(len(wakati_list) * u.TRAINING_DATA_RATIO)
    print(train_length)
    return (wakati_list[0:train_length], wakati_list[train_length + 1:])


def accuracy(model, wakati_list):
    failed_count = 0
    for category, tokens in wakati_list:
        known_tokens = tokens
        # infer_vectorならボキャブラリー外の単語があってもエラーにはならない
        infer_vector = model.infer_vector(known_tokens)
        sim_doc = model.docvecs.most_similar([infer_vector])
        if category != sim_doc[0][0] and category != sim_doc[1][0]:
            failed_count += 1
    return 1 - failed_count / len(wakati_list)


def main(filetype):
    dirname = './data/vector/d2v'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    print('Calculating length')
    '''
    import logging
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    '''
    print('Building wakati')
    train, test = divid_data()
    data = sentences(train)
    print('Building model')
    model = doc2vec.Doc2Vec(data,
                            size=200,
                            alpha=0.0002,
                            min_alpha=0.000001,
                            window=15,
                            workers=8)
    print('Training model epoch')
    training = 30
    test_words = ['iphone', 'サッカー', 'apple', '憲法', '技術']
    for epoch in range(training):
        data = sentences(train)
        model.train(data, total_examples=model.corpus_count, epochs=model.iter)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print('epoch {} ======================'.format(epoch))
            for w in test_words:
                print('\t{0}\t=> {1}'.format(
                    w, model.wv.most_similar(w, topn=3)))
            print('accuracy rate: {}\n'.format(accuracy(model, test)))
    model.save('./data/vector/d2v/category.model')
