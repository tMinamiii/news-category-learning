import csv
import glob
import json
import math
import os
import random
from collections import Counter

import numpy as np
from sklearn.decomposition import IncrementalPCA

import utils as u
from vectorize.news_tokenizer import YahooNewsTokenizer


class Metadata:
    '''
    ニュースのCSVを読み込んで、学習用データの前処理を施していく。
    ニュース原稿を形態素解析にかけ、Counterを用いて出現頻度を数えておく。
    また各単語のIDFも前処理で計算して連想配列に保存しておく。
    '''

    def __init__(self):
        self.loaded_csv_paths = []
        self.token_to_id = {}
        # self.id_to_token = {}
        self.categories = set()
        self.category_dic = {}
        self.token_seq_no = 0
        self.doc_seq_no = 0
        self.token_to_docid = {}
        self.tokenized_news = []
        self.idf = {}

    def append(self, news_chunks: list,
               min_manuscript_len: int,
               min_token_len: int):
        ynt = YahooNewsTokenizer()
        for chunk in news_chunks:
            category = chunk['category']
            manuscript_len = chunk['manuscript_len']
            manuscript = chunk['manuscript']
            if min_manuscript_len >= manuscript_len:
                continue
            sanitized = ynt.sanitize(manuscript)
            tokens = ynt.tokenize(sanitized)
            if min_token_len >= len(tokens):
                continue
            token_counter = Counter(tokens)
            self.categories.add(category)
            self.update_token_dics(token_counter)
            self.tokenized_news.append([category, token_counter])
        self.update_idf()
        self.update_category_dic()

    def update_category_dic(self):
        cat_list = list(self.categories)
        cat_len = len(self.categories)
        for cat in cat_list:
            category_vec = [0] * cat_len
            category_vec[cat_list.index(cat)] = 1
            self.category_dic[cat] = category_vec

    def update_token_dics(self, token_counter: Counter):
        for tok, _ in token_counter.items():
            if tok not in self.token_to_id:
                self.token_to_id[tok] = self.token_seq_no
                # self.id_to_token[self.token_seq_no] = tok
                self.token_seq_no += 1
            self.token_to_docid.setdefault(tok, set()).add(self.doc_seq_no)
        self.doc_seq_no += 1

    def update_idf(self) -> dict:
        self.idf = {token: self.__idf(docids)
                    for token, docids in self.token_to_docid.items()}

    def __idf(self, docids) -> float:
        return math.log(self.doc_seq_no / len(docids)) + 1


def tfidf(meta: Metadata, token_counter: Counter):
    '''
    TF-IDFベクトルを作成
    '''

    max_dim = meta.token_seq_no
    tf_vec = np.zeros(max_dim)
    total_tokens = sum(token_counter.values())
    for token, count in token_counter.items():
        uid = meta.token_to_id[token]
        tf_vec[uid] = count / total_tokens * meta.idf[token]

    return tf_vec


class PcaTfidfVectorizer:
    def __init__(self, meta: Metadata):
        self.meta = meta

    def incremental_fit(self, tokenized_news):
        ipca = IncrementalPCA(n_components=u.PCA_DIMENSION)

        '''
          ニュース原稿のtfidfの主成分をbatch_sizeで指定した分ずつ求めていく
        '''
        news_len = len(tokenized_news)
        random.shuffle(tokenized_news)
        batch = u.PCA_BATCH_DATA_LENGTH
        for i in range(0, news_len, batch):
            chunks = tokenized_news[i:i + batch]
            mat = np.array([tfidf(self.meta, c) for _, c in chunks])
            ipca.partial_fit(mat)
        return ipca

    def vectorize(self, tokenized_news: list) -> np.array:
        '''
         ニュース原稿のTF-IDFベクトルを求めたのち主成分分析で次元削減する
        '''
        ipca = self.incremental_fit(tokenized_news)
        data = []
        for category, counter in tokenized_news:
            category_vec = self.meta.category_dic[category]
            vec = tfidf(self.meta, counter).reshape(1, -1)
            # transformの結果は2重リストになっているので、最初の要素を取り出す
            dimred_tfidf = ipca.transform(vec)[0]
            data.append((category_vec, dimred_tfidf))
        return np.array(data)



def main(filetype='json'):
    all_chunks = u.find_and_load_news(filetype)

    meta = Metadata()
    meta.append(all_chunks, min_manuscript_len=u.MINIMUM_MANUSCRIPT_LENGTH,
                min_token_len=u.MINIMUM_TOKEN_LENGTH)
    print('TFIDF calculated')

    pca_tfidf = PcaTfidfVectorizer(meta)
    print('IncrementPCA fitting finished')

    learning_data = pca_tfidf.vectorize(meta.tokenized_news)
    print('vectorizing finished')

    dirname = './data/vector/tfidf/'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    u.pickle_dump(meta, dirname + 'tfidf.meta')
    print('Meta data was dumped.')

    u.pickle_dump(learning_data, dirname + 'tfidf.data')
    print('Learning data was dumped.')
