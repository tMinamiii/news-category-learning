import csv
import math
import pickle
import random
from collections import Counter

import numpy as np
from sklearn.decomposition import IncrementalPCA
from vectorize.news_tokenizer import YahooNewsTokenizer


class Preprocessor:
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

    def append(self, csv_paths: list,
               min_manuscript_len: int,
               min_token_len: int):
        for path in csv_paths:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                self.loaded_csv_paths.append(path)
                for row in reader:
                    if len(row) != 4:
                        continue
                    category, _, length, manuscript = row
                    if int(length) < min_manuscript_len:
                        continue
                    tokenizer = YahooNewsTokenizer()
                    sanitized = tokenizer.sanitize(manuscript)
                    tokens = tokenizer.tokenize(sanitized)
                    if tokens is None or len(tokens) < min_token_len:
                        continue
                    # Counterで単語の出現頻度を数える
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


class PCATfidfVectorizer:
    '''
    TF-IDFベクトルを作成し主成分分析(IncrementalPCA)を適用するクラス
    '''

    def __init__(self, prep: Preprocessor, dimension: int):
        self.prep = prep
        self.cat_list = list(self.prep.categories)
        self.cat_len = len(self.prep.categories)
        self.max_dim = self.prep.token_seq_no
        self.ipca = IncrementalPCA(n_components=dimension)

    def fit(self, tokenized_news: list, batch_size: int) -> None:
        '''
          ニュース原稿のtfidfの主成分をbatch_sizeで指定した分ずつ求めていく
        '''
        news_len = len(tokenized_news)
        random.shuffle(tokenized_news)
        for i in range(0, news_len, batch_size):
            chunks = tokenized_news[i:i + batch_size]
            mat = np.array([self.tfidf(c) for _, c in chunks])
            self.ipca.partial_fit(mat)

    def vectorize(self, tokenized_news: list) -> np.array:
        '''
         ニュース原稿のTF-IDFベクトルを求めたのち主成分分析で次元削減する
        '''
        data = []
        for category, counter in tokenized_news:
            category_vec = self.prep.category_dic[category]
            vec = self.tfidf(counter).reshape(1, -1)
            # transformの結果は2重リストになっているので、最初の要素を取り出す
            dimred_tfidf = self.ipca.transform(vec)[0]
            data.append((category_vec, dimred_tfidf))
        return np.array(data)

    def tfidf(self, token_counter: Counter) -> np.array:
        '''
        TF-IDFベクトルを求める。
        '''
        tf_vec = np.zeros(self.max_dim)
        total_tokens = sum(token_counter.values())
        for token, count in token_counter.items():
            uid = self.prep.token_to_id[token]
            tf_vec[uid] = count / total_tokens * self.prep.idf[token]

        return tf_vec


def dump(dumpdata, filepath: str) -> None:
    with open(filepath, mode='wb') as f:
        pickle.dump(dumpdata, f)
        f.flush()


def load(filepath: str):
    with open(filepath, mode='rb') as f:
        return pickle.load(f)

