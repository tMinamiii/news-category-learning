import csv
import math
import pickle
import re
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse.linalg
from janome.tokenizer import Tokenizer

import mojimoji


class Token:
    '''
    ニュースのCSVを読み込んで、学習用データに変換する。
    ニュース原稿はTokenizeされ、各単語にUnique IDを割り振る。
    UIDは単なる連番で、このクラスのtoken_dicに登録された順番で決まる。
    '''

    def __init__(self, min_manuscript_len: int, min_token_len: int):
        self.loaded_csv_paths = []
        self.token_to_uid = {}
        self.uid_to_token = {}
        self.categories = set()
        self.token_seq_no = 0
        self.doc_seq_no = 0
        self.token_to_doc_uids = {}
        self.min_manuscript_len = min_manuscript_len
        self.min_token_len = min_token_len
        self.tokenized_news = []
        self.idf = {}

    def update(self, csv_paths: list):
        for path in csv_paths:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                self.loaded_csv_paths.append(path)
                for row in reader:
                    if len(row) != 4:
                        continue
                    wc = int(row[2])
                    if wc < self.min_manuscript_len:
                        continue
                    manuscript = row[3]
                    filtered = filter_manuscript(manuscript)
                    tokens = tokenize(filtered)
                    if tokens is None or len(tokens) < self.min_token_len:
                        continue
                    self.categories.add(row[0])
                    self.update_token_dics(tokens)
                    self.tokenized_news.append([row[0], tokens])
        self.update_idf()

    def update_token_dics(self, tokens: list):
        for tok in tokens:
            if tok not in self.token_to_uid:
                self.token_to_uid[tok] = self.token_seq_no
                self.uid_to_token[self.token_seq_no] = tok
                self.token_seq_no += 1
                if tok in self.token_to_doc_uids:
                    self.token_to_doc_uids[tok].add(self.doc_seq_no)
                else:
                    self.token_to_doc_uids[tok] = set([self.doc_seq_no])
        self.doc_seq_no += 1

    def update_idf(self) -> float:
        for token, _ in self.token_to_doc_uids.items():
            self.idf[token] = math.log(float(self.doc_seq_no) /
                                       len(self.token_to_doc_uids[token])) + 1


class DimensionReduction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform(self, vecs: np.array) -> np.array:
        pass


class NoReduction(DimensionReduction):
    def transform(self, vecs: np.array) -> np.array:
        return vecs


class SparseSVD(DimensionReduction):
    def __init__(self, vecs: list, k: int = None):
        if k is None:
            k = len(vecs) - 1
        _, _, self.V = scipy.sparse.linalg.svds(vecs, k=k)

    def transform(self, vecs: np.array) -> np.array:
        return np.dot(vecs, self.V.T)

    def transform_each(self, matrix: np.array) -> list:
        transform = self.transform
        dimred_matrix = [transform(vec).tolist() for vec in matrix]
        return dimred_matrix


class TfidfVectorizer:
    def __init__(self, tuid: Token,
                 dim_red: DimensionReduction = NoReduction()):
        self.tuid = tuid
        self.cat_list = list(self.tuid.categories)
        self.cat_len = len(self.tuid.categories)
        self.max_dim = self.tuid.token_seq_no
        self.dim_red = dim_red

    def vectorize(self, tokenized_news: list) -> np.array:
        train_data = []
        append = train_data.append
        for line in tokenized_news:
            tokens = line[1]
            tf_vec = self.calc_tfidf(tokens)
            category = line[0]
            category_vec = [0] * self.cat_len
            category_vec[self.cat_list.index(category)] = 1
            append((category_vec, tf_vec))

        return np.array(train_data)

    def calc_tfidf(self, tokens: list) -> np.array:
        '''
         素性に割り振られた連番のユニークIDをもとに
        TFベクトル(Term Frequency)を求める。
        '''
        tuid = self.tuid
        tf_vec = [0.0] * self.max_dim
        for tok in tokens:
            uid = tuid.token_to_uid[str(tok)]
            tf_vec[uid] += 1

        for uid in range(self.max_dim):
            token = tuid.uid_to_token[uid]
            tf_vec[uid] *= tuid.idf[token]

        return self.dim_red.transform(np.array(tf_vec) / len(tokens))


def dump(dumpdata, filepath: str) -> None:
    with open(filepath, mode='wb') as f:
        pickle.dump(dumpdata, f)
        f.flush()


def load(filepath: str):
    with open(filepath, mode='rb') as f:
        return pickle.load(f)


def filter_manuscript(manuscript: str) -> str:
    # 英文を取り除く（日本語の中の英字はそのまま）
    manuscript = re.sub(r'[a-zA-Z0-9]+[ \,\.\':;\-\+?!]', '', manuscript)
    # 記号や数字は「、」に変換する。
    # (単純に消してしまうと意味不明な長文になりjanomeがエラーを起こす)
    manuscript = re.sub(r'[0-9]+', '、', manuscript)
    manuscript = re.sub(
        r'[!"“#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}~]+', '、', manuscript)
    manuscript = re.sub(r'[（）【】『』｛｝「」［］《》〈〉]', '、', manuscript)
    return manuscript


tokenizer = Tokenizer()


def tokenize(manuscript: str) -> list:
    token_list = []
    append = token_list.append
    try:
        tokens = tokenizer.tokenize(manuscript)
    except IndexError:
        print(manuscript)
        return None
    for tok in tokens:
        ps = tok.part_of_speech.split(',')[0]
        if ps not in ['名詞', '動詞', '形容詞']:
            continue
        # 原形があれば原形をリストに入れる
        w = tok.base_form
        if w == '*' or w == '':
            # 原形がなければ表層系(原稿の単語そのまま)をリストに入れる
            w = tok.surface
        if w == '' or w == '\n':
            continue
        lower_w = mojimoji.zen_to_han(w, kana=False, digit=False)
        append(lower_w)
    return token_list
