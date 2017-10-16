import csv
import pickle
import re
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse.linalg
from janome.tokenizer import Tokenizer

import mojimoji


class TokenUID:
    '''
    ニュースのCSVを読み込んで、学習用データに変換する。
    ニュース原稿はTokenizeされ、各単語にUnique IDを割り振る。
    UIDは単なる連番で、このクラスのtoken_dicに登録された順番で決まる。
    '''

    def __init__(self):
        self.loaded_csv_list = []
        self.token_dic = {}
        self.categories = set()
        self.seq_no_uid = 0

    def update(self, csv_list: list):
        for csv_path in csv_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                self.loaded_csv_list.append(csv_path)
                for row in reader:
                    if len(row) != 4:
                        continue
                    manuscript = row[3]
                    try:
                        token_list = tokenize(manuscript)
                    except IndexError:
                        print(manuscript)
                        continue
                    self.categories.add(row[0])
                    for tok in token_list:
                        if tok not in self.token_dic:
                            self.token_dic[tok] = self.seq_no_uid
                            self.seq_no_uid += 1


class DimensionReduction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform(self, vecs: list):
        pass


class NoReduction(DimensionReduction):
    def transform(self, vecs: list):
        pass


class SVD(DimensionReduction):
    def __init__(self, vecs: list, k: int = None):
        if k is None:
            k = len(vecs) - 1
        _, _, self.V = scipy.sparse.linalg.svds(vecs, k=k)

    def transform(self, vecs: np.array) -> np.array:
        return np.dot(vecs, self.V.T)


class LearningDataVectorizer:
    def __init__(self, tuid: TokenUID,
                 dim_red: DimensionReduction = NoReduction()):
        self.tuid = tuid
        self.dim_red = dim_red

    def vectorize(self, news: list, manuscript_min_len: int = 100) -> np.array:
        train_data = []
        append = train_data.append
        cat_list = list(self.tuid.categories)
        cat_len = len(self.tuid.categories)
        max_dim = self.tuid.seq_no_uid + 1
        for line in news:
            wc = int(line[2])
            if wc < manuscript_min_len:
                continue
            category = line[0]
            category_vec = [0] * cat_len
            category_vec[cat_list.index(category)] = 1
            manuscript = line[3]
            try:
                tokens = tokenize(manuscript)
            except IndexError:
                continue
            if tokens is None:
                continue
            tf_vec = self.calc_tf_vec(tokens, max_dim)
            tf_vec = self.dim_red.transform(tf_vec)
            append((category_vec, tf_vec))

        return np.array(train_data)

    def calc_tf_vec(self, tokens: list, max_dim: int) -> np.array:
        '''
         素性に割り振られた連番のユニークIDをもとに
        TFベクトル(Term Frequency)を求める。
        '''
        tf_vec = [0.0] * max_dim
        for tok in tokens:
            uid = self.tuid.token_dic[str(tok)]
            tf_vec[uid] += 1
        return np.array(tf_vec) / len(tokens)


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
    manuscript = filter_manuscript(manuscript)
    tokens = tokenizer.tokenize(manuscript)
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
