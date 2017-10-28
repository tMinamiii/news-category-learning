import csv
import math
import pickle
import random
import re
from collections import Counter

import numpy as np
from janome.tokenizer import Tokenizer
from sklearn.decomposition import IncrementalPCA

import constant_values as c
import mojimoji


class Token:
    '''
    ニュースのCSVを読み込んで、学習用データに変換する。
    ニュース原稿はTokenizeされ、各単語にUnique IDを割り振る。
    UIDは単なる連番で、このクラスのtoken_dicに登録された順番で決まる。
    '''

    def __init__(self):
        self.loaded_csv_paths = []
        self.token_to_id = {}
        self.id_to_token = {}
        self.categories = set()
        self.category_dic = {}
        self.token_seq_no = 0
        self.doc_seq_no = 0
        self.token_to_docid = {}
        self.tokenized_news = []
        self.idf = {}
        self.token_to_tfidf = {}

    def update(self, csv_paths: list,
               min_manuscript_len: int = c.MINIMUM_MANUSCRIPT_LENGTH,
               min_token_len: int = c.MINIMUM_TOKEN_LENGTH):
        for path in csv_paths:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                self.loaded_csv_paths.append(path)
                for row in reader:
                    if len(row) != 4:
                        continue
                    if int(row[2]) < min_manuscript_len:
                        continue
                    filtered = filter_manuscript(row[3])
                    tokens = tokenize(filtered)
                    if tokens is None or len(tokens) < min_token_len:
                        continue
                    # カウンターオブジェクトにして重複を排除する
                    token_counter = Counter(tokens)
                    self.categories.add(row[0])
                    self.update_token_dics(token_counter)
                    self.tokenized_news.append([row[0], token_counter])
                    self.doc_seq_no += 1
        self.update_idf()
        self.update_category_dic()

    def update_category_dic(self):
        cat_list = list(self.categories)
        cat_len = len(self.categories)
        for cat in cat_list:
            category_vec = [0] * cat_len
            category_vec[cat_list.index(cat)] = 1
            self.category_dic[cat] = category_vec

    def update_token_dics(self, token_counter: dict):
        for tok, _ in token_counter.items():
            if tok not in self.token_to_id:
                self.token_to_id[tok] = self.token_seq_no
                self.id_to_token[self.token_seq_no] = tok
                self.token_seq_no += 1
                if tok in self.token_to_docid:
                    self.token_to_docid[tok].add(self.doc_seq_no)
                else:
                    self.token_to_docid[tok] = set([self.doc_seq_no])

    def update_idf(self) -> float:
        for token, _ in self.token_to_docid.items():
            self.idf[token] = math.log(float(self.doc_seq_no) /
                                       len(self.token_to_docid[token])) + 1


class TfidfVectorizer:
    def __init__(self, tuid: Token, dimension: int):
        self.tuid = tuid
        self.cat_list = list(self.tuid.categories)
        self.cat_len = len(self.tuid.categories)
        self.max_dim = self.tuid.token_seq_no
        self.ipca = IncrementalPCA(n_components=dimension)

    def fit(self, tokenized_news: list,
            batch_size: int = c.PCA_BATCH_DATA_LENGTH) -> None:
        news_len = len(tokenized_news)
        random.shuffle(tokenized_news)
        for i in range(0, news_len, batch_size):
            chunks = tokenized_news[i:i + batch_size]
            mat = [self.tfidf(c[1]) for c in chunks]
            self.ipca.partial_fit(mat)

    def vectorize(self, tokenized_news: list) -> np.array:
        data = []
        for news in tokenized_news:
            category = news[0]
            category_vec = self.tuid.category_dic[category]
            token_counter = news[1]
            tf_vec = self.tfidf(token_counter)
            reshaped = np.array(tf_vec).reshape(1, -1)
            dimred_tfidf = self.ipca.transform(reshaped)[0]
            data.append((category_vec, dimred_tfidf))
        return np.array(data)

    def tfidf(self, token_counter: Counter) -> list:
        '''
         素性に割り振られた連番のユニークIDをもとに
        TFベクトル(Term Frequency)を求める。
        '''
        tuid = self.tuid
        tf_vec = [0.0] * self.max_dim
        total_tokens = sum(token_counter.values())
        for token, count in token_counter.items():
            uid = tuid.token_to_id[token]
            tf_vec[uid] = float(count) / total_tokens * tuid.idf[token]

        return tf_vec


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
        _ps = tok.part_of_speech.split(',')[0]
        if _ps not in ['名詞', '動詞', '形容詞']:
            continue
        # 原形があれば原形をリストに入れる
        _w = tok.base_form
        if _w == '*' or _w == '':
            # 原形がなければ表層系(原稿の単語そのまま)をリストに入れる
            _w = tok.surface
        if _w == '' or _w == '\n':
            continue
        # 全角英数はすべて半角英数にする
        _w = mojimoji.zen_to_han(_w, kana=False, digit=False)
        # 半角カタカナはすべて全角にする
        _w = mojimoji.han_to_zen(_w, digit=False, ascii=False)
        # 英語はすべて小文字にする
        _w = _w.lower()
        append(_w)
    return token_list
