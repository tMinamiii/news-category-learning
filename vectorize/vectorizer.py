import csv
import math
import pickle
import random
import re
from collections import Counter

import numpy as np
from janome.tokenizer import Tokenizer
from sklearn.decomposition import IncrementalPCA

import mojimoji


class Preprocessor:
    '''
    ニュースのCSVを読み込んで、学習用データの前処理を施していく。
    ニュース原稿を形態素解析にかけ、Counterを用いて出現頻度を数えておく。
    また各単語のIDFも前処理で計算して連想配列に保存しておく。
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
                    if int(row[2]) < min_manuscript_len:
                        continue
                    filtered = sanitize(row[3])
                    tokens = tokenize(filtered)
                    if tokens is None or len(tokens) < min_token_len:
                        continue
                    # Counterで単語の出現頻度を数える
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
            self.token_to_docid.setdefault(tok, set()).add(self.doc_seq_no)

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
         batch_sizeで指定した数値ずつtokenized_newsを
         主成分分析にかけていく
        '''
        news_len = len(tokenized_news)
        random.shuffle(tokenized_news)
        for i in range(0, news_len, batch_size):
            chunks = tokenized_news[i:i + batch_size]
            mat = np.array([self.tfidf(c[1]) for c in chunks])
            self.ipca.partial_fit(mat)

    def vectorize(self, tokenized_news: list) -> np.array:
        '''
         tokenized_newsのTF-IDFベクトルを求めたのち主成分分析で次元削減する
        '''
        data = []
        for news in tokenized_news:
            category = news[0]
            category_vec = self.prep.category_dic[category]
            tc = news[1]
            tfidf = np.array(self.tfidf(tc)).reshape(1, -1)
            # transformの結果は2重リストになっているので、最初の要素を取り出す
            dimred_tfidf = self.ipca.transform(tfidf)[0]
            data.append((category_vec, dimred_tfidf))
        return np.array(data)

    def tfidf(self, token_counter: Counter) -> list:
        '''
        TF-IDFベクトルを求める。
        '''
        prep = self.prep
        tf_vec = [0.0] * self.max_dim
        total_tokens = sum(token_counter.values())
        for token, count in token_counter.items():
            uid = prep.token_to_id[token]
            tf_vec[uid] = count / total_tokens * prep.idf[token]

        return tf_vec


def dump(dumpdata, filepath: str) -> None:
    with open(filepath, mode='wb') as f:
        pickle.dump(dumpdata, f)
        f.flush()


def load(filepath: str):
    with open(filepath, mode='rb') as f:
        return pickle.load(f)


def sanitize(manu: str) -> str:
    # 英文を取り除く（日本語の中の英字はそのまま）
    manu = re.sub(r'[a-zA-Z0-9]+[ \,\.\':;\-\+?!]', '', manu)
    # 記号や数字は「、」に変換する。
    # (単純に消してしまうと意味不明な長文になりjanomeがエラーを起こす)
    manu = re.sub(r'[0-9０-９]+', '0', manu)
    manu = re.sub(
        r'[!"#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}]+', '、', manu)
    manu = re.sub(r'[“（）【】『』｛｝「」［］《》〈〉]', '、', manu)
    return manu


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
        # 全角英数はすべて半角英数にする
        w = mojimoji.zen_to_han(w, kana=False, digit=False)
        # 半角カタカナはすべて全角にする
        w = mojimoji.han_to_zen(w, digit=False, ascii=False)
        # 英語はすべて小文字にする
        w = w.lower()
        append(w)
    return token_list
