from janome.tokenizer import Tokenizer
import os
import csv
import numpy as np
import tensorflow as tf
import pickle


class TokenDict:
    def __init__(self):
        self.token_dic = {}
        self.seq_no = 0

    def dump(self, filepath: str):
        with open(filepath, mode='wb') as f:
            pickle.dump(self.token_dic, f)

    def load(self, filepath):
        with open(filepath, mode='rb') as f:
            self.token_dic = pickle.load(f)
            self.seq_no = len(self.token_dic)

    def add(self, token):
        if not token in self.token_dic:
            self.token_dic[token] = self.seq_no
            self.seq_no += 1


def tokenize(manuscript: str):
    token_list = []
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(manuscript)
    for tok in tokens:
        ps = tok.part_of_speech.split(',')[0]
        if not ps in ['名詞', '動詞', '形容詞']:
            continue
        # 原形があれば原形をリストに入れる
        w = tok.base_form
        if w == '*' or w == '':
            # 原形がなければ表層系(原稿の単語そのまま)をリストに入れる
            w = tok.surface
        if w == '' or w == '\n':
            continue
        token_list.append(w)
    return token_list


def scalar_to_vec(uniq_id, max_dim) -> np.array:
    '''
     形態素に割り振られた連番のユニークIDから基底ベクトルを作成する。
     形態素の'Python'のユニークIDが「3」、次元数が「5」
     のときは、[0, 0, 0, 1, 0] というベクトルになる
    '''
    vec = np.zeros(max_dim)
    vec[uniq_id] = 1
    return vec


def calc_norm_tf_vector(manuscript_vecs: list) -> np.array:
    '''
    1原稿分の形態素のベクトルの総和を求めて、TFベクトル(Term Frequency)を求める。
    総和したTFベクトルは、1原稿の単語数で割り算することで平準化する。
    平準化することで、長文にある1単語より、短文の1単語のほうが特徴に重みが付く。
    ※ただし、平準化は実験的な手法なので、もっといい方法があれば変更する
    '''
    total = np.array(0)
    for vec in manuscript_vecs:
        total += vec
    total /= len(manuscript_vecs)
    return total
