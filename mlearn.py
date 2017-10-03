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

    def convert2uid(self, manuscript_tokens: list):
        manuscript_token_uid_list = []
        for tok in manuscript_tokens:
            self.update(str)
            manuscript_token_uid_list.append(self.token_dic[tok])
        return manuscript_token_uid_list

    def update(self, manuscript_tokens: list):
        for tok in manuscript_tokens:
            if not tok in self.token_dic:
                self.token_dic[tok] = self.seq_no
                self.seq_no += 1


class NewsCategoryLearning:
    def __init__(self):
        self.tdict = TokenDict()

    def initialize(self, csv_list: str):
        self.all_news = self.read_csv_data(csv_list)
        self.learning_data = []
        for news in self.all_news:
            category = news[0]
            vec_list = self.token_uid_list_2_vec_list(
                news[1], self.tdict.seq_no)
            tf_vec = self.calc_norm_tf_vector(vec_list)
            self.learning_data.append((category, tf_vec))

    def read_csv_data(self, csv_list: str, wc_lower=200):
        '''
         ニュースのCSVを読み込んで、学習用データに変換する。
        ニュース原稿はTokenizeされ、更にuidのリスト(単なる数字のリスト)に変換される。
        ベクトルの最大次元数は、全データを読み込まないとわからないので、ここではまだTFベクトルに変換しない。
        '''
        all_news = []
        for csv_path in csv_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    word_count = row[2]
                    # 原稿の文字数が200文字以下なら飛ばす
                    if word_count < wc_lower:
                        continue
                    category = row[0]
                    manuscript = row[3]
                    token_list = self.tokenize(manuscript)
                    self.tdict.update(token_list)
                    token_uid_list = self.tdict.convert2uid(token_list)
                    all_news.append((category, token_uid_list))
        return all_news

    def tokenize(self, manuscript: str):
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

    def token_uid_list_2_vec_list(self, token_uid_list: list, max_dim: int):
        '''
         形態素に割り振られた連番のユニークIDから基底ベクトルを作成する。
        形態素の'Python'のユニークIDが「3」、次元数が「5」
        のときは、[0, 0, 0, 1, 0] というベクトルになる
        '''
        vec_list = []
        for uid in token_uid_list:
            vec = np.zeros(max_dim)
            vec[uid] = 1
            vec_list.append(vec)
        return vec_list

    def calc_norm_tf_vector(self, manuscript_vecs: list) -> np.array:
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
