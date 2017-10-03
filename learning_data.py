from janome.tokenizer import Tokenizer
import csv
import numpy as np
import pickle


class TokenUID:
    def __init__(self, csv_list: list, token_dic=None):
        self.csv_list = csv_list
        if token_dic is None:
            token_dic = {}
        self.token_dic = token_dic
        self.seq_no_uid = len(token_dic)

    def update(self, manuscript_tokens: list):
        for tok in manuscript_tokens:
            if not tok in self.token_dic:
                self.token_dic[tok] = self.seq_no_uid
                self.seq_no_uid += 1

    def dump(self, filepath: str) -> None:
        with open(filepath, mode='wb') as f:
            pickle.dump(self.token_dic, f)
        # csvリストもdumpする
        # オブジェクトごとdumpすればよいのでは？

    def load(self, filepath: str) -> None:
        with open(filepath, mode='rb') as f:
            loaded_dic = pickle.load(f)
            self.token_dic = loaded_dic
            self.seq_no_uid = len(loaded_dic)
        # csvリストもloadする

    def initialize(self) -> list:
        '''
         ニュースのCSVを読み込んで、学習用データに変換する。
        ニュース原稿はTokenizeされ、更にuidのリスト(単なる数字のリスト)に変換される。
        ベクトルの最大次元数は、全データを読み込まないとわからないので、ここではまだTFベクトルに変換しない。
        '''
        for csv_path in self.csv_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    manuscript = row[3]
                    token_list = tokenize(manuscript)
                    self.update(token_list)


class LearningData:
    def __init__(self, token_uid: TokenUID):
        self.token_uid = token_uid

    def make(self, wc_lower: int=200) -> list:
        learning_data = []
        for csv_path in self.token_uid.csv_list:
            data = self.read_csv(csv_path, wc_lower)
            learning_data.extend(data)
        return learning_data

    def read_csv(self, csv_path: str, wc_lower: int=200) -> list:
        learning_data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                wc = int(line[2])
                if wc < wc_lower:
                    continue
                category = line[0]
                manuscript = line[3]
                token_list = tokenize(manuscript)
                uid_list = self.token_2_uid(token_list)
                vec_list = self.token_uid_list_2_vec_list(
                    uid_list, self.token_uid.seq_no_uid)
                tf_vec = self.calc_norm_tf_vector(vec_list)
                learning_data.append((category, tf_vec))
        return learning_data

    def token_2_uid(self, manuscript_tokens: list)-> list:
        manuscript_token_uid_list = []
        for tok in manuscript_tokens:
            manuscript_token_uid_list.append(self.token_uid.token_dic[tok])
        return manuscript_token_uid_list

    def token_uid_list_2_vec_list(self, token_uid_list: list, max_dim: int) -> list:
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
        # total = np.zeros(self.td.seq_no_uid)
        total = None
        for vec in manuscript_vecs:
            if total is None:
                total = vec
            else:
                total += vec
        total /= len(manuscript_vecs)
        return total


def tokenize(manuscript: str) -> list:
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
