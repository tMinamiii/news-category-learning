from janome.tokenizer import Tokenizer
import csv
import numpy as np
import pickle
import re


class TokenUID:
    def __init__(self):
        self.loaded_csv_list = []
        self.token_dic = {}
        self.categories = set()
        self.seq_no_uid = 0
        '''
         ニュースのCSVを読み込んで、学習用データに変換する。
        ニュース原稿はTokenizeされ、更にuidのリスト(単なる数字のリスト)に変換される。
        ベクトルの最大次元数は、全データを読み込まないとわからないので、ここではまだTFベクトルに変換しない。
        '''

    def update(self, csv_list: list):
        for csv_path in csv_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                self.loaded_csv_list.append(csv_path)
                for row in reader:
                    if len(row) != 4:
                        continue
                    self.categories.add(row[0])
                    manuscript = row[3]
                    try:
                        token_list = tokenize(manuscript)
                    except IndexError:
                        print(row)

                    for tok in token_list:
                        if not tok in self.token_dic:
                            self.token_dic[tok] = self.seq_no_uid
                            self.seq_no_uid += 1


class LearningData:
    def __init__(self, token_uid: TokenUID):
        self.token_uid = token_uid
        self.train_data = []
        self.predict_data = []

    def make(self, ratio_of_train=10, wc_lower: int=200):
        train_data = []
        predict_data = []

        count = 1
        for csv_path in self.token_uid.loaded_csv_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    wc = int(line[2])
                    if wc < wc_lower:
                        continue
                    category = line[0]
                    category_vec = np.zeros(
                        len(self.token_uid.categories))
                    cat_list = list(self.token_uid.categories)
                    category_vec[cat_list.index(category)] += 1
                    manuscript = line[3]
                    token_list = tokenize(manuscript)
                    uid_list = self.token_list_2_tuid_list(token_list)
                    vec_list = self.tuid_list_2_vec_list(
                        uid_list, self.token_uid.seq_no_uid + 1)
                    tf_vec = self.calc_norm_tf_vector(vec_list)
                    if count <= ratio_of_train:
                        train_data.append((category_vec, tf_vec))
                    else:
                        predict_data.append((category_vec, tf_vec))

                    if count >= 10:
                        count = 0
                    else:
                        count += 1
        self.train_data = np.array(train_data)
        self.predict_data = np.array(predict_data)

    def token_list_2_tuid_list(self, manuscript_tokens: list)-> list:
        manuscript_token_uid_list = []
        for tok in manuscript_tokens:
            manuscript_token_uid_list.append(
                self.token_uid.token_dic[str(tok)])
        return manuscript_token_uid_list

    def tuid_list_2_vec_list(self, token_uid_list: list, max_dim: int) -> list:
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

    def calc_norm_tf_vector(self, manuscript_vecs: list) -> list:
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


def dump(dumpdata, filepath: str) -> None:
    with open(filepath, mode='wb') as f:
        pickle.dump(dumpdata, f)
        f.flush()


def load(filepath: str):
    with open(filepath, mode='rb') as f:
        return pickle.load(f)


def tokenize(manuscript: str) -> list:
    token_list = []
    tokenizer = Tokenizer()
    manuscript = re.sub(r'[0-9\@\"\,\.]+', '', manuscript)
    manuscript = re.sub(
        r'[!"“#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}~]', '', manuscript)
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
