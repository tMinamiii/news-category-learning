import os
import pickle
import json
import csv
import glob


NUM_UNITS = 2000
DATA_FILE = './data/vector/tfidf/tfidf.data'
METADATA_FILE = './data/vector/tfidf/tfidf.meta'
LOG_FILE = '/tmp/yn_categories_logs'
BATCH_SIZE = 500
TOTAL_STEP = 1000000
LEARNING_RATE = 0.0001  # 学習率
TRAINING_DATA_RATIO = 0.8  # 全データのうち訓練用に使う割合
MINIMUM_MANUSCRIPT_LENGTH = 300
MINIMUM_TOKEN_LENGTH = 150
KEEP_PROB = 0.5
# smaller equal than BATCH_SIZE
# 最大にしたい場合はNoneを設定する
PCA_DIMENSION = 100
CATEGORIES = ['IT総合', '映画', '経済総合', '野球',
              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']

# CATEGORIES = ['映画', '経済総合', '野球',
#              '社会', 'ライフ総合', 'エンタメ総合', ]
# CATEGORIES = ['IT総合', '映画']
PCA_BATCH_DATA_LENGTH = 150


def pickle_load(filepath) -> None:
    with open(filepath, mode='rb') as f:
        data = pickle.load(f)
    return data


def pickle_dump(dumpdata, filepath: str) -> None:
    with open(filepath, mode='wb') as f:
        pickle.dump(dumpdata, f)
        f.flush()


def extract_category(path):
    basename = os.path.basename(path)
    category, _ = os.path.splitext(basename)
    return category


def find_and_load_news(filetype, time=None):
    all_paths = []
    for cat in CATEGORIES:
        if time is not None:
            timestr = time.strftime('%Y-%m-%d')
            regex = './data/{0}/{1}/*_{2}.{0}'.format(filetype, cat, timestr)
        else:
            regex = './data/{0}/{1}/*.{0}'.format(filetype, cat)
        all_paths += glob.glob(regex)
        print(all_paths)
    all_chunks = []
    for path in all_paths:
        with open(path, 'r') as f:
            if filetype == 'json':
                chunk = json.load(f)
            elif filetype == 'csv':
                chunk = []
                for line in csv.reader(f):
                    line_dic = {'category': line[0],
                                'title': line[1],
                                'manuscript_len': line[2],
                                'manuscript': line[3]}
                    chunk.append(line_dic)
        all_chunks += chunk
    return all_chunks
