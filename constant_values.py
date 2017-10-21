NUM_UNITS = 1000
DATA_FILE = 'ldata/2017-10-21.svdtd'
TUID_FILE = 'ldata/2017-10-21.tuid'
LOG_FILE = '/tmp/yn_categories_logs'
BATCH_SIZE = 200
TOTAL_STEP = 1000000
LEARNING_RATIO = 0.005  # 学習率
TRAINING_DATA_RATIO = 0.9  # 全データのうち訓練用に使う割合
MINIMUM_MANUSCRIPT_LENGTH = 300
MINIMUM_TOKEN_LENGTH = 150
# smaller equal than BATCH_SIZE
# 最大にしたい場合はNoneを設定する
SVD_DIMENSION = 150

CATEGORIES = ['IT総合', '映画', '経済総合', '野球',
              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']
# CATEGORIES = ['IT総合', '映画']

SVD_DATA_LENGTH = 5000