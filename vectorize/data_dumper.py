import csv
import datetime
import glob
import random

import vectorize.learning_data as ld

MANUSCRIPT_MINIMUM_LENGTH = 300
SVD_DATA_LENGTH = 5000
# SVD_DIMENSION < SVD_DATA_LENGTH
# 最大にしたい場合はNoneを設定する
SVD_DIMENSION = 500
CATEGORIES = ['IT総合', '映画', '経済総合', '野球',
              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']
# CATEGORIES = ['IT総合', '映画']


def find_all_csvs() -> list:
    csv_list = []
    for cat in CATEGORIES:
        path = glob.glob('csv/' + cat + '/*.csv')
        csv_list.extend(path)
    return csv_list


def load_all_csvs(csv_list: list) -> list:
    all_news = []
    for path in csv_list:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                all_news.append(row)
    random.shuffle(all_news)
    print('All news were loaded. total: ' + str(len(all_news)))
    return all_news


def calc_svd(tuid: ld.TokenUID, all_news: list) -> ld.SparseSVD:
    svd_news = all_news[0:SVD_DATA_LENGTH]
    ldata = ld.TermFrequencyVectorizer(tuid)
    svd_data = ldata.vectorize(svd_news, MANUSCRIPT_MINIMUM_LENGTH)
    vecs = svd_data[:, 1].tolist()
    svd = ld.SparseSVD(vecs, SVD_DIMENSION)
    print('SVD was finished.')
    return svd


def dump_all_csv(tuid: ld.TokenUID,
                 svd_ldata: ld.TermFrequencyVectorizer, all_news: list):

    current_time = datetime.datetime.now()
    output_name = current_time.strftime('%Y-%m-%d')
    print('TUID data was dumped.')
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')
    # td = ldata.make(tuid, all_news)
    td = svd_ldata.vectorize(all_news, MANUSCRIPT_MINIMUM_LENGTH)
    # ld.dump(td, 'ldata/' + output_name + '.td')
    ld.dump(td, 'ldata/' + output_name + '.svdtd')
    print('Learning data was dumped.')


def main():
    csv_list = find_all_csvs()
    all_news = load_all_csvs(csv_list)
    tuid = ld.TokenUID()
    tuid.update(csv_list)
    svd = calc_svd(tuid, all_news)
    svd_ldata = ld.TermFrequencyVectorizer(tuid, dim_red=svd)
    dump_all_csv(tuid, svd_ldata, all_news)


if __name__ == '__main__':
    main()
