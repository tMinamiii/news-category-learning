import glob
import learning_data as ld
import datetime
import csv
import random


MANUSCRIPT_MINIMUM_LENGTH = 400
SVD_DIMENSION = 2000
CATEGORIES = ['IT総合', '映画', '経済総合', '野球',
              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']
# CATEGORIES = ['IT総合']


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
    print('all news loaded')
    return all_news


def calc_svd(tuid: ld.TokenUID, all_news: list) -> ld.SVD:
    svd_news = all_news[0:SVD_DIMENSION]
    ldata = ld.LearningData()
    svd_data = ldata.make(tuid, svd_news, MANUSCRIPT_MINIMUM_LENGTH)
    print('created training data for svd')
    return ld.SVD(svd_data[:, 1].tolist())


def dump_all_csv(svd_ldata: ld.LearningData, all_news: list):
    current_time = datetime.datetime.now()
    output_name = current_time.strftime('%Y-%m-%d')
    print('dumping tuid data...')
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')
    print('making train data...')
    # td = ldata.make(tuid, all_news)
    td = svd_ldata.make(tuid, all_news, MANUSCRIPT_MINIMUM_LENGTH)
    print('dumping train data...')
    # ld.dump(td, 'ldata/' + output_name + '.td')
    ld.dump(td, 'ldata/' + output_name + '.svdtd')


if __name__ == '__main__':
    csv_list = find_all_csvs()
    all_news = load_all_csvs(csv_list)
    tuid = ld.TokenUID()
    tuid.update(csv_list)
    svd = calc_svd(tuid, all_news)
    svd_ldata = ld.LearningData(svd)
    dump_all_csv(svd_ldata, all_news)
