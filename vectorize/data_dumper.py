import datetime
import glob

import numpy as np

import constant_values as c
import vectorize.learning_data as ld


def find_all_csvs() -> list:
    csv_list = []
    for cat in c.CATEGORIES:
        path = glob.glob('csv/' + cat + '/*.csv')
        csv_list.extend(path)
    return csv_list


def dump_data(tuid: ld.Token, td: np.array):
    current_time = datetime.datetime.now()
    output_name = current_time.strftime('%Y-%m-%d')
    print('TUID data was dumped.')
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')
    ld.dump(td, 'ldata/' + output_name + '.td')
    print('Learning data was dumped.')


def main():
    csv_list = find_all_csvs()
    tuid = ld.Token()
    tuid.update(csv_list)
    print('TUID calculated')
    ldata = ld.TfidfVectorizer(tuid, c.SVD_DIMENSION)
    ldata.fit(tuid.tokenized_news)
    print('IncrementPCA fitting finished')
    td = ldata.vectorize(tuid.tokenized_news)
    print('vectorizing finished')
    dump_data(tuid, td)


if __name__ == '__main__':
    main()
