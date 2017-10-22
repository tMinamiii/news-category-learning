import datetime
import glob

import vectorize.learning_data as ld
import constant_values as c


def find_all_csvs() -> list:
    csv_list = []
    for cat in c.CATEGORIES:
        path = glob.glob('csv/' + cat + '/*.csv')
        csv_list.extend(path)
    return csv_list


def calc_svd(tuid: ld.Token) -> ld.SparseSVD:
    svd_news = tuid.tokenized_news[0:c.SVD_DATA_LENGTH]
    ldata = ld.TfidfVectorizer(tuid)
    svd_data = ldata.vectorize(svd_news)
    vecs = svd_data[:, 1].tolist()
    svd = ld.SparseSVD(vecs, c.SVD_DIMENSION)
    print('SVD was finished.')
    return svd


def dump_all_csv(tuid: ld.Token,
                 svd_ldata: ld.TfidfVectorizer):

    current_time = datetime.datetime.now()
    output_name = current_time.strftime('%Y-%m-%d')
    print('TUID data was dumped.')
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')
    # td = ldata.make(tuid, all_news)
    td = svd_ldata.vectorize(tuid.tokenized_news)
    # ld.dump(td, 'ldata/' + output_name + '.td')
    ld.dump(td, 'ldata/' + output_name + '.svdtd')
    print('Learning data was dumped.')


def dump_tuid(tuid: ld.Token):
    current_time = datetime.datetime.now()
    output_name = current_time.strftime('%Y-%m-%d')
    print('TUID data was dumped.')
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')


def main():
    csv_list = find_all_csvs()
    tuid = ld.Token(c.MINIMUM_MANUSCRIPT_LENGTH,
                    c.MINIMUM_TOKEN_LENGTH)
    tuid.update(csv_list)
    # dump_tuid(tuid)
    svd = calc_svd(tuid)
    svd_ldata = ld.TfidfVectorizer(tuid, dim_red=svd)
    dump_all_csv(tuid, svd_ldata)


if __name__ == '__main__':
    main()
