import datetime
import glob
import os
import numpy as np

import constant_values as c
import vectorize.vectorizer as ld


def find_all_csvs() -> list:
    csv_list = []
    for cat in c.CATEGORIES:
        path = glob.glob('csv/' + cat + '/*.csv')
        csv_list.extend(path)
    return csv_list


def dump_data(prep: ld.Preprocessor, data_for_learn: np.array):
    current_time = datetime.datetime.now()
    output_name = current_time.strftime('%Y-%m-%d')
    if not os.path.isdir('ldata'):
        os.mkdir('ldata')
    ld.dump(prep, 'ldata/' + output_name + '.prep')
    print('Preprocessed data was dumped.')
    ld.dump(data_for_learn, 'ldata/' + output_name + '.td')
    print('Learning data was dumped.')


def main():
    csv_list = find_all_csvs()
    prep = ld.Preprocessor()
    prep.append(csv_list, min_manuscript_len=c.MINIMUM_MANUSCRIPT_LENGTH,
                min_token_len=c.MINIMUM_TOKEN_LENGTH)
    print('TUID calculated')
    vectorizer = ld.PCATfidfVectorizer(prep, c.PCA_DIMENSION)
    vectorizer.fit(prep.tokenized_news, c.PCA_BATCH_DATA_LENGTH)
    print('IncrementPCA fitting finished')
    data_for_learn = vectorizer.vectorize(prep.tokenized_news)
    print('vectorizing finished')
    dump_data(prep, data_for_learn)


if __name__ == '__main__':
    main()
