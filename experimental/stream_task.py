import glob

import numpy as np

import constant_values as c
import double_layer_nn as dlnn
import vectorize.learning_data as ld


def find_all_csvs() -> list:
    csv_list = []
    for cat in c.CATEGORIES:
        path = glob.glob('csv/' + cat + '/*.csv')
        csv_list.extend(path)
    return csv_list


def constract(tfidf: ld.TfidfVectorizer, tokenized_news):
    label_and_data = tfidf.vectorize(tokenized_news)
    label = label_and_data[:, 0].tolist()
    data = label_and_data[:, 1].tolist()
    return label, data


def main():
    tuid = ld.load(c.TUID_FILE)
    print('TUID loaded')
    svd = ld.load(c.SVD_FILE)
    print('SVD loaded')
    tfidf = ld.TfidfVectorizer(tuid, svd)
    print('Vectorizer initialized')
    num_categories = len(tuid.categories)
    tokenized_news = tuid.tokenized_news
    all_news_len = len(tokenized_news)

    boundary = int(all_news_len * c.TRAINING_DATA_RATIO)
    test_csv = tokenized_news[boundary:]
    train_csv = tokenized_news[0:boundary - 1]
    print(len(test_csv))
    print(len(train_csv))

    test_label, test_data = constract(tfidf, test_csv)
    print('TEST DATA calculated')
    nn = dlnn.DoubleLayerNetwork(c.LEARNING_RATIO, c.NUM_UNITS,
                                 c.SVD_DIMENSION, num_categories,
                                 c.LOG_FILE)
    i = 0
    for _ in range(c.TOTAL_STEP):
        i += 1

        np.random.shuffle(train_csv)
        batch_train = train_csv[:c.BATCH_SIZE]
        train_label, train_data = constract(tfidf, batch_train)

        nn.sess.run(nn.train_step, feed_dict={
            nn.x: train_data, nn.t: train_label, nn.keep_prob: 0.5})
        if i % 100 == 0:
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: test_data,
                           nn.t: test_label,
                           nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            nn.writer.add_summary(summary, i)


if __name__ == '__main__':
    main()
