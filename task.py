import numpy as np
import double_layer_nn as dlnn
import vectorize.learning_data as ld

NUM_UNITS = 1000
DATA_FILE = 'ldata/2017-10-20.svdtd'
TUID_FILE = 'ldata/2017-10-20.tuid'
LOG_FILE = '/tmp/yn_categories_logs'
BATCH_SIZE = 100
TOTAL_STEP = 1000000
LEARNING_RATIO = 0.005  # 学習率
TRAINING_DATA_RATIO = 0.9  # 全データのうち訓練用に使う割合


def main():
    tuid = ld.load(TUID_FILE)
    td = ld.load(DATA_FILE)
    np.random.shuffle(td)
    tdlen = len(td)
    boundary = int(tdlen * TRAINING_DATA_RATIO)
    pd = td[boundary:tdlen - 1]
    td = td[0:boundary - 1]
    print(len(pd))
    print(len(td))
    vec_dim = len(td[:, 1][0])
    num_categories = len(tuid.categories)
    predict_label = pd[:, 0].tolist()
    predict_data = pd[:, 1].tolist()

    nn = dlnn.DoubleLayerNetwork(LEARNING_RATIO, NUM_UNITS,
                                 vec_dim, num_categories, LOG_FILE)
    i = 0
    for _ in range(TOTAL_STEP):
        i += 1
        np.random.shuffle(td)
        batch_label = td[:BATCH_SIZE, 0].tolist()
        batch_data = td[:BATCH_SIZE, 1].tolist()

        nn.sess.run(nn.train_step, feed_dict={
            nn.x: batch_data, nn.t: batch_label, nn.keep_prob: 0.5})
        if i % 100 == 0:
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: predict_data,
                           nn.t: predict_label,
                           nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            nn.writer.add_summary(summary, i)


if __name__ == '__main__':
    main()
